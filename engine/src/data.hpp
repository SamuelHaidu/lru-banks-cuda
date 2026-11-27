#pragma once
// Training batches from a memory-mapped uint16 .npy token file (`<bos>doc<eos>` documents concatenated), as the
// Python pipeline's random windows: each pass cuts the stream into windows of T + 1 tokens from a random offset and
// visits them in a shuffled order. Batch g of the run is batch g % steps_per_pass of pass g / steps_per_pass, so a
// resumed run continues with exactly the next batch. Targets are the next tokens, -1 where the next token is the
// document marker; the state resets wherever the input token is the marker. A CPU thread fills pinned buffers ahead
// of the GPU (the disk reads happen there, through the page cache).

#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <cuda_runtime.h>

namespace lbc {

class TokenFile {
  public:
    explicit TokenFile(const std::string& path);
    ~TokenFile();
    TokenFile(const TokenFile&) = delete;
    TokenFile& operator=(const TokenFile&) = delete;
    TokenFile(TokenFile&&) = delete;
    TokenFile& operator=(TokenFile&&) = delete;

    const uint16_t* data() const { return data_; }

    size_t size() const { return n_; }

  private:
    void* map_ = nullptr;
    size_t map_bytes_ = 0;
    const uint16_t* data_ = nullptr;
    size_t n_ = 0;
};

struct Span {  // a dataset's loss span (loss_spans [[open, close]]): -1, no span (every target)
    int open = -1, close = -1;
};

struct HostBatch {
    int* ids = nullptr;        // [B * T]
    int* y = nullptr;          // [B * T]
    uint8_t* reset = nullptr;  // [B * T]
    float* inv_n = nullptr;    // [1] 1 / targets (pinned, copied with the batch)
    int n_targets = 0;
    // unlikelihood training (Loader nc > 0): [B * T][nc] per position the most recent distinct tokens of its
    // document so far in the row (the input token included, the marker and the target excluded), -1 padded
    int* cand = nullptr;
};

// A row's unlikelihood candidates (HostBatch::cand) from its ids / targets / resets (resets clear the list)
void fill_candidates(const int* ids, const int* y, const uint8_t* reset, int T, int nc, int marker, int* cand);

class Loader {
  public:
    // score_after >= 0: only the targets after that token (within its document) are scored (e.g. a copy task's
    // separator: the copied half); -1: every target
    // score_prev >= 0: only targets whose input token follows that token (a query marker) are scored
    // aligned: the file is made of whole rows of T tokens (documents never cross a row boundary): every pass cuts the
    // windows at offset 0 instead of a random one
    // stream_chunks K > 1 (truncated BPTT): each pass cuts windows of K T tokens instead; B of them make a stream of
    // K batches, batch k holding their k-th T tokens, so row b of a batch continues row b of the previous one. A
    // stream's first batch, and the first batch this loader produces (a start or a resume: the model holds no
    // carried state), mark every row's first token as a reset, so the row starts from an empty state.
    Loader(const TokenFile& file,
           int B,
           int T,
           uint64_t seed,
           int marker = 0,
           uint64_t start_batch = 0,
           int score_after = -1,
           int score_prev = -1,
           bool aligned = false,
           int slots = 3,
           Span span = {},
           int stream_chunks = 1,
           int nc = 0);  // unlikelihood candidates per position (0: none)

    int nc() const { return nc_; }

    // batches of B windows in one pass (the windows that fit at any offset: (n - T) / T), as the Python sampler;
    // with streams of K batches, whole streams only
    static size_t steps_per_pass(size_t tokens, int B, int T, int K = 1) {
        const size_t W = size_t(T) * size_t(K);
        return tokens > W ? (tokens - W) / W / size_t(B) * size_t(K) : 0;
    }

    ~Loader();
    Loader(const Loader&) = delete;
    Loader& operator=(const Loader&) = delete;
    Loader(Loader&&) = delete;
    Loader& operator=(Loader&&) = delete;

    HostBatch* next();                                 // blocks until the next batch is ready
    void release(HostBatch* hb, cudaStream_t stream);  // call after queuing its copies on `stream`: refilled once done

    double wait_ms() const { return wait_ms_; }  // total time next() has waited

    double fill_ms() const { return fill_ms_; }  // total time spent filling (the loader thread)

  private:
    void run();
    void fill(HostBatch& hb);

    const TokenFile& file_;
    int B_, T_, marker_, score_after_, score_prev_;
    Span span_;
    bool aligned_;
    uint64_t seed_;
    int stream_;  // batches per stream (1: independent windows)
    int nc_;      // unlikelihood candidates per position
    uint64_t next_batch_, first_batch_, pass_ = ~uint64_t(0), steps_per_pass_;
    size_t offset_ = 0;
    std::vector<uint32_t> order_;

    struct Slot {
        HostBatch hb;
        cudaEvent_t done = nullptr;
        bool ready = false, pending = false;
    };

    std::vector<Slot> slots_;
    size_t produce_ = 0, consume_ = 0;
    std::mutex mu_;
    std::condition_variable cv_;
    bool stop_ = false;
    std::thread thread_;
    double wait_ms_ = 0.0, fill_ms_ = 0.0;
};

// Batches mixing several token files at fixed row counts: every batch holds rows[i] rows of source i, each source
// with its own Loader (its own passes and shuffled order, so a source's sampling is the same as alone), so the mix
// is exactly the same at every step. inv_n: 1 / the whole batch's targets. With one source it is that Loader.
class MixLoader {
  public:
    MixLoader(std::vector<std::unique_ptr<Loader>> sources, std::vector<int> rows, int T);
    ~MixLoader();
    MixLoader(const MixLoader&) = delete;
    MixLoader& operator=(const MixLoader&) = delete;
    MixLoader(MixLoader&&) = delete;
    MixLoader& operator=(MixLoader&&) = delete;
    HostBatch* next();                                 // blocks until every source's next batch is ready
    void release(HostBatch* hb, cudaStream_t stream);  // after queuing its copies on `stream`

    const std::vector<long long>& targets() const { return targets_; }  // scored targets per source so far

  private:
    std::vector<std::unique_ptr<Loader>> src_;
    std::vector<int> rows_;
    int T_, B_ = 0, nc_ = 0;
    HostBatch buf_[2];
    cudaEvent_t done_[2] = {nullptr, nullptr};
    bool recorded_[2] = {false, false};
    int k_ = 0;
    cudaStream_t side_ = nullptr;
    std::vector<long long> targets_;
};

// Validation rows, the same every epoch (as the Python pipeline's dense validation, `fixed` rows): `steps` batches of
// B rows of T tokens, each row from a random document start (seeded per batch); past the end of the file the rows
// are padded with the marker (no target). Built once into pinned memory.
class ValidationSet {
  public:
    // aligned: rows start at multiples of T (whole packed rows) instead of at random document starts
    ValidationSet(const TokenFile& file,
                  int B,
                  int T,
                  int steps,
                  uint64_t seed,
                  int marker = 0,
                  int score_after = -1,
                  int score_prev = -1,
                  bool aligned = false,
                  Span span = {});
    ~ValidationSet();
    ValidationSet(const ValidationSet&) = delete;
    ValidationSet& operator=(const ValidationSet&) = delete;
    ValidationSet(ValidationSet&&) = delete;
    ValidationSet& operator=(ValidationSet&&) = delete;

    int steps() const { return int(batches_.size()); }

    const HostBatch& batch(int i) const { return batches_[size_t(i)]; }

  private:
    std::vector<HostBatch> batches_;
};

}  // namespace lbc
