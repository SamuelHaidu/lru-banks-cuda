#pragma once
// Training batches from a memory-mapped uint16 .npy token file (`<bos>doc<eos>` documents concatenated), as the
// Python pipeline's random windows: each pass cuts the stream into windows of T + 1 tokens from a random offset and
// visits them in a shuffled order. Batch g of the run is batch g % steps_per_pass of pass g / steps_per_pass, so a
// resumed run continues with exactly the next batch. Targets are the next tokens, -1 where the next token is the
// document marker; the state resets wherever the input token is the marker. A CPU thread fills pinned buffers ahead
// of the GPU (the disk reads happen there, through the page cache).

#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <cuda_runtime.h>

namespace lbc {

class TokenFile {
  public:
    explicit TokenFile(const std::string& path);
    ~TokenFile();
    TokenFile(const TokenFile&) = delete;
    TokenFile& operator=(const TokenFile&) = delete;
    const uint16_t* data() const { return data_; }
    size_t size() const { return n_; }

  private:
    void* map_ = nullptr;
    size_t map_bytes_ = 0;
    const uint16_t* data_ = nullptr;
    size_t n_ = 0;
};

struct HostBatch {
    int* ids = nullptr;        // [B * T]
    int* y = nullptr;          // [B * T]
    uint8_t* reset = nullptr;  // [B * T]
    float* inv_n = nullptr;    // [1] 1 / targets (pinned, copied with the batch)
    int n_targets = 0;
};

class Loader {
  public:
    // score_after >= 0: only the targets after that token (within its document) are scored (e.g. a copy task's
    // separator: the copied half); -1: every target
    Loader(const TokenFile& file, int B, int T, uint64_t seed, int marker = 0, uint64_t start_batch = 0,
           int score_after = -1, int slots = 3);
    // batches of B windows in one pass (the windows that fit at any offset: (n - T) / T), as the Python sampler
    static size_t steps_per_pass(size_t tokens, int B, int T) { return tokens > size_t(T) ? (tokens - T) / T / B : 0; }
    ~Loader();
    Loader(const Loader&) = delete;
    Loader& operator=(const Loader&) = delete;

    HostBatch* next();                            // blocks until the next batch is ready
    void release(HostBatch* hb, cudaStream_t stream);  // call after queuing its copies on `stream`: refilled once done
    double wait_ms() const { return wait_ms_; }   // total time next() has waited
    double fill_ms() const { return fill_ms_; }   // total time spent filling (the loader thread)

  private:
    void run();
    void fill(HostBatch& hb);

    const TokenFile& file_;
    int B_, T_, marker_, score_after_;
    uint64_t seed_;
    uint64_t next_batch_, pass_ = ~uint64_t(0), steps_per_pass_;
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

// Validation rows, the same every epoch (as the Python pipeline's dense validation, `fixed` rows): `steps` batches of
// B rows of T tokens, each row from a random document start (seeded per batch); past the end of the file the rows
// are padded with the marker (no target). Built once into pinned memory.
class ValidationSet {
  public:
    ValidationSet(const TokenFile& file, int B, int T, int steps, uint64_t seed, int marker = 0, int score_after = -1);
    ~ValidationSet();
    ValidationSet(const ValidationSet&) = delete;
    ValidationSet& operator=(const ValidationSet&) = delete;
    int steps() const { return int(batches_.size()); }
    const HostBatch& batch(int i) const { return batches_[size_t(i)]; }

  private:
    std::vector<HostBatch> batches_;
};

}  // namespace lbc
