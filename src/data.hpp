#pragma once
// Training batches from a memory-mapped uint16 .npy token file (`<bos>doc<eos>` documents concatenated), as the
// Python pipeline's random windows: each pass cuts the stream into windows of T + 1 tokens from a random offset and
// visits them in a shuffled order. Targets are the next tokens, -1 where the next token is the document marker; the
// state resets wherever the input token is the marker. A CPU thread fills pinned buffers ahead of the GPU (the disk
// reads happen there, through the page cache).

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
    Loader(const TokenFile& file, int B, int T, uint64_t seed, int marker = 0, int slots = 3);
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
    int B_, T_, marker_;
    uint64_t seed_;
    uint64_t pass_ = 0;
    size_t pos_ = 0, offset_ = 0;
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

}  // namespace lbc
