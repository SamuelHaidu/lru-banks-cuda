#pragma once
// Microbenchmark timing: warm-up runs, then CUDA events around each timed run (median / min / mean in ms).
// With an L2Flusher the L2 is overwritten before every run, outside the timed region: cold-cache numbers. Our
// operands often fit in L2 (72 MB on the RTX 4090), so warm and cold can differ a lot: report which one.

#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <numeric>
#include <vector>

#include <cuda_runtime.h>

#include "lbc/check.hpp"

namespace lbc {

struct Stats {
    double median_ms;
    double min_ms;
    double mean_ms;
    int runs;
};

class L2Flusher {
  public:
    L2Flusher() {
        int dev = 0, l2 = 0;
        LBC_CUDA(cudaGetDevice(&dev));
        LBC_CUDA(cudaDeviceGetAttribute(&l2, cudaDevAttrL2CacheSize, dev));
        bytes_ = static_cast<std::size_t>(l2) * 2;  // twice the L2: nothing of the operands survives
        LBC_CUDA(cudaMalloc(&buf_, bytes_));
    }

    ~L2Flusher() { LBC_CUDA_CLEANUP(cudaFree(buf_)); }

    L2Flusher(const L2Flusher&) = delete;
    L2Flusher& operator=(const L2Flusher&) = delete;
    L2Flusher(L2Flusher&&) = delete;
    L2Flusher& operator=(L2Flusher&&) = delete;

    void flush(cudaStream_t stream) { LBC_CUDA(cudaMemsetAsync(buf_, 0, bytes_, stream)); }

  private:
    void* buf_ = nullptr;
    std::size_t bytes_ = 0;
};

// Runs `fn(stream)` `warmup` times, then `runs` timed times (one event pair each). `fn` launches work on `stream`.
template <class Fn>
Stats time_it(Fn fn, cudaStream_t stream, int warmup = 3, int runs = 20, L2Flusher* flusher = nullptr) {
    for (int i = 0; i < warmup; ++i) { fn(stream); }
    LBC_CUDA(cudaGetLastError());
    cudaEvent_t start = nullptr, stop = nullptr;
    LBC_CUDA(cudaEventCreate(&start));
    LBC_CUDA(cudaEventCreate(&stop));
    std::vector<double> ms(static_cast<std::size_t>(runs));
    for (auto& t : ms) {
        if (flusher) { flusher->flush(stream); }
        LBC_CUDA(cudaEventRecord(start, stream));
        fn(stream);
        LBC_CUDA(cudaEventRecord(stop, stream));
        LBC_CUDA(cudaEventSynchronize(stop));
        float elapsed = 0.f;
        LBC_CUDA(cudaEventElapsedTime(&elapsed, start, stop));
        t = elapsed;
    }
    LBC_CUDA(cudaGetLastError());
    LBC_CUDA(cudaEventDestroy(start));
    LBC_CUDA(cudaEventDestroy(stop));
    std::vector<double> sorted = ms;
    std::ranges::sort(sorted);
    const double mean = std::accumulate(ms.begin(), ms.end(), 0.0) / runs;
    return {sorted[sorted.size() / 2], sorted.front(), mean, runs};
}

// One result line: name, median and min time, and the rates when `flops` / `bytes` are given (0: not shown).
inline void report(const char* name, const Stats& s, double flops = 0.0, double bytes = 0.0) {
    std::printf("%-40s median %9.4f ms  min %9.4f ms", name, s.median_ms, s.min_ms);
    if (flops > 0) { std::printf("  %7.1f TFLOPS", flops / (s.median_ms * 1e-3) / 1e12); }
    if (bytes > 0) { std::printf("  %7.1f GB/s", bytes / (s.median_ms * 1e-3) / 1e9); }
    std::printf("  (%d runs)\n", s.runs);
}

}  // namespace lbc
