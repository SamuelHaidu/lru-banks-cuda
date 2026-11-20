#include "data.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <random>
#include <stdexcept>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include "lbc/check.hpp"

namespace lbc {

TokenFile::TokenFile(const std::string& path) {
    const int fd = open(path.c_str(), O_RDONLY);
    if (fd < 0) throw std::runtime_error("cannot open " + path);
    struct stat st{};
    fstat(fd, &st);
    map_bytes_ = size_t(st.st_size);
    map_ = mmap(nullptr, map_bytes_, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (map_ == MAP_FAILED) throw std::runtime_error("cannot mmap " + path);
    madvise(map_, map_bytes_, MADV_RANDOM);  // windows are read at random: no readahead
    // .npy: magic, version, header length, a dict ('descr': '<u2', 'fortran_order': False, 'shape': (n,))
    const auto* bytes = static_cast<const unsigned char*>(map_);
    if (map_bytes_ < 12 || std::memcmp(bytes, "\x93NUMPY", 6) != 0) throw std::runtime_error(path + " is not .npy");
    const int major = bytes[6];
    const size_t hlen = major == 1 ? size_t(bytes[8] | bytes[9] << 8)
                                   : size_t(bytes[8] | bytes[9] << 8 | bytes[10] << 16 | size_t(bytes[11]) << 24);
    const size_t start = (major == 1 ? 10 : 12) + hlen;
    const std::string header(reinterpret_cast<const char*>(bytes) + (major == 1 ? 10 : 12), hlen);
    if (header.find("'<u2'") == std::string::npos) throw std::runtime_error(path + ": expected uint16 ('<u2')");
    data_ = reinterpret_cast<const uint16_t*>(bytes + start);
    n_ = (map_bytes_ - start) / 2;
}

TokenFile::~TokenFile() {
    if (map_) munmap(map_, map_bytes_);
}

Loader::Loader(const TokenFile& file, int B, int T, uint64_t seed, int marker, int slots)
    : file_(file), B_(B), T_(T), marker_(marker), seed_(seed), slots_(size_t(slots)) {
    for (auto& s : slots_) {
        const size_t n = size_t(B) * T;
        LBC_CUDA(cudaHostAlloc(&s.hb.ids, n * sizeof(int), cudaHostAllocDefault));
        LBC_CUDA(cudaHostAlloc(&s.hb.y, n * sizeof(int), cudaHostAllocDefault));
        LBC_CUDA(cudaHostAlloc(&s.hb.reset, n, cudaHostAllocDefault));
        LBC_CUDA(cudaHostAlloc(&s.hb.inv_n, sizeof(float), cudaHostAllocDefault));
        LBC_CUDA(cudaEventCreateWithFlags(&s.done, cudaEventDisableTiming));
    }
    thread_ = std::thread([this] { run(); });
}

Loader::~Loader() {
    {
        std::lock_guard<std::mutex> lk(mu_);
        stop_ = true;
    }
    cv_.notify_all();
    thread_.join();
    for (auto& s : slots_) {
        cudaFreeHost(s.hb.ids);
        cudaFreeHost(s.hb.y);
        cudaFreeHost(s.hb.reset);
        cudaFreeHost(s.hb.inv_n);
        cudaEventDestroy(s.done);
    }
}

void Loader::fill(HostBatch& hb) {
    const size_t n = file_.size();
    if (order_.empty() || pos_ + B_ > order_.size()) {  // a new pass: random offset, shuffled windows
        std::mt19937_64 rng(seed_ * 1000003u + pass_++);
        offset_ = rng() % size_t(T_);
        order_.resize((n - offset_ - 1) / size_t(T_));
        std::iota(order_.begin(), order_.end(), 0u);
        std::shuffle(order_.begin(), order_.end(), rng);
        pos_ = 0;
    }
    const uint16_t* tok = file_.data();
    int targets = 0;
    for (int b = 0; b < B_; ++b) {
        const size_t start = offset_ + size_t(order_[pos_ + b]) * T_;
        const size_t o = size_t(b) * T_;
        for (int t = 0; t < T_; ++t) {
            const int x = tok[start + t], y = tok[start + t + 1];
            hb.ids[o + t] = x;
            hb.y[o + t] = y == marker_ ? -1 : y;
            hb.reset[o + t] = x == marker_;
            targets += y != marker_;
        }
    }
    pos_ += B_;
    hb.n_targets = targets;
    *hb.inv_n = 1.f / float(targets > 0 ? targets : 1);
}

void Loader::run() {
    for (;;) {
        Slot* s;
        {
            std::unique_lock<std::mutex> lk(mu_);
            cv_.wait(lk, [&] { return stop_ || !slots_[produce_ % slots_.size()].ready; });
            if (stop_) return;
            s = &slots_[produce_ % slots_.size()];
        }
        if (s->pending) {  // its previous batch's copy to the GPU must be done before the buffer is rewritten
            LBC_CUDA(cudaEventSynchronize(s->done));
            s->pending = false;
        }
        const auto t0 = std::chrono::steady_clock::now();
        fill(s->hb);
        fill_ms_ += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        {
            std::lock_guard<std::mutex> lk(mu_);
            s->ready = true;
            ++produce_;
        }
        cv_.notify_all();
    }
}

HostBatch* Loader::next() {
    const auto t0 = std::chrono::steady_clock::now();
    std::unique_lock<std::mutex> lk(mu_);
    cv_.wait(lk, [&] { return slots_[consume_ % slots_.size()].ready; });
    wait_ms_ += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    return &slots_[consume_ % slots_.size()].hb;
}

void Loader::release(HostBatch* hb, cudaStream_t stream) {
    {
        std::lock_guard<std::mutex> lk(mu_);
        Slot& s = slots_[consume_ % slots_.size()];
        if (&s.hb != hb) throw std::runtime_error("Loader::release: batches must be released in order");
        LBC_CUDA(cudaEventRecord(s.done, stream));  // after the copies queued on `stream`
        s.pending = true;
        s.ready = false;
        ++consume_;
    }
    cv_.notify_all();
}

}  // namespace lbc
