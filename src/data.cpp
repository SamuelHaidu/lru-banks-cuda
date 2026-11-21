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

namespace {

// whether a row starting at `start` begins inside the scored part of its document: `score_after` seen since the
// document's marker (scanning back at most a million tokens)
bool scored_at(const uint16_t* tok, size_t start, int marker, int score_after) {
    if (score_after < 0) return true;
    for (size_t p = start, k = 0; p > 0 && k < (1u << 20); --p, ++k) {
        const int x = tok[p - 1];
        if (x == score_after) return true;
        if (x == marker) return false;
    }
    return false;
}

// one row of T tokens from `start` (positions past n read as the marker); returns the scored targets
int fill_row(const uint16_t* tok, size_t n, size_t start, int T, int marker, int score_after, int* ids, int* y,
             uint8_t* reset) {
    bool on = scored_at(tok, start, marker, score_after);
    int targets = 0;
    for (int t = 0; t < T; ++t) {
        const size_t p = start + size_t(t);
        const int x = p < n ? tok[p] : marker, nx = p + 1 < n ? tok[p + 1] : marker;
        if (x == marker) on = score_after < 0;
        if (x == score_after) on = true;
        const bool scored = nx != marker && on;
        ids[t] = x;
        y[t] = scored ? nx : -1;
        reset[t] = x == marker;
        targets += scored;
    }
    return targets;
}

}  // namespace

Loader::Loader(const TokenFile& file, int B, int T, uint64_t seed, int marker, uint64_t start_batch, int score_after,
               int slots)
    : file_(file), B_(B), T_(T), marker_(marker), score_after_(score_after), seed_(seed), next_batch_(start_batch),
      steps_per_pass_(steps_per_pass(file.size(), B, T)), slots_(size_t(slots)) {
    if (steps_per_pass_ == 0)
        throw std::runtime_error("Loader: the token file holds less than one batch of " + std::to_string(B) +
                                 " x " + std::to_string(T));
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
    const uint64_t g = next_batch_++, pass = g / steps_per_pass_, j = g % steps_per_pass_;
    if (pass != pass_) {  // a pass: random offset, shuffled windows (a pure function of the seed and the pass)
        std::mt19937_64 rng(seed_ * 1000003u + pass);
        offset_ = rng() % size_t(T_);
        order_.resize((n - offset_ - 1) / size_t(T_));
        std::iota(order_.begin(), order_.end(), 0u);
        std::shuffle(order_.begin(), order_.end(), rng);
        pass_ = pass;
    }
    const uint16_t* tok = file_.data();
    int targets = 0;
    for (int b = 0; b < B_; ++b) {
        const size_t start = offset_ + size_t(order_[j * B_ + b]) * T_;
        const size_t o = size_t(b) * T_;
        targets += fill_row(tok, n, start, T_, marker_, score_after_, hb.ids + o, hb.y + o, hb.reset + o);
    }
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

ValidationSet::ValidationSet(const TokenFile& file, int B, int T, int steps, uint64_t seed, int marker,
                             int score_after) {
    const size_t n = file.size();
    const uint16_t* tok = file.data();
    std::vector<size_t> starts;
    for (size_t i = 0; i < n; ++i)
        if (tok[i] == marker) starts.push_back(i);
    if (starts.empty()) starts.push_back(0);
    const size_t rows = size_t(B) * T;
    batches_.resize(size_t(steps));
    for (int i = 0; i < steps; ++i) {
        HostBatch& hb = batches_[size_t(i)];
        LBC_CUDA(cudaHostAlloc(&hb.ids, rows * sizeof(int), cudaHostAllocDefault));
        LBC_CUDA(cudaHostAlloc(&hb.y, rows * sizeof(int), cudaHostAllocDefault));
        LBC_CUDA(cudaHostAlloc(&hb.reset, rows, cudaHostAllocDefault));
        LBC_CUDA(cudaHostAlloc(&hb.inv_n, sizeof(float), cudaHostAllocDefault));
        std::mt19937_64 rng(seed * 7919u + 104729u * uint64_t(i) + 1);
        int targets = 0;
        for (int b = 0; b < B; ++b) {
            const size_t start = starts[rng() % starts.size()];
            const size_t o = size_t(b) * T;
            targets += fill_row(tok, n, start, T, marker, score_after, hb.ids + o, hb.y + o, hb.reset + o);
        }
        hb.n_targets = targets;
        *hb.inv_n = 1.f / float(targets > 0 ? targets : 1);
    }
}

ValidationSet::~ValidationSet() {
    for (auto& hb : batches_) {
        cudaFreeHost(hb.ids);
        cudaFreeHost(hb.y);
        cudaFreeHost(hb.reset);
        cudaFreeHost(hb.inv_n);
    }
}

}  // namespace lbc
