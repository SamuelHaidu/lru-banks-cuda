#include "data.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <random>
#include <stdexcept>
#include <utility>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include "lbc/check.hpp"

namespace lbc {

TokenFile::TokenFile(const std::string& path) {
    const int fd = open(path.c_str(), O_RDONLY);
    if (fd < 0) { throw std::runtime_error("cannot open " + path); }
    struct stat st{};
    fstat(fd, &st);
    map_bytes_ = size_t(st.st_size);
    map_ = mmap(nullptr, map_bytes_, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (map_ == MAP_FAILED) { throw std::runtime_error("cannot mmap " + path); }
    madvise(map_, map_bytes_, MADV_RANDOM);  // windows are read at random: no readahead
    // .npy: magic, version, header length, a dict ('descr': '<u2', 'fortran_order': False, 'shape': (n,))
    const auto* bytes = static_cast<const unsigned char*>(map_);
    if (map_bytes_ < 12 || std::memcmp(bytes, "\x93NUMPY", 6) != 0) { throw std::runtime_error(path + " is not .npy"); }
    const int major = bytes[6];
    const size_t hlen = major == 1 ? size_t(bytes[8] | bytes[9] << 8)
                                   : bytes[8] | bytes[9] << 8 | bytes[10] << 16 | size_t(bytes[11]) << 24;
    const size_t start = (major == 1 ? 10 : 12) + hlen;
    const std::string header(reinterpret_cast<const char*>(bytes) + (major == 1 ? 10 : 12), hlen);
    if (header.find("'<u2'") == std::string::npos) { throw std::runtime_error(path + ": expected uint16 ('<u2')"); }
    data_ = reinterpret_cast<const uint16_t*>(bytes + start);
    n_ = (map_bytes_ - start) / 2;
}

TokenFile::~TokenFile() {
    if (map_) { munmap(map_, map_bytes_); }
}

namespace {

// whether a row starting at `start` begins inside the scored part of its document: `score_after` seen since the
// document's marker (scanning back at most a million tokens)
bool scored_at(const uint16_t* tok, size_t start, int marker, int score_after) {
    if (score_after < 0) { return true; }
    for (size_t p = start, k = 0; p > 0 && k < (1u << 20); --p, ++k) {
        const int x = tok[p - 1];
        if (x == score_after) { return true; }
        if (x == marker) { return false; }
    }
    return false;
}

// whether a row starting at `start` begins inside a loss span: the last open tag, close tag or marker before it is
// the open tag (scanning back at most a million tokens)
bool in_span_at(const uint16_t* tok, size_t start, int marker, Span span) {
    if (span.open < 0) { return true; }
    for (size_t p = start, k = 0; p > 0 && k < (1u << 20); --p, ++k) {
        const int x = tok[p - 1];
        if (x == span.open) { return true; }
        if (x == span.close || x == marker) { return false; }
    }
    return false;
}

// one row of T tokens from `start` (positions past n read as the marker); returns the scored targets. score_prev >= 0:
// only targets whose input token follows that token are scored (a recall task's query marker: `? key` -> value).
// span (the dataset's loss_spans, as the Python pipeline): a target counts when the last open / close tag before it
// is the open tag, so the span's tokens and its close tag count, the open tag and everything outside do not; the
// marker closes any span
int fill_row(const uint16_t* tok,
             size_t n,
             size_t start,
             int T,
             int marker,
             int score_after,
             int score_prev,
             int* ids,
             int* y,
             uint8_t* reset,
             Span span) {
    bool on = scored_at(tok, start, marker, score_after);
    bool in = in_span_at(tok, start, marker, span);
    int targets = 0;
    for (int t = 0; t < T; ++t) {
        const size_t p = start + size_t(t);
        const int x = p < n ? tok[p] : marker, nx = p + 1 < n ? tok[p + 1] : marker;
        if (x == marker) { on = score_after < 0; }
        if (x == score_after) { on = true; }
        if (span.open >= 0) {
            if (x == span.open) {
                in = true;
            } else if (x == span.close || x == marker) {
                in = false;
            }
        }
        const bool scored = nx != marker && on && in
                            && (score_prev < 0 || (p > 0 && p - 1 < n && std::cmp_equal(tok[p - 1], score_prev)));
        ids[t] = x;
        y[t] = scored ? nx : -1;
        reset[t] = static_cast<uint8_t>(x == marker);
        targets += static_cast<int>(scored);
    }
    return targets;
}

}  // namespace

void fill_candidates(const int* ids, const int* y, const uint8_t* reset, int T, int nc, int marker, int* cand) {
    std::vector<int> recent(size_t(nc) + 1);  // distinct tokens, most recent first (move to front)
    int n = 0;
    for (int t = 0; t < T; ++t) {
        if (reset[t] != 0) { n = 0; }
        const int x = ids[t];
        if (x != marker) {
            int at = 0;  // x's position, or n: not in the list
            while (at < n && recent[size_t(at)] != x) { ++at; }
            if (at == n && n < nc) { ++n; }
            for (int i = std::min(at, nc - 1); i > 0; --i) { recent[size_t(i)] = recent[size_t(i) - 1]; }
            recent[0] = x;
        }
        int* c = cand + size_t(t) * nc;
        int k = 0;
        if (y[t] >= 0) {
            for (int i = 0; i < n; ++i) {
                if (recent[size_t(i)] != y[t]) { c[k++] = recent[size_t(i)]; }
            }
        }
        for (; k < nc; ++k) { c[k] = -1; }
    }
}

Loader::Loader(const TokenFile& file,
               int B,
               int T,
               uint64_t seed,
               int marker,
               uint64_t start_batch,
               int score_after,
               int score_prev,
               bool aligned,
               int slots,
               Span span,
               int stream_chunks,
               int nc)
    : file_(file),
      B_(B),
      T_(T),
      marker_(marker),
      score_after_(score_after),
      score_prev_(score_prev),
      span_(span),
      aligned_(aligned),
      seed_(seed),
      stream_(stream_chunks),
      nc_(nc),
      next_batch_(start_batch),
      first_batch_(start_batch),
      steps_per_pass_(steps_per_pass(file.size(), B, T, stream_chunks)),
      slots_(size_t(slots)) {
    if (stream_chunks < 1) { throw std::runtime_error("Loader: stream_chunks must be >= 1"); }
    if (steps_per_pass_ == 0) {
        throw std::runtime_error("Loader: the token file holds less than one batch of " + std::to_string(B) + " x "
                                 + std::to_string(T));
    }
    for (auto& s : slots_) {
        const size_t n = size_t(B) * T;
        LBC_CUDA(cudaHostAlloc(&s.hb.ids, n * sizeof(int), cudaHostAllocDefault));
        LBC_CUDA(cudaHostAlloc(&s.hb.y, n * sizeof(int), cudaHostAllocDefault));
        LBC_CUDA(cudaHostAlloc(&s.hb.reset, n, cudaHostAllocDefault));
        LBC_CUDA(cudaHostAlloc(&s.hb.inv_n, sizeof(float), cudaHostAllocDefault));
        if (nc > 0) { LBC_CUDA(cudaHostAlloc(&s.hb.cand, n * size_t(nc) * sizeof(int), cudaHostAllocDefault)); }
        LBC_CUDA(cudaEventCreateWithFlags(&s.done, cudaEventDisableTiming));
    }
    thread_ = std::thread([this] { run(); });
}

Loader::~Loader() {
    {
        std::lock_guard<std::mutex> const lk(mu_);
        stop_ = true;
    }
    cv_.notify_all();
    thread_.join();
    for (auto& s : slots_) {
        LBC_CUDA_CLEANUP(cudaFreeHost(s.hb.ids));
        LBC_CUDA_CLEANUP(cudaFreeHost(s.hb.y));
        LBC_CUDA_CLEANUP(cudaFreeHost(s.hb.reset));
        LBC_CUDA_CLEANUP(cudaFreeHost(s.hb.inv_n));
        if (s.hb.cand != nullptr) { LBC_CUDA_CLEANUP(cudaFreeHost(s.hb.cand)); }
        LBC_CUDA_CLEANUP(cudaEventDestroy(s.done));
    }
}

void Loader::fill(HostBatch& hb) {
    const size_t n = file_.size();
    const uint64_t g = next_batch_++, pass = g / steps_per_pass_, j = g % steps_per_pass_;
    const size_t W = size_t(T_) * size_t(stream_);  // a window: one stream's rows (T without streams)
    if (pass != pass_) {  // a pass: random offset, shuffled windows (a pure function of the seed and the pass)
        std::mt19937_64 rng(seed_ * 1000003u + pass);
        offset_ = aligned_ ? 0 : rng() % W;
        order_.resize((n - offset_ - 1) / W);
        std::iota(order_.begin(), order_.end(), 0u);
        std::shuffle(order_.begin(), order_.end(), rng);
        pass_ = pass;
    }
    const uint16_t* tok = file_.data();
    int targets = 0;
    const uint64_t group = j / uint64_t(stream_), k = j % uint64_t(stream_);
    // no carried state to continue from: a new stream, or this loader's first batch (a fresh or resumed model)
    const bool fresh = stream_ > 1 && (k == 0 || g == first_batch_);
    for (int b = 0; b < B_; ++b) {
        const size_t start = offset_ + size_t(order_[group * B_ + b]) * W + size_t(k) * T_;
        const size_t o = size_t(b) * T_;
        targets +=
            fill_row(tok, n, start, T_, marker_, score_after_, score_prev_, hb.ids + o, hb.y + o, hb.reset + o, span_);
        if (fresh) { hb.reset[o] = 1; }
        if (nc_ > 0) {
            fill_candidates(hb.ids + o, hb.y + o, hb.reset + o, T_, nc_, marker_, hb.cand + o * size_t(nc_));
        }
    }
    hb.n_targets = targets;
    *hb.inv_n = 1.f / float(targets > 0 ? targets : 1);
}

void Loader::run() {
    for (;;) {
        Slot* s = nullptr;
        {
            std::unique_lock<std::mutex> lk(mu_);
            cv_.wait(lk, [&] { return stop_ || !slots_[produce_ % slots_.size()].ready; });
            if (stop_) { return; }
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
            std::lock_guard<std::mutex> const lk(mu_);
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
        std::lock_guard<std::mutex> const lk(mu_);
        Slot& s = slots_[consume_ % slots_.size()];
        if (&s.hb != hb) { throw std::runtime_error("Loader::release: batches must be released in order"); }
        LBC_CUDA(cudaEventRecord(s.done, stream));  // after the copies queued on `stream`
        s.pending = true;
        s.ready = false;
        ++consume_;
    }
    cv_.notify_all();
}

MixLoader::MixLoader(std::vector<std::unique_ptr<Loader>> sources, std::vector<int> rows, int T)
    : src_(std::move(sources)), rows_(std::move(rows)), T_(T), targets_(src_.size(), 0) {
    if (src_.empty() || src_.size() != rows_.size()) {
        throw std::runtime_error("MixLoader: one row count per source");
    }
    for (int const r : rows_) { B_ += r; }
    nc_ = src_[0]->nc();
    for (const auto& s : src_) {
        if (s->nc() != nc_) { throw std::runtime_error("MixLoader: every source needs the same candidate count"); }
    }
    const size_t n = size_t(B_) * T_;
    for (int i = 0; i < 2; ++i) {
        LBC_CUDA(cudaHostAlloc(&buf_[i].ids, n * sizeof(int), cudaHostAllocDefault));
        LBC_CUDA(cudaHostAlloc(&buf_[i].y, n * sizeof(int), cudaHostAllocDefault));
        LBC_CUDA(cudaHostAlloc(&buf_[i].reset, n, cudaHostAllocDefault));
        LBC_CUDA(cudaHostAlloc(&buf_[i].inv_n, sizeof(float), cudaHostAllocDefault));
        if (nc_ > 0) { LBC_CUDA(cudaHostAlloc(&buf_[i].cand, n * size_t(nc_) * sizeof(int), cudaHostAllocDefault)); }
        LBC_CUDA(cudaEventCreateWithFlags(&done_[i], cudaEventDisableTiming));
    }
    LBC_CUDA(cudaStreamCreateWithFlags(&side_, cudaStreamNonBlocking));
}

MixLoader::~MixLoader() {
    for (int i = 0; i < 2; ++i) {
        LBC_CUDA_CLEANUP(cudaFreeHost(buf_[i].ids));
        LBC_CUDA_CLEANUP(cudaFreeHost(buf_[i].y));
        LBC_CUDA_CLEANUP(cudaFreeHost(buf_[i].reset));
        LBC_CUDA_CLEANUP(cudaFreeHost(buf_[i].inv_n));
        if (buf_[i].cand != nullptr) { LBC_CUDA_CLEANUP(cudaFreeHost(buf_[i].cand)); }
        LBC_CUDA_CLEANUP(cudaEventDestroy(done_[i]));
    }
    LBC_CUDA_CLEANUP(cudaStreamDestroy(side_));
}

HostBatch* MixLoader::next() {
    const int k = k_++ % 2;
    HostBatch& out = buf_[k];
    if (recorded_[k]) {
        LBC_CUDA(cudaEventSynchronize(done_[k]));  // the copies from this buffer have been done
    }
    size_t row = 0;
    int targets = 0;
    for (size_t i = 0; i < src_.size(); ++i) {
        HostBatch* h = src_[i]->next();
        const size_t o = row * T_, n = size_t(rows_[i]) * T_;
        std::memcpy(out.ids + o, h->ids, n * sizeof(int));
        std::memcpy(out.y + o, h->y, n * sizeof(int));
        std::memcpy(out.reset + o, h->reset, n);
        if (nc_ > 0) { std::memcpy(out.cand + o * size_t(nc_), h->cand, n * size_t(nc_) * sizeof(int)); }
        targets += h->n_targets;
        targets_[i] += h->n_targets;
        src_[i]->release(h, side_);  // copied on the CPU: free at once (the side stream is idle)
        row += size_t(rows_[i]);
    }
    out.n_targets = targets;
    *out.inv_n = 1.f / float(targets > 0 ? targets : 1);
    return &out;
}

void MixLoader::release(HostBatch* hb, cudaStream_t stream) {
    const int k = hb == &buf_[0] ? 0 : 1;
    LBC_CUDA(cudaEventRecord(done_[k], stream));
    recorded_[k] = true;
}

ValidationSet::ValidationSet(const TokenFile& file,
                             int B,
                             int T,
                             int steps,
                             uint64_t seed,
                             int marker,
                             int score_after,
                             int score_prev,
                             bool aligned,
                             Span span) {
    const size_t n = file.size();
    const uint16_t* tok = file.data();
    std::vector<size_t> starts;
    if (aligned) {
        for (size_t i = 0; i + size_t(T) <= n; i += size_t(T)) { starts.push_back(i); }
    } else {
        for (size_t i = 0; i < n; ++i) {
            if (std::cmp_equal(tok[i], marker)) { starts.push_back(i); }
        }
    }
    if (starts.empty()) { starts.push_back(0); }
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
            targets +=
                fill_row(tok, n, start, T, marker, score_after, score_prev, hb.ids + o, hb.y + o, hb.reset + o, span);
        }
        hb.n_targets = targets;
        *hb.inv_n = 1.f / float(targets > 0 ? targets : 1);
    }
}

ValidationSet::~ValidationSet() {
    for (auto& hb : batches_) {
        LBC_CUDA_CLEANUP(cudaFreeHost(hb.ids));
        LBC_CUDA_CLEANUP(cudaFreeHost(hb.y));
        LBC_CUDA_CLEANUP(cudaFreeHost(hb.reset));
        LBC_CUDA_CLEANUP(cudaFreeHost(hb.inv_n));
    }
}

}  // namespace lbc
