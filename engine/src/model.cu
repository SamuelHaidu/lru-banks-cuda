#include "model.hpp"

#include <algorithm>
#include <cmath>
#include <map>
#include <random>

#include "kernels.hpp"
#include "lbc/check.hpp"

namespace lbc {

int Model::add(const std::string& name, int rows, int cols, bool decay) {
    tensors_.push_back({name, n_, rows, cols, decay});
    n_ += (size_t(rows) * cols + 63) / 64 * 64;  // 256-byte aligned offsets
    return int(tensors_.size()) - 1;
}

template <class T>
T* Model::alloc(size_t n) {
    void* ptr = nullptr;
    LBC_CUDA(cudaMalloc(&ptr, std::max<size_t>(n, 1) * sizeof(T)));
    allocations_.push_back(ptr);
    return static_cast<T*>(ptr);
}

Model::Model(const Config& c, uint64_t seed) : c_(c) {
    const int D = c.D, HD = c.HD(), S2 = c.S2(), H = c.H, F = c.F, E0 = c.E0, V = c.V, M = c.M();
    if (int(c.hops.size()) != 0 && int(c.hops.size()) != c.L) {
        fail("Model", "hops needs one entry per layer", __FILE__, __LINE__);
    }
    if (!c.slots.empty() && int(c.slots.size()) != c.L) {
        fail("Model", "slots needs one entry per layer", __FILE__, __LINE__);
    }
    if (c.dh != 32 && c.dh != 64) { fail("Model", "the head size must be 32 or 64", __FILE__, __LINE__); }
    if (!c.factorized && E0 != D) { fail("Model", "an unfactorized embedding is D wide (E0 = D)", __FILE__, __LINE__); }
    E_ = add("E", V, E0, true);
    if (!c.tied) { E_out_ = add("E_out", V, E0, true); }
    if (c.factorized) { W_in_ = add("W_in", E0, D, true); }
    final_norm_ = add("final_norm", 1, D, false);
    if (c.factorized) { W_out_ = add("W_out", D, E0, true); }
    for (int n = 0; n < c.L; ++n) {
        const bool two = c.hops_of(n) == 2;
        const std::string l = "l" + std::to_string(n) + "_";
        LayerIdx x;
        x.norm1 = add(l + "norm1", 1, D, false);
        x.Wa = add(l + "Wa", D, c.ldPV(), true);
        x.nu = add(l + "nu", 1, c.N, false);
        x.theta = add(l + "theta", 1, c.N, false);
        x.Ws = add(l + "Ws", S2, c.ldG(), true);
        x.bk = add(l + "bk", 1, HD, false);
        x.brg = add(l + "brg", 1, H, false);
        x.bwg = add(l + "bwg", 1, H, false);
        x.R = add(l + "R", H, c.dh * SLOTS, false);
        x.ltau = add(l + "log_tau", 1, H, false);
        if (two) { x.ltau2 = add(l + "log_tau2", 1, H, false); }
        x.Wo = add(l + "Wo", HD, D, true);
        if (two) { x.Wo2 = add(l + "Wo2", HD, D, true); }
        x.ffn_norm = add(l + "ffn_norm", 1, D, false);
        x.W1 = add(l + "W1", D, F, true);
        x.b1 = add(l + "b1", 1, F, false);
        x.W2 = add(l + "W2", F, D, true);
        x.b2 = add(l + "b2", 1, D, false);
        if (two) {
            x.q2_norm = add(l + "q2_norm", 1, D, false);
            x.Wq2 = add(l + "Wq2", D, c.ldG2(), true);
            x.brg2 = add(l + "brg2", 1, H, false);
        }
        li_.push_back(x);
    }
    P_ = alloc<float>(n_);
    G_ = alloc<float>(n_);
    M1_ = alloc<float>(n_);
    M2_ = alloc<float>(n_);
    decay_ = alloc<uint8_t>(n_);
    LBC_CUDA(cudaMemset(M1_, 0, n_ * 4));
    LBC_CUDA(cudaMemset(M2_, 0, n_ * 4));

    if (c.factorized || c.tied) {
        Eb_ = alloc<bf16>(size_t(V) * E0);  // the input gather and / or the tied readout
    }
    if (!c.tied) { Eoutb_ = alloc<bf16>(size_t(V) * E0); }
    if (c.factorized) {
        Winb_ = alloc<bf16>(size_t(E0) * D);
        Woutb_ = alloc<bf16>(size_t(D) * E0);
    }
    loga_ = alloc<float>(size_t(c.L) * H * SLOTS);  // per layer: its bank size sets the half-life spacing
    const int nchunks = (c.T + c.bank_chunk - 1) / c.bank_chunk;
    const size_t ckpt_floats = size_t(c.B) * H * nchunks * ckpt_fields(c.dh) * SLOTS;
    bool any_two = false;
    for (int n = 0; n < c.L; ++n) {
        const bool two = c.hops_of(n) == 2;
        any_two |= two;
        LayerW w{};
        w.Wa = alloc<bf16>(size_t(D) * c.ldPV());
        w.Ws = alloc<bf16>(size_t(S2) * c.ldG());
        w.Wo = alloc<bf16>(size_t(HD) * D);
        if (two) { w.Wo2 = alloc<bf16>(size_t(HD) * D); }
        w.W1 = alloc<bf16>(size_t(D) * F);
        w.W2 = alloc<bf16>(size_t(F) * D);
        if (two) { w.Wq2 = alloc<bf16>(size_t(D) * c.ldG2()); }
        w.b1 = alloc<bf16>(F);
        lw_.push_back(w);
        LayerA a{};
        a.xh1 = alloc<bf16>(size_t(M) * D);
        a.rstd1 = alloc<float>(M);
        a.PV = alloc<bf16>(size_t(M) * c.ldPV());
        a.s = alloc<float>(size_t(M) * S2);
        a.sb = alloc<bf16>(size_t(M) * S2);
        a.k = alloc<bf16>(size_t(M) * HD);
        a.q = alloc<bf16>(size_t(M) * HD);
        a.kn = alloc<float>(size_t(M) * H);
        a.qn = alloc<float>(size_t(M) * H);
        a.ridx = alloc<uint8_t>(size_t(M) * H * 2);
        a.rp1 = alloc<float>(size_t(M) * H);
        a.gw = alloc<uint8_t>(size_t(M) * H);
        a.gr = alloc<uint8_t>(size_t(M) * H);
        a.aw = alloc<float>(size_t(M) * H);
        a.ar = alloc<float>(size_t(M) * H);
        a.o1 = alloc<bf16>(size_t(M) * HD);
        a.xh2 = alloc<bf16>(size_t(M) * D);
        a.rstd2 = alloc<float>(M);
        a.Z = alloc<bf16>(size_t(M) * F);
        a.U = alloc<bf16>(size_t(M) * F);
        if (two) {
            a.xh3 = alloc<bf16>(size_t(M) * D);
            a.rstd3 = alloc<float>(M);
            a.q2 = alloc<bf16>(size_t(M) * HD);
            a.q2n = alloc<float>(size_t(M) * H);
            a.ar2 = alloc<float>(size_t(M) * H);
            a.gr2 = alloc<uint8_t>(size_t(M) * H);
            a.o2 = alloc<bf16>(size_t(M) * HD);
        }
        a.info = alloc<uint2>(size_t(M) * H);
        a.zpre = alloc<float>(size_t(M) * H * 2);
        a.ckpt = nchunks > 1 ? alloc<float>(ckpt_floats) : nullptr;
        la_.push_back(a);
    }
    const int Mc = std::min(c.ce_chunk, M);
    h_ = alloc<float>(size_t(M) * D);
    if (c.factorized) { x0_ = alloc<bf16>(size_t(M) * E0); }
    xhF_ = alloc<bf16>(size_t(M) * D);
    rstdF_ = alloc<float>(M);
    hr_ = alloc<bf16>(size_t(M) * E0);
    Gt_ = alloc<bf16>(size_t(M) * c.ldG());
    G2t_ = any_two ? alloc<bf16>(size_t(M) * c.ldG2()) : nullptr;
    logits_ = alloc<float>(size_t(Mc) * V);
    dlog_ = alloc<bf16>(size_t(Mc) * V);
    dhrb_ = alloc<bf16>(size_t(M) * E0);
    dh_ = alloc<float>(size_t(M) * D);
    dhb_ = alloc<bf16>(size_t(M) * D);
    dxh_ = alloc<float>(size_t(M) * D);
    dO1_ = alloc<float>(size_t(M) * HD);
    dO2_ = any_two ? alloc<float>(size_t(M) * HD) : nullptr;
    db1b_ = alloc<bf16>(F);
    dZ_ = alloc<bf16>(size_t(M) * F);
    dG_ = alloc<bf16>(size_t(M) * c.ldG());
    dG2_ = any_two ? alloc<bf16>(size_t(M) * c.ldG2()) : nullptr;
    dS_ = alloc<float>(size_t(M) * S2);
    dPV_ = alloc<bf16>(size_t(M) * c.ldPV());
    dk2_ = alloc<bf16>(size_t(M) * HD * 2);
    dv2_ = alloc<bf16>(size_t(M) * HD * 2);
    dr2_ = alloc<float>(size_t(M) * H * 2);
    dq1_ = alloc<float>(size_t(M) * HD);
    dq2_ = any_two ? alloc<float>(size_t(M) * HD) : nullptr;
    dgr1_ = alloc<float>(size_t(M) * H);
    dgr2_ = any_two ? alloc<float>(size_t(M) * H) : nullptr;
    coef1_ = alloc<bf16>(3 * size_t(M) * H * SLOTS);
    coef2_ = any_two ? alloc<bf16>(3 * size_t(M) * H * SLOTS) : nullptr;
    if (c.factorized) { dx0_ = alloc<float>(size_t(M) * E0); }
    size_t wmax = 0;
    for (const auto& t : tensors_) { wmax = std::max(wmax, size_t(t.rows) * t.cols); }
    dWp_ = alloc<float>(wmax);
    metrics_ = alloc<float>(size_t(metrics_count()));
    LBC_CUDA(cudaMemset(metrics_, 0, size_t(metrics_count()) * sizeof(float)));
    loss_ = metrics_;
    acc_ = metrics_ + 1;
    stats_ = metrics_ + 2;
    gsq_ = metrics_ + 2 + 6 * c.L;
    lr_ = gsq_ + 1;
    step_ = alloc<int>(1);
    LBC_CUDA(cudaMemset(step_, 0, sizeof(int)));
    carry_ = nchunks > 1 ? alloc<float>(ckpt_floats) : nullptr;
    cflags_ = nchunks > 1 ? alloc<int>(size_t(c.B) * H * nchunks) : nullptr;
    init(seed);
}

Model::~Model() {
    for (auto& mk : marks_) {
        cudaEventDestroy(mk.a);
        cudaEventDestroy(mk.b);
    }
    for (void* ptr : allocations_) { cudaFree(ptr); }
}

void Model::init(uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::vector<float> hp(n_, 0.f);
    std::vector<uint8_t> hd(n_, 0);
    auto normal = [&](int i, float std) {
        std::normal_distribution<float> nd(0.f, std);
        const Tensor& t = tensors_[i];
        for (size_t j = 0; j < size_t(t.rows) * t.cols; ++j) { hp[t.off + j] = nd(rng); }
    };
    auto fill = [&](int i, float v) {
        const Tensor& t = tensors_[i];
        std::fill(hp.begin() + t.off, hp.begin() + t.off + size_t(t.rows) * t.cols, v);
    };
    int reads = 0;
    for (int n = 0; n < c_.L; ++n) { reads += c_.hops_of(n); }
    const float out_std = 0.02f / std::sqrt(2.f * reads);  // every read writes into the stream
    normal(E_, 0.02f);
    if (E_out_ >= 0) { normal(E_out_, 0.02f); }
    if (W_in_ >= 0) { normal(W_in_, 0.02f); }
    fill(final_norm_, 1.f);
    if (W_out_ >= 0) { normal(W_out_, 0.02f); }
    const int N = c_.N;
    for (int n = 0; n < c_.L; ++n) {
        const LayerIdx& x = li_[n];
        fill(x.norm1, 1.f);
        normal(x.Wa, 0.02f);
        // LRU half-lives log-spaced over [lru_hl_min, lru_hl_max]: |lambda| = 2^(-1 / half-life)
        std::uniform_real_distribution<float> ud(0.f, 6.283185307179586f);
        for (int j = 0; j < N; ++j) {
            const double hl = std::exp(std::log(c_.lru_hl_min)
                                       + (std::log(c_.lru_hl_max) - std::log(c_.lru_hl_min)) * j / std::max(1, N - 1));
            hp[tensors_[x.nu].off + j] = float(std::log(std::log(2.0) / hl));
            hp[tensors_[x.theta].off + j] = ud(rng);
        }
        normal(x.Ws, 0.02f);
        fill(x.bk, 0.f);
        fill(x.brg, c_.gate_bias_r);
        fill(x.bwg, c_.gate_bias_w);
        normal(x.R, 1.f / std::sqrt(float(c_.dh)));
        const bool two = c_.hops_of(n) == 2;  // (the draws keep the order of the all-2-hop model)
        fill(x.ltau, std::log(c_.tau));
        if (two) { fill(x.ltau2, std::log(c_.tau)); }
        normal(x.Wo, out_std);
        if (two) { normal(x.Wo2, out_std); }
        fill(x.ffn_norm, 1.f);
        normal(x.W1, 0.02f);
        fill(x.b1, 0.f);
        normal(x.W2, out_std);
        fill(x.b2, 0.f);
        if (two) {
            fill(x.q2_norm, 1.f);
            normal(x.Wq2, 0.02f);
            fill(x.brg2, c_.gate_bias_r);
        }
    }
    for (const auto& t : tensors_) {
        if (t.decay) { std::fill(hd.begin() + t.off, hd.begin() + t.off + size_t(t.rows) * t.cols, 1); }
    }
    LBC_CUDA(cudaMemcpy(P_, hp.data(), n_ * 4, cudaMemcpyHostToDevice));
    LBC_CUDA(cudaMemcpy(decay_, hd.data(), n_, cudaMemcpyHostToDevice));
    // slot decays: half-lives log-spaced over [slot_hl_min, slot_hl_max] across the layer's bank size (the same for
    // every head); slots past it are never written (their decay is the slowest one's)
    std::vector<float> la(size_t(c_.L) * c_.H * SLOTS);
    for (int n = 0; n < c_.L; ++n) {
        const int ns = c_.slots_of(n);
        for (int m = 0; m < SLOTS; ++m) {
            const int mm = std::min(m, ns - 1);
            const double hl = std::exp(std::log(c_.slot_hl_min)
                                       + (std::log(c_.slot_hl_max) - std::log(c_.slot_hl_min)) * mm / (ns - 1));
            for (int h = 0; h < c_.H; ++h) { la[(size_t(n) * c_.H + h) * SLOTS + m] = float(std::log(0.5) / hl); }
        }
    }
    LBC_CUDA(cudaMemcpy(loga_, la.data(), la.size() * 4, cudaMemcpyHostToDevice));
    prepare_weights(nullptr);
    LBC_CUDA(cudaDeviceSynchronize());
}

void Model::prepare_weights(cudaStream_t s) {
    const int D = c_.D, HD = c_.HD(), F = c_.F, E0 = c_.E0;
    if (Eb_) { prep_weight(p(E_), nullptr, Eb_, c_.V, E0, s); }
    if (Eoutb_) { prep_weight(p(E_out_), nullptr, Eoutb_, c_.V, E0, s); }
    if (c_.factorized) {
        prep_weight(p(W_in_), nullptr, Winb_, E0, D, s);
        prep_weight(p(W_out_), p(final_norm_), Woutb_, D, E0, s);
    }
    for (int n = 0; n < c_.L; ++n) {
        const LayerIdx& x = li_[n];
        const LayerW& w = lw_[n];
        prep_weight(p(x.Wa), p(x.norm1), w.Wa, D, c_.ldPV(), s);
        prep_weight(p(x.Ws), nullptr, w.Ws, c_.S2(), c_.ldG(), s);
        prep_weight(p(x.Wo), nullptr, w.Wo, HD, D, s);
        if (c_.hops_of(n) == 2) {
            prep_weight(p(x.Wo2), nullptr, w.Wo2, HD, D, s);
            prep_weight(p(x.Wq2), p(x.q2_norm), w.Wq2, D, c_.ldG2(), s);
        }
        prep_weight(p(x.W1), p(x.ffn_norm), w.W1, D, F, s);
        prep_weight(p(x.W2), nullptr, w.W2, F, D, s);
        prep_weight(p(x.b1), nullptr, w.b1, 1, F, s);
    }
}

void Model::tic(cudaStream_t s, const char* cat) {
    if (!profile_) { return; }
    Mark mk{cat, nullptr, nullptr};
    LBC_CUDA(cudaEventCreate(&mk.a));
    LBC_CUDA(cudaEventCreate(&mk.b));
    LBC_CUDA(cudaEventRecord(mk.a, s));
    marks_.push_back(mk);
}

void Model::toc(cudaStream_t s) {
    if (!profile_) { return; }
    LBC_CUDA(cudaEventRecord(marks_.back().b, s));
}

std::vector<std::pair<std::string, float>> Model::profile_ms() {
    std::map<std::string, float> acc;
    for (auto& mk : marks_) {
        LBC_CUDA(cudaEventSynchronize(mk.b));
        float ms = 0.f;
        LBC_CUDA(cudaEventElapsedTime(&ms, mk.a, mk.b));
        acc[mk.cat] += ms;
        cudaEventDestroy(mk.a);
        cudaEventDestroy(mk.b);
    }
    marks_.clear();
    std::vector<std::pair<std::string, float>> out(acc.begin(), acc.end());
    std::sort(out.begin(), out.end(), [](auto& a, auto& b) { return a.second > b.second; });
    return out;
}

void Model::mm(cudaStream_t s,
               bool tA,
               bool tB,
               int M,
               int N,
               int K,
               const bf16* A,
               int lda,
               const bf16* B,
               int ldb,
               void* D,
               int ldd,
               Out out,
               float beta,
               const Epilogue& epi) {
    tic(s, "gemm");
    gemm_.run(s, tA, tB, M, N, K, A, lda, B, ldb, D, ldd, out, beta, 1.f, epi);
    toc(s);
}

BankWrites Model::writes(int n, const Batch& b) {
    const LayerA& a = la_[n];
    (void)b;
    return BankWrites{a.k,  a.PV + c_.S2(), c_.ldPV(), a.info, loga_ + size_t(n) * c_.H * SLOTS, c_.B, c_.T,
                      c_.H, c_.bank_chunk,  a.ckpt,    c_.dh};
}

void Model::layer_fwd(int n, const Batch& b, cudaStream_t s) {
    const int M = c_.M(), D = c_.D, HD = c_.HD(), H = c_.H, F = c_.F, S2 = c_.S2();
    const LayerIdx& x = li_[n];
    const LayerW& w = lw_[n];
    LayerA& a = la_[n];
    float* st = stats_ + 6 * n;

    tic(s, "norm+elementwise");
    rmsnorm_fwd(h_, a.xh1, a.rstd1, M, D, s);
    toc(s);
    mm(s, false, false, M, c_.ldPV(), D, a.xh1, D, w.Wa, c_.ldPV(), a.PV, c_.ldPV(), Out::BF16);
    tic(s, "lru");
    lru_fwd(a.PV, c_.ldPV(), p(x.nu), p(x.theta), b.reset, a.s, a.sb, c_.B, c_.T, c_.N, s);
    toc(s);
    mm(s, false, false, M, c_.ldG(), S2, a.sb, S2, w.Ws, c_.ldG(), Gt_, c_.ldG(), Out::BF16);
    tic(s, "addr_fwd");
    addr_fwd(
        AddrFwd{Gt_,   c_.ldG(), p(x.bk), p(x.brg), p(x.bwg), p(x.R), b.reset, a.k, a.q,  a.kn, a.qn,  a.ridx,
                a.rp1, a.gw,     a.gr,    a.aw,     a.ar,     st,     a.info,  M,   c_.T, H,    c_.dh, c_.slots_of(n)},
        s);
    toc(s);
    const BankWrites bw = writes(n, b);
    if (a.ckpt) {
        tic(s, "bank_fwd");
        bank_checkpoint(bw, a.ckpt, carry_, cflags_, s);  // the adjoint's scratch is free in the forward
        toc(s);
    }
    tic(s, "bank_fwd");
    bank_fwd(bw, BankRead{a.q, a.gr, p(x.ltau), 2, a.o1, nullptr, nullptr, nullptr, nullptr, nullptr, a.zpre}, s);
    toc(s);
    if (!(silence_.size() > size_t(n) && (silence_[size_t(n)] & 1))) {  // analysis: a silenced read adds nothing
        mm(s, false, false, M, D, HD, a.o1, HD, w.Wo, D, h_, D, Out::F32, 1.f);
    }
    tic(s, "norm+elementwise");
    rmsnorm_fwd(h_, a.xh2, a.rstd2, M, D, s);
    toc(s);
    // U = gelu(xh2 W1' + b1) with the pre-activation Z kept (aux), then h += U W2 + b2: both in GEMM epilogues
    mm(s, false, false, M, F, D, a.xh2, D, w.W1, F, a.U, F, Out::BF16, 0.f,
       Epilogue{Epilogue::GeluAuxBias, w.b1, nullptr, a.Z, F});
    if (!(silence_.size() > size_t(n) && (silence_[size_t(n)] & 4))) {  // analysis: a silenced FFN adds nothing
        mm(s, false, false, M, D, F, a.U, F, w.W2, D, h_, D, Out::F32, 1.f, Epilogue{Epilogue::Bias, p(x.b2)});
    }
    if (c_.hops_of(n) == 1) { return; }
    tic(s, "norm+elementwise");
    rmsnorm_fwd(h_, a.xh3, a.rstd3, M, D, s);
    toc(s);
    mm(s, false, false, M, c_.ldG2(), D, a.xh3, D, w.Wq2, c_.ldG2(), G2t_, c_.ldG2(), Out::BF16);
    tic(s, "addr_q2");
    q2_fwd(Q2Fwd{G2t_, c_.ldG2(), p(x.brg2), a.q2, a.q2n, a.ar2, a.gr2, st + 4, a.info, M, H, c_.dh}, s);
    toc(s);
    tic(s, "bank_fwd");
    bank_fwd(bw, BankRead{a.q2, a.gr2, p(x.ltau2), 8, a.o2, nullptr, nullptr, nullptr, nullptr, nullptr}, s);
    toc(s);
    if (!(silence_.size() > size_t(n) && (silence_[size_t(n)] & 2))) {
        mm(s, false, false, M, D, HD, a.o2, HD, w.Wo2, D, h_, D, Out::F32, 1.f);
    }
}

void Model::layer_bwd(int n, const Batch& b, cudaStream_t s) {
    const int M = c_.M(), D = c_.D, HD = c_.HD(), H = c_.H, F = c_.F, S2 = c_.S2();
    const LayerIdx& x = li_[n];
    const LayerW& w = lw_[n];
    const LayerA& a = la_[n];
    const float* st = stats_ + 6 * n;
    const float count = float(M) * H;
    const BankWrites bw = writes(n, b);
    const bool two = c_.hops_of(n) == 2;

    // hop 2: h += o2 Wo2 (dhb = bf16(dh), written by the norm backward above). In a 1-hop layer the backward starts
    // at the FFN, and b2's gradient (the column sums of dh at the layer's output) came with the norm backward that
    // wrote dh (the next layer's norm1, or the final norm)
    if (two) {
        mm(s, false, true, M, HD, D, dhb_, D, w.Wo2, D, dO2_, HD, Out::F32);
        mm(s, true, false, HD, D, M, a.o2, HD, dhb_, D, g(x.Wo2), D, Out::F32, 1.f);
        tic(s, "bank_rewalk");
        bank_rewalk(bw, BankRead{a.q2, a.gr2, p(x.ltau2), 8, nullptr, dO2_, dq2_, dgr2_, coef2_, g(x.ltau2)}, s);
        toc(s);
        tic(s, "addr_q2");
        q2_bwd(Q2Bwd{dq2_, dgr2_, a.q2, a.q2n, a.ar2, g(x.brg2), st + 4, c_.budget, c_.budget_target, count, dG2_,
                     c_.ldG2(), M, H, c_.dh},
               s);
        toc(s);
        mm(s, false, true, M, D, c_.ldG2(), dG2_, c_.ldG2(), w.Wq2, c_.ldG2(), dxh_, D, Out::F32);
        mm(s, true, false, D, c_.ldG2(), M, a.xh3, D, dG2_, c_.ldG2(), dWp_, c_.ldG2(), Out::F32);
        tic(s, "norm+elementwise");
        unfold_grad(dWp_, p(x.Wq2), p(x.q2_norm), g(x.Wq2), g(x.q2_norm), D, c_.ldG2(), s);
        rmsnorm_bwd(dxh_, a.xh3, a.rstd3, dh_, dhb_, g(x.b2), M, D, true, s);  // and b2's gradient
        toc(s);
    }
    // FFN: h += W2 gelu(W1 rms(h) + b1) + b2; dZ = (dh W2^T) gelu'(Z) and b1's gradient from the GEMM epilogue
    mm(s, false, true, M, F, D, dhb_, D, w.W2, D, dZ_, F, Out::BF16, 0.f,
       Epilogue{Epilogue::DGeluBgrad, nullptr, db1b_, a.Z, F});
    mm(s, true, false, F, D, M, a.U, F, dhb_, D, g(x.W2), D, Out::F32, 1.f);
    tic(s, "norm+elementwise");
    add_to_f32(db1b_, g(x.b1), F, s);
    toc(s);
    mm(s, false, true, M, D, F, dZ_, F, w.W1, F, dxh_, D, Out::F32);
    mm(s, true, false, D, F, M, a.xh2, D, dZ_, F, dWp_, F, Out::F32);
    tic(s, "norm+elementwise");
    unfold_grad(dWp_, p(x.W1), p(x.ffn_norm), g(x.W1), g(x.ffn_norm), D, F, s);
    rmsnorm_bwd(dxh_, a.xh2, a.rstd2, dh_, dhb_, nullptr, M, D, true, s);
    toc(s);
    // hop 1: h += o1 Wo
    mm(s, false, true, M, HD, D, dhb_, D, w.Wo, D, dO1_, HD, Out::F32);
    mm(s, true, false, HD, D, M, a.o1, HD, dhb_, D, g(x.Wo), D, Out::F32, 1.f);
    tic(s, "bank_rewalk");
    bank_rewalk(bw, BankRead{a.q, a.gr, p(x.ltau), 2, nullptr, dO1_, dq1_, dgr1_, coef1_, g(x.ltau)}, s);
    toc(s);
    // the writes, for both reads
    tic(s, "bank_adjoint");
    bank_adjoint(bw,
                 BankAdjoint{{a.q, a.q2},
                             {dO1_, two ? dO2_ : nullptr},
                             {coef1_, two ? coef2_ : nullptr},
                             two ? 2 : 1,
                             dk2_,
                             dv2_,
                             dr2_,
                             carry_,
                             cflags_},
                 s);
    toc(s);
    tic(s, "addr_bwd");
    if (c_.ldG() != 2 * HD + 2 * H) {  // padding columns (small widths only) must stay zero for the GEMMs
        LBC_CUDA(cudaMemsetAsync(dG_, 0, size_t(M) * c_.ldG() * sizeof(bf16), s));
    }
    addr_bwd(AddrBwd{dk2_,     dv2_,    dr2_,     dq1_,     dgr1_,  a.gw,      dPV_ + S2,        c_.ldPV(), a.k,
                     a.q,      a.kn,    a.qn,     a.rp1,    a.aw,   a.ar,      a.ridx,           b.reset,   p(x.R),
                     g(x.R),   g(x.bk), g(x.brg), g(x.bwg), st,     c_.budget, c_.budget_target, count,     dG_,
                     c_.ldG(), M,       c_.T,     H,        a.zpre, c_.dh},
             s);
    toc(s);
    mm(s, false, true, M, S2, c_.ldG(), dG_, c_.ldG(), w.Ws, c_.ldG(), dS_, S2, Out::F32);
    mm(s, true, false, S2, c_.ldG(), M, a.sb, S2, dG_, c_.ldG(), g(x.Ws), c_.ldG(), Out::F32, 1.f);
    tic(s, "lru");
    lru_bwd(dS_, a.s, a.PV, c_.ldPV(), p(x.nu), p(x.theta), b.reset, dPV_, c_.ldPV(), g(x.nu), g(x.theta), c_.B, c_.T,
            c_.N, s);
    toc(s);
    mm(s, false, true, M, D, c_.ldPV(), dPV_, c_.ldPV(), w.Wa, c_.ldPV(), dxh_, D, Out::F32);
    mm(s, true, false, D, c_.ldPV(), M, a.xh1, D, dPV_, c_.ldPV(), dWp_, c_.ldPV(), Out::F32);
    tic(s, "norm+elementwise");
    unfold_grad(dWp_, p(x.Wa), p(x.norm1), g(x.Wa), g(x.norm1), D, c_.ldPV(), s);
    rmsnorm_bwd(dxh_, a.xh1, a.rstd1, dh_, dhb_, n > 0 ? b2_colsum(n - 1) : nullptr, M, D, true, s);
    toc(s);
}

float* Model::b2_colsum(int n) { return c_.hops_of(n) == 1 ? g(li_[n].b2) : nullptr; }

void Model::embed_fwd(const Batch& b, cudaStream_t s) {
    const int M = c_.M(), D = c_.D, E0 = c_.E0;
    tic(s, "embed");
    if (c_.factorized) {
        embed_gather(b.ids, Eb_, x0_, M, E0, s);
    } else {
        embed_gather_f32(b.ids, p(E_), h_, M, D, s);
    }
    toc(s);
    if (c_.factorized) { mm(s, false, false, M, D, E0, x0_, E0, Winb_, D, h_, D, Out::F32); }
}

// the final norm, the readout and the loss, one chunk of rows at a time (logits = hr E^T); with `backward`, also the
// readout's gradients and dh (into dh_ / dhb_)
void Model::readout_loss(const Batch& b, bool backward, cudaStream_t s) {
    const int M = c_.M(), D = c_.D, E0 = c_.E0, V = c_.V;
    const bf16* Eo = readout_table();
    float* gEo = g(c_.tied ? E_ : E_out_);
    tic(s, "norm+elementwise");
    rmsnorm_fwd(h_, xhF_, rstdF_, M, D, s);
    if (!c_.factorized) { scale_cols(xhF_, p(final_norm_), hr_, M, D, s); }
    toc(s);
    if (c_.factorized) { mm(s, false, false, M, E0, D, xhF_, D, Woutb_, E0, hr_, E0, Out::BF16); }
    const int Mc = std::min(c_.ce_chunk, M);
    for (int r0 = 0; r0 < M; r0 += Mc) {
        const int rows = std::min(Mc, M - r0);
        mm(s, false, true, rows, V, E0, hr_ + size_t(r0) * E0, E0, Eo, E0, logits_, V, Out::F32);
        tic(s, "ce_softmax");
        ce_rows(logits_, b.y + r0, backward ? dlog_ : nullptr, loss_, acc_, rows, V, b.inv_n, s,
                row_loss_ ? row_loss_ + r0 : nullptr, b.w ? b.w + r0 : nullptr, b.lp_old ? b.lp_old + r0 : nullptr,
                b.tis_cap, b.clip_low, b.clip_high);
        toc(s);
        if (!backward) { continue; }
        mm(s, false, false, rows, E0, V, dlog_, V, Eo, E0, dhrb_ + size_t(r0) * E0, E0, Out::BF16);
        mm(s, true, false, V, E0, rows, dlog_, V, hr_ + size_t(r0) * E0, E0, gEo, E0, Out::F32, 1.f);
    }
    if (!backward) { return; }
    if (c_.factorized) {
        mm(s, false, true, M, D, E0, dhrb_, E0, Woutb_, E0, dxh_, D, Out::F32);
        mm(s, true, false, D, E0, M, xhF_, D, dhrb_, E0, dWp_, E0, Out::F32);
    }
    tic(s, "norm+elementwise");
    if (c_.factorized) {
        unfold_grad(dWp_, p(W_out_), p(final_norm_), g(W_out_), g(final_norm_), D, E0, s);
    } else {
        scale_cols_bwd(dhrb_, xhF_, p(final_norm_), dxh_, g(final_norm_), M, D, s);
    }
    rmsnorm_bwd(dxh_, xhF_, rstdF_, dh_, dhb_, b2_colsum(c_.L - 1), M, D, false, s);
    toc(s);
}

void Model::forward_backward(const Batch& b, cudaStream_t s, bool accumulate) {
    const int M = c_.M(), D = c_.D, E0 = c_.E0;
    if (!accumulate) {  // a new update: the gradient and the step's metrics (loss, accuracy, gate stats) restart
        LBC_CUDA(cudaMemsetAsync(G_, 0, n_ * sizeof(float), s));
        LBC_CUDA(cudaMemsetAsync(metrics_, 0, size_t(2 + 6 * c_.L) * sizeof(float), s));
    }

    embed_fwd(b, s);
    for (int n = 0; n < c_.L; ++n) { layer_fwd(n, b, s); }
    readout_loss(b, true, s);
    for (int n = c_.L - 1; n >= 0; --n) { layer_bwd(n, b, s); }
    if (c_.factorized) {
        mm(s, false, true, M, E0, D, dhb_, D, Winb_, D, dx0_, E0, Out::F32);
        mm(s, true, false, E0, D, M, x0_, E0, dhb_, D, g(W_in_), D, Out::F32, 1.f);
    }
    tic(s, "embed");
    embed_scatter(b.ids, c_.factorized ? dx0_ : dh_, g(E_), M, E0, s);
    toc(s);
}

void Model::keep_row_loss() {
    if (!row_loss_) { row_loss_ = alloc<float>(size_t(c_.M())); }
}

std::vector<float> Model::row_loss(cudaStream_t s) {
    std::vector<float> out(size_t(c_.M()), 0.f);
    if (row_loss_) {
        LBC_CUDA(cudaMemcpyAsync(out.data(), row_loss_, out.size() * 4, cudaMemcpyDeviceToHost, s));
        LBC_CUDA(cudaStreamSynchronize(s));
    }
    return out;
}

std::vector<bf16> Model::download_read(int n, int which, cudaStream_t s) {
    const LayerA& a = la_[size_t(n)];
    const bf16* p = which == 0 ? a.q : which == 1 ? a.q2 : which == 2 ? a.o1 : which == 3 ? a.o2 : a.k;
    std::vector<bf16> out;
    if (!p) { return out; }
    out.resize(size_t(c_.M()) * c_.HD());
    LBC_CUDA(cudaMemcpyAsync(out.data(), p, out.size() * sizeof(bf16), cudaMemcpyDeviceToHost, s));
    LBC_CUDA(cudaStreamSynchronize(s));
    return out;
}

std::vector<float> Model::readout_input(int r, cudaStream_t s) {
    std::vector<bf16> hb(size_t(c_.E0));
    LBC_CUDA(cudaMemcpyAsync(hb.data(), hr_ + size_t(r) * c_.E0, hb.size() * sizeof(bf16), cudaMemcpyDeviceToHost, s));
    LBC_CUDA(cudaStreamSynchronize(s));
    std::vector<float> out(hb.size());
    for (size_t i = 0; i < hb.size(); ++i) { out[i] = __bfloat162float(hb[i]); }
    return out;
}

void Model::forward_loss(const Batch& b, cudaStream_t s) {
    LBC_CUDA(cudaMemsetAsync(metrics_, 0, size_t(2 + 6 * c_.L) * sizeof(float), s));
    embed_fwd(b, s);
    for (int n = 0; n < c_.L; ++n) { layer_fwd(n, b, s); }
    readout_loss(b, false, s);
}

void Model::optimizer_step(cudaStream_t s) {
    tic(s, "optimizer");
    LBC_CUDA(cudaMemsetAsync(gsq_, 0, sizeof(float), s));
    sumsq(G_, n_, gsq_, s);
    increment(step_, s);
    adamw(P_, G_, M1_, M2_, decay_, n_, schedule(), c_.beta1, c_.beta2, c_.eps, c_.wd, step_, gsq_, c_.clip, lr_, s);
    prepare_weights(s);
    toc(s);
}

StepMetrics Model::decode_metrics(const float* r) const {
    StepMetrics m{};
    m.loss = r[0];
    m.accuracy = r[1];
    const float count = float(c_.M()) * c_.H * float(accum_);  // gate stats summed over the update's micro-batches
    int twos = 0;
    for (int n = 0; n < c_.L; ++n) {
        const float* st = r + 2 + 6 * n;
        m.open_w += st[1] / count / c_.L;
        m.open_r1 += st[3] / count / c_.L;
        if (c_.hops_of(n) == 2) {
            m.open_r2 += st[5] / count;
            ++twos;
        }
    }
    m.open_r2 = twos ? m.open_r2 / twos : -1.f;
    m.grad_norm = std::sqrt(r[2 + 6 * c_.L]);
    m.lr = r[3 + 6 * c_.L];
    return m;
}

size_t Model::embedding_params() const {
    size_t n = size_t(c_.V) * c_.E0;
    return c_.tied ? n : 2 * n;
}

std::vector<float> Model::gate_stats(cudaStream_t s) {
    std::vector<float> st(size_t(6) * c_.L), out;
    LBC_CUDA(cudaMemcpyAsync(st.data(), stats_, st.size() * 4, cudaMemcpyDeviceToHost, s));
    LBC_CUDA(cudaStreamSynchronize(s));
    const float count = float(c_.M()) * c_.H;
    for (int n = 0; n < c_.L; ++n) {
        out.push_back(st[6 * n + 1] / count);
        out.push_back(st[6 * n + 3] / count);
        out.push_back(st[6 * n + 5] / count);
    }
    return out;
}

void Model::decisions(int n,
                      std::vector<uint8_t>& ridx,
                      std::vector<uint8_t>& gw,
                      std::vector<uint8_t>& gr,
                      std::vector<uint8_t>& gr2,
                      cudaStream_t s) {
    const size_t MH = size_t(c_.M()) * c_.H;
    ridx.resize(MH * 2);
    gw.resize(MH);
    gr.resize(MH);
    gr2.resize(MH);
    LBC_CUDA(cudaMemcpyAsync(ridx.data(), la_[n].ridx, MH * 2, cudaMemcpyDeviceToHost, s));
    LBC_CUDA(cudaMemcpyAsync(gw.data(), la_[n].gw, MH, cudaMemcpyDeviceToHost, s));
    LBC_CUDA(cudaMemcpyAsync(gr.data(), la_[n].gr, MH, cudaMemcpyDeviceToHost, s));
    if (la_[n].gr2) {
        LBC_CUDA(cudaMemcpyAsync(gr2.data(), la_[n].gr2, MH, cudaMemcpyDeviceToHost, s));
    } else {
        std::fill(gr2.begin(), gr2.end(), uint8_t(0));
    }
    LBC_CUDA(cudaStreamSynchronize(s));
}

}  // namespace lbc
