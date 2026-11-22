#include "decode.hpp"

#include <algorithm>
#include <cstring>

#include "kernels.hpp"
#include "lbc/check.hpp"

namespace lbc {

template <class T>
T* Decoder::alloc(size_t n) {
    T* p = nullptr;
    LBC_CUDA(cudaMalloc(&p, std::max<size_t>(n, 1) * sizeof(T)));
    LBC_CUDA(cudaMemset(p, 0, std::max<size_t>(n, 1) * sizeof(T)));
    allocs_.push_back(p);
    return p;
}

Decoder::Decoder(Model& model, int slots, int max_rows) : m_(model), slots_(slots), max_rows_(max_rows) {
    const Config& c = m_.c_;
    const size_t S = size_t(slots_) + 1, R = size_t(max_rows_);
    for (int n = 0; n < c.L; ++n) {
        lru_.push_back(alloc<float>(S * c.S2()));
        gprev_.push_back(alloc<bf16>(S * c.HD()));
        bank_.push_back(alloc<float>(S * c.H * ckpt_fields(c.dh) * SLOTS));
    }
    d_ids_ = alloc<int>(R);
    d_map_ = alloc<int>(R);
    d_tpos_ = alloc<int>(R);
    d_tok_ = alloc<int>(R);
    d_reset_ = alloc<uint8_t>(R);
    d_key_ = alloc<uint64_t>(R);
    d_logp_ = alloc<float>(R);
    d_ent_ = alloc<float>(R);
    d_pairs_ = nullptr;
    h_ = alloc<float>(R * c.D);
    rstd_ = alloc<float>(R);
    kn_ = alloc<float>(R * c.H);
    qn_ = alloc<float>(R * c.H);
    rp1_ = alloc<float>(R * c.H);
    aw_ = alloc<float>(R * c.H);
    ar_ = alloc<float>(R * c.H);
    q2n_ = alloc<float>(R * c.H);
    ar2_ = alloc<float>(R * c.H);
    stats_ = alloc<float>(8);
    logit_rows_ = std::min(max_rows_, 1024);
    logits_ = alloc<float>(size_t(logit_rows_) * c.V);
    x0_ = c.factorized ? alloc<bf16>(R * c.E0) : nullptr;
    xh_ = alloc<bf16>(R * c.D);
    PV_ = alloc<bf16>(R * c.ldPV());
    sb_ = alloc<bf16>(R * c.S2());
    G_ = alloc<bf16>(R * c.ldG());
    k_ = alloc<bf16>(R * c.HD());
    q_ = alloc<bf16>(R * c.HD());
    o_ = alloc<bf16>(R * c.HD());
    U_ = alloc<bf16>(R * c.F);
    Z_ = alloc<bf16>(R * c.F);
    G2_ = alloc<bf16>(R * c.ldG2());
    q2_ = alloc<bf16>(R * c.HD());
    hr_ = alloc<bf16>(R * c.E0);
    ridx_ = alloc<uint8_t>(R * c.H * 2);
    gw_ = alloc<uint8_t>(R * c.H);
    gr_ = alloc<uint8_t>(R * c.H);
    gr2_ = alloc<uint8_t>(R * c.H);
    info_ = alloc<uint2>(R * c.H);
    LBC_CUDA(cudaMallocHost(&h_in_, R * 4 * sizeof(int) + R * sizeof(uint64_t)));
    LBC_CUDA(cudaMallocHost(&h_out_, R * sizeof(int)));
    LBC_CUDA(cudaMallocHost(&h_outf_, R * 2 * sizeof(float)));
}

Decoder::~Decoder() {
    for (void* p : allocs_) cudaFree(p);
    if (d_pairs_) cudaFree(d_pairs_);
    cudaFreeHost(h_in_);
    cudaFreeHost(h_out_);
    cudaFreeHost(h_outf_);
}

size_t Decoder::state_bytes() const {
    const Config& c = m_.c_;
    return size_t(slots_ + 1) * c.L * (c.S2() * 4 + c.HD() * 2 + size_t(c.H) * ckpt_fields(c.dh) * SLOTS * 4);
}

void Decoder::copy_slots(const std::vector<int2>& pairs, cudaStream_t s) {
    if (pairs.empty()) return;
    const Config& c = m_.c_;
    if (int(pairs.size()) > pairs_cap_) {
        if (d_pairs_) {
            LBC_CUDA(cudaStreamSynchronize(s));
            cudaFree(d_pairs_);
        }
        pairs_cap_ = int(pairs.size());
        LBC_CUDA(cudaMalloc(&d_pairs_, size_t(pairs_cap_) * sizeof(int2)));
    }
    LBC_CUDA(cudaMemcpyAsync(d_pairs_, pairs.data(), pairs.size() * sizeof(int2), cudaMemcpyHostToDevice, s));
    const int n = int(pairs.size());
    for (int l = 0; l < c.L; ++l) {
        copy_rows(lru_[size_t(l)], size_t(c.S2()) * 4, d_pairs_, n, s);
        copy_rows(gprev_[size_t(l)], size_t(c.HD()) * 2, d_pairs_, n, s);
        copy_rows(bank_[size_t(l)], size_t(c.H) * ckpt_fields(c.dh) * SLOTS * 4, d_pairs_, n, s);
    }
    LBC_CUDA(cudaStreamSynchronize(s));  // the pairs buffer is reused by the next call
}

void Decoder::step(const int* ids, const int* slot, const int* tpos, const uint8_t* reset, int R, float temperature,
                   const uint64_t* key, int* tok, float* logp, float* ent, cudaStream_t s) {
    if (R <= 0) return;
    if (R > max_rows_) fail("Decoder::step", "more rows than max_rows", __FILE__, __LINE__);
    Model& m = m_;
    const Config& c = m.c_;
    LBC_CUDA(cudaStreamSynchronize(s));  // the pinned staging buffer is about to be rewritten
    // rows padded to a multiple of 64 (few GEMM shapes, so few plans): padding rows use the scratch slot
    const int Rp = std::min(max_rows_, (R + 63) / 64 * 64);
    int* hi = h_in_;
    for (int r = 0; r < Rp; ++r) {
        const bool real = r < R;
        hi[r] = real ? ids[r] : 0;
        hi[Rp + r] = real ? slot[r] : slots_;
        hi[2 * Rp + r] = real ? tpos[r] : 0;
        reinterpret_cast<uint8_t*>(hi + 3 * Rp)[r] = real ? reset[r] : 1;
    }
    uint64_t* hk = reinterpret_cast<uint64_t*>(hi + 4 * Rp);
    if (key)
        for (int r = 0; r < Rp; ++r) hk[r] = r < R ? key[r] : 0;
    LBC_CUDA(cudaMemcpyAsync(d_ids_, hi, size_t(Rp) * 4, cudaMemcpyHostToDevice, s));
    LBC_CUDA(cudaMemcpyAsync(d_map_, hi + Rp, size_t(Rp) * 4, cudaMemcpyHostToDevice, s));
    LBC_CUDA(cudaMemcpyAsync(d_tpos_, hi + 2 * Rp, size_t(Rp) * 4, cudaMemcpyHostToDevice, s));
    LBC_CUDA(cudaMemcpyAsync(d_reset_, hi + 3 * Rp, size_t(Rp), cudaMemcpyHostToDevice, s));
    if (key) LBC_CUDA(cudaMemcpyAsync(d_key_, hk, size_t(Rp) * 8, cudaMemcpyHostToDevice, s));

    const int D = c.D, HD = c.HD(), H = c.H, F = c.F, S2 = c.S2(), E0 = c.E0;
    if (c.factorized) {
        embed_gather(d_ids_, m.Eb_, x0_, Rp, E0, s);
        m.mm(s, false, false, Rp, D, E0, x0_, E0, m.Winb_, D, h_, D, Out::F32);
    } else {
        embed_gather_f32(d_ids_, m.p(m.E_), h_, Rp, D, s);
    }
    for (int n = 0; n < c.L; ++n) {
        const auto& x = m.li_[size_t(n)];
        const auto& w = m.lw_[size_t(n)];
        rmsnorm_fwd(h_, xh_, rstd_, Rp, D, s);
        m.mm(s, false, false, Rp, c.ldPV(), D, xh_, D, w.Wa, c.ldPV(), PV_, c.ldPV(), Out::BF16);
        lru_step(PV_, c.ldPV(), m.p(x.nu), m.p(x.theta), d_reset_, d_map_, lru_[size_t(n)], sb_, Rp, c.N, s);
        m.mm(s, false, false, Rp, c.ldG(), S2, sb_, S2, w.Ws, c.ldG(), G_, c.ldG(), Out::BF16);
        AddrFwd af{G_,  c.ldG(), m.p(x.bk), m.p(x.brg), m.p(x.bwg), m.p(x.R), d_reset_, k_,  q_, kn_,
                   qn_, ridx_,   rp1_,      gw_,        gr_,        aw_,       ar_,     stats_, info_, Rp,
                   1,   H,       c.dh,      c.slots_of(n)};
        af.Gprev = gprev_[size_t(n)];
        af.ldGprev = HD;
        af.map = d_map_;
        af.tpos = d_tpos_;
        addr_fwd(af, s);
        store_rows(G_, c.ldG(), HD, d_map_, gprev_[size_t(n)], HD, Rp, s);  // after addr_fwd read the previous
        const BankWrites bw{k_, PV_ + S2, c.ldPV(), info_, m.loga_ + size_t(n) * H * SLOTS, Rp, 1, H, 1, nullptr,
                            c.dh};
        bank_step(bw, BankRead{q_, gr_, m.p(x.ltau), 2, o_, nullptr, nullptr, nullptr, nullptr, nullptr}, d_map_,
                  d_tpos_, bank_[size_t(n)], true, s);
        m.mm(s, false, false, Rp, D, HD, o_, HD, w.Wo, D, h_, D, Out::F32, 1.f);
        rmsnorm_fwd(h_, xh_, rstd_, Rp, D, s);
        m.mm(s, false, false, Rp, F, D, xh_, D, w.W1, F, U_, F, Out::BF16, 0.f,
             Epilogue{Epilogue::GeluAuxBias, w.b1, nullptr, Z_, F});
        m.mm(s, false, false, Rp, D, F, U_, F, w.W2, D, h_, D, Out::F32, 1.f, Epilogue{Epilogue::Bias, m.p(x.b2)});
        if (c.hops_of(n) == 1) continue;
        rmsnorm_fwd(h_, xh_, rstd_, Rp, D, s);
        m.mm(s, false, false, Rp, c.ldG2(), D, xh_, D, w.Wq2, c.ldG2(), G2_, c.ldG2(), Out::BF16);
        q2_fwd(Q2Fwd{G2_, c.ldG2(), m.p(x.brg2), q2_, q2n_, ar2_, gr2_, stats_ + 4, info_, Rp, H, c.dh}, s);
        bank_step(bw, BankRead{q2_, gr2_, m.p(x.ltau2), 8, o_, nullptr, nullptr, nullptr, nullptr, nullptr}, d_map_,
                  d_tpos_, bank_[size_t(n)], false, s);
        m.mm(s, false, false, Rp, D, HD, o_, HD, w.Wo2, D, h_, D, Out::F32, 1.f);
    }
    rmsnorm_fwd(h_, xh_, rstd_, Rp, D, s);
    if (c.factorized)
        m.mm(s, false, false, Rp, E0, D, xh_, D, m.Woutb_, E0, hr_, E0, Out::BF16);
    else
        scale_cols(xh_, m.p(m.final_norm_), hr_, Rp, D, s);
    for (int r0 = 0; r0 < R; r0 += logit_rows_) {
        const int rows = std::min(logit_rows_, R - r0);
        m.mm(s, false, true, rows, c.V, E0, hr_ + size_t(r0) * E0, E0, m.readout_table(), E0, logits_, c.V, Out::F32);
        if (key) sample_rows(logits_, rows, c.V, temperature, d_key_ + r0, d_tok_ + r0, d_logp_ + r0, d_ent_ + r0, s);
    }
    if (!key) return;
    LBC_CUDA(cudaMemcpyAsync(h_out_, d_tok_, size_t(R) * 4, cudaMemcpyDeviceToHost, s));
    LBC_CUDA(cudaMemcpyAsync(h_outf_, d_logp_, size_t(R) * 4, cudaMemcpyDeviceToHost, s));
    LBC_CUDA(cudaMemcpyAsync(h_outf_ + R, d_ent_, size_t(R) * 4, cudaMemcpyDeviceToHost, s));
    LBC_CUDA(cudaStreamSynchronize(s));
    std::memcpy(tok, h_out_, size_t(R) * 4);
    if (logp) std::memcpy(logp, h_outf_, size_t(R) * 4);
    if (ent) std::memcpy(ent, h_outf_ + R, size_t(R) * 4);
}

}  // namespace lbc
