// The banks, as an exact walk per (row, head): one block of SLOTS threads, thread m owns slot m.
//
// Mass normalization makes a slot's content its weighted average of writes: the per-token decay scales the summed
// keys / values and the mass alike, so the averages change only when the slot is written. Thread m keeps the average
// key Kb, the average value Vb, the mass Zs at its last write and that write's time tm; the decay enters only the
// write update and the validity test (mass now = Zs a^(t - tm) > BANK_EPS). A step with a closed read costs the
// two routed threads' writes and no barrier; an open read is the softmax over the slots of tau q . Kb_m, then
// sum_m p_m Vb_m (one block barrier).
//
// Backward, per layer: a re-walk per read recomputes the softmax and stores the read's local gradient on each
// slot's linear state (K = Zs Kb, V, Z): dK_m = alpha q, dV_m = beta dy, dZ_m = gamma (bf16: same range as fp32,
// half the traffic of the 3 x SLOTS coefficients per open read); then one reverse adjoint walk
// for both reads. The adjoint has no cross-slot coupling at all (each thread carries its slot's lambda, decayed
// lazily: lambda_t = local_t + a lambda_{t+1}), so it runs without barriers; the routed threads write their slot's
// contributions to the write gradients (k, v, routing weight, write gate) into per-slot buffers that the
// addressing backward sums.

#include <cmath>

#include "kernels.hpp"
#include "lbc/check.hpp"

namespace lbc {
namespace {

constexpr unsigned FULL = 0xffffffffu;
constexpr int NWARP = SLOTS / 32;

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o; o >>= 1) { v += __shfl_xor_sync(FULL, v, o); }
    return v;
}

__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
    for (int o = 16; o; o >>= 1) { v = fmaxf(v, __shfl_xor_sync(FULL, v, o)); }
    return v;
}

template <int O>
__device__ __forceinline__ void reduce_scatter_step(float (&u)[32], int lane) {
    const bool upper = lane & O;
#pragma unroll
    for (int j = 0; j < O; ++j) {
        const float send = upper ? u[j] : u[j + O];
        const float keep = upper ? u[j + O] : u[j];
        u[j] = keep + __shfl_xor_sync(FULL, send, O);
    }
}

// Sum over the warp of the lanes' vectors u[0..31]; lane d returns component d (31 shuffles).
__device__ __forceinline__ float reduce_scatter32(float (&u)[32], int lane) {
    reduce_scatter_step<16>(u, lane);
    reduce_scatter_step<8>(u, lane);
    reduce_scatter_step<4>(u, lane);
    reduce_scatter_step<2>(u, lane);
    reduce_scatter_step<1>(u, lane);
    return u[0];
}

// 32 bf16 (64-byte aligned) -> floats
__device__ __forceinline__ void load32(const bf16* p, float (&x)[32]) {
    const uint4* q = reinterpret_cast<const uint4*>(p);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const uint4 u = q[i];
        const unsigned w[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            x[8 * i + 2 * j] = __uint_as_float(w[j] << 16);
            x[8 * i + 2 * j + 1] = __uint_as_float(w[j] & 0xffff0000u);
        }
    }
}

// 32 floats (128-byte aligned)
__device__ __forceinline__ void load32(const float* p, float (&x)[32]) {
    const float4* q = reinterpret_cast<const float4*>(p);
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const float4 u = q[i];
        x[4 * i] = u.x;
        x[4 * i + 1] = u.y;
        x[4 * i + 2] = u.z;
        x[4 * i + 3] = u.w;
    }
}

// scale * x as 32 bf16 (64-byte aligned)
__device__ __forceinline__ void store32(bf16* p, const float* x, float scale) {
    uint4* q = reinterpret_cast<uint4*>(p);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        unsigned w[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const __nv_bfloat162 v = __floats2bfloat162_rn(scale * x[8 * i + 2 * j], scale * x[8 * i + 2 * j + 1]);
            w[j] = *reinterpret_cast<const unsigned*>(&v);
        }
        q[i] = make_uint4(w[0], w[1], w[2], w[3]);
    }
}

struct Info {
    int m1, m2, flags;
    float p1;
};

__device__ __forceinline__ Info unpack(uint2 w) {
    return Info{int(w.x & 0xffu), int((w.x >> 8) & 0xffu), int((w.x >> 16) & 0xffu), __uint_as_float(w.y)};
}

// Head sizes DH = 32 or 64 (templates): a slot's state is DH-wide per thread; per-lane vectors (loads, the
// reduce-scatter, outputs) go in chunks of 32, lane l holding components l + 32 c, so temporaries stay 32 wide.
template <int DH>
constexpr int ckpt_fields_t() {
    return 2 * DH + 2;
}

// the write of token t into this thread's slot with weight r: running averages of the decayed writes
template <int DH>
__device__ __forceinline__ void write_slot(float (&Kb)[DH],
                                           float (&Vb)[DH],
                                           float& Zs,
                                           int& tm,
                                           float la,
                                           int t,
                                           float r,
                                           const bf16* k,
                                           const bf16* v) {
    const float zn = Zs * __expf(la * float(t - tm)) + r;
    const float c = r / zn;
#pragma unroll
    for (int ch = 0; ch < DH / 32; ++ch) {
        float x[32];
        load32(k + 32 * ch, x);
#pragma unroll
        for (int d = 0; d < 32; ++d) { Kb[32 * ch + d] += c * (x[d] - Kb[32 * ch + d]); }
        load32(v + 32 * ch, x);
#pragma unroll
        for (int d = 0; d < 32; ++d) { Vb[32 * ch + d] += c * (x[d] - Vb[32 * ch + d]); }
    }
    Zs = zn;
    tm = t;
}

// x . y over DH, x loaded from p (bf16 or float) 32 at a time
template <int DH, class P>
__device__ __forceinline__ float dot_load(const P* p, const float (&y)[DH]) {
    float s = 0.f;
#pragma unroll
    for (int ch = 0; ch < DH / 32; ++ch) {
        float x[32];
        load32(p + 32 * ch, x);
#pragma unroll
        for (int d = 0; d < 32; ++d) { s += x[d] * y[32 * ch + d]; }
    }
    return s;
}

// y += a * x over DH, x loaded from p
template <int DH, class P>
__device__ __forceinline__ void axpy_load(const P* p, float a, float (&y)[DH]) {
#pragma unroll
    for (int ch = 0; ch < DH / 32; ++ch) {
        float x[32];
        load32(p + 32 * ch, x);
#pragma unroll
        for (int d = 0; d < 32; ++d) { y[32 * ch + d] += a * x[d]; }
    }
}

// the warp sum of s * y (DH-wide): lane l gets components l + 32 c in out[c]
template <int DH>
__device__ __forceinline__ void warp_scatter(const float (&y)[DH], float s, int lane, float (&out)[DH / 32]) {
#pragma unroll
    for (int ch = 0; ch < DH / 32; ++ch) {
        float u[32];
#pragma unroll
        for (int d = 0; d < 32; ++d) { u[d] = s * y[32 * ch + d]; }
        out[ch] = reduce_scatter32(u, lane);
    }
}

// a slot state <-> a checkpoint record (field-major: consecutive slots are consecutive floats)
template <int DH>
__device__ __forceinline__ void ckpt_store(float* base,
                                           int m,
                                           const float (&A)[DH],
                                           const float (&Bv)[DH],
                                           float z,
                                           float extra) {
#pragma unroll
    for (int d = 0; d < DH; ++d) {
        base[d * SLOTS + m] = A[d];
        base[(DH + d) * SLOTS + m] = Bv[d];
    }
    base[2 * DH * SLOTS + m] = z;
    base[(2 * DH + 1) * SLOTS + m] = extra;
}

template <int DH>
__device__ __forceinline__ void ckpt_load(const float* base,
                                          int m,
                                          float (&A)[DH],
                                          float (&Bv)[DH],
                                          float& z,
                                          float& extra) {
#pragma unroll
    for (int d = 0; d < DH; ++d) {
        A[d] = base[d * SLOTS + m];
        Bv[d] = base[(DH + d) * SLOTS + m];
    }
    z = base[2 * DH * SLOTS + m];
    extra = base[(2 * DH + 1) * SLOTS + m];
}

template <int DH>
__device__ __forceinline__ size_t ckpt_offset(int b, int h, int c, int H, int nc) {
    return ((size_t(b) * H + h) * nc + c) * ckpt_fields_t<DH>() * SLOTS;
}

// the state update of token t (reset, then the write when this thread's slot is routed and the write gate open)
template <int DH>
__device__ __forceinline__ void step_state(float (&Kb)[DH],
                                           float (&Vb)[DH],
                                           float& Zs,
                                           int& tm,
                                           float la,
                                           int t,
                                           int m,
                                           const Info& si,
                                           const BankWrites& w,
                                           size_t r,
                                           int h) {
    if (si.flags & 4) { Zs = 0.f; }
    // a confident router rounds p1 to 1 and the second weight to exactly 0: that write is a no-op (written, an empty
    // slot would get r / zn = 0 / 0, and the NaN reaches every later read through 0 * NaN)
    const float rw = m == si.m1 ? si.p1 : 1.f - si.p1;
    if ((si.flags & 1) && (m == si.m1 || m == si.m2) && rw > 0.f) {
        write_slot(Kb, Vb, Zs, tm, la, t, rw, w.k + (r * w.H + h) * DH, w.v + r * w.ldv + h * DH);
    }
}

// the state at the start of chunk c (zero for the first chunk)
template <int DH>
__device__ __forceinline__ void
chunk_start(const BankWrites& w, int b, int h, int c, int m, float (&Kb)[DH], float (&Vb)[DH], float& Zs, int& tm) {
    if (c == 0) {
#pragma unroll
        for (int d = 0; d < DH; ++d) { Kb[d] = Vb[d] = 0.f; }
        Zs = 0.f;
        tm = 0;
    } else {
        float tmf;
        ckpt_load(w.ckpt + ckpt_offset<DH>(b, h, c, w.H, bank_chunks(w)), m, Kb, Vb, Zs, tmf);
        tm = __float_as_int(tmf);
    }
}

// Checkpoints in two passes. Local: each chunk's slot states from empty, its writes only (no reads, no barriers),
// with a flag when a reset occurs in it. Merge: per slot, from the first chunk on, state(c + 1) = merge(state(c),
// local(c)); mass-weighted averages compose exactly: the earlier mass decayed to the later last write time, then the
// mass-weighted mean of the averages; a reset in chunk c discards state(c).
template <int DH>
__global__ void __launch_bounds__(SLOTS) k_bank_ckpt_local(BankWrites w, float* local, int* rflags) {
    const int b = blockIdx.x, h = blockIdx.y, c = blockIdx.z, m = threadIdx.x;
    const int T = w.T, H = w.H, nc = bank_chunks(w), t0 = c * w.chunk, t1 = min(T, t0 + w.chunk);
    if (c == nc - 1) {
        return;  // the last chunk's writes reach no checkpoint
    }
    const float la = w.loga[h * SLOTS + m];
    float Kb[DH], Vb[DH], Zs = 0.f;
    int tm = t0;
    bool reset = false;
#pragma unroll
    for (int d = 0; d < DH; ++d) { Kb[d] = Vb[d] = 0.f; }
    const size_t base = size_t(b) * T;
    uint2 nxt = w.info[(base + t0) * H + h];
    for (int t = t0; t < t1; ++t) {
        const Info si = unpack(nxt);
        if (t + 1 < t1) { nxt = w.info[(base + t + 1) * H + h]; }
        reset |= (si.flags & 4) != 0;
        step_state(Kb, Vb, Zs, tm, la, t, m, si, w, base + t, h);
    }
    ckpt_store(local + ckpt_offset<DH>(b, h, c, H, nc), m, Kb, Vb, Zs, __int_as_float(tm));
    if (m == 0) { rflags[(size_t(b) * H + h) * nc + c] = reset; }
}

template <int DH>
__global__ void __launch_bounds__(SLOTS)
    k_bank_ckpt_merge(BankWrites w, const float* local, const int* rflags, float* ckpt) {
    const int b = blockIdx.x, h = blockIdx.y, m = threadIdx.x;
    const int H = w.H, nc = bank_chunks(w);
    const float la = w.loga[h * SLOTS + m];
    float Kb[DH], Vb[DH], Zs = 0.f;
    int tm = 0;
#pragma unroll
    for (int d = 0; d < DH; ++d) { Kb[d] = Vb[d] = 0.f; }
    for (int c = 0; c + 1 < nc; ++c) {
        float LK[DH], LV[DH], LZ, ltmf;
        ckpt_load(local + ckpt_offset<DH>(b, h, c, H, nc), m, LK, LV, LZ, ltmf);
        const int ltm = __float_as_int(ltmf);
        if (rflags[(size_t(b) * H + h) * nc + c]) {  // state(c + 1) = local(c)
#pragma unroll
            for (int d = 0; d < DH; ++d) {
                Kb[d] = LK[d];
                Vb[d] = LV[d];
            }
            Zs = LZ;
            tm = ltm;
        } else if (LZ > 0.f) {
            const float za = Zs * __expf(la * float(ltm - tm));
            const float zn = za + LZ;
            const float fa = za / zn, fb = LZ / zn;
#pragma unroll
            for (int d = 0; d < DH; ++d) {
                Kb[d] = fa * Kb[d] + fb * LK[d];
                Vb[d] = fa * Vb[d] + fb * LV[d];
            }
            Zs = zn;
            tm = ltm;
        }
        ckpt_store(ckpt + ckpt_offset<DH>(b, h, c + 1, H, nc), m, Kb, Vb, Zs, __int_as_float(tm));
    }
}

template <int DH>
__global__ void __launch_bounds__(SLOTS) k_bank_fwd(BankWrites w, BankRead rd) {
    const int b = blockIdx.x, h = blockIdx.y, m = threadIdx.x, lane = m & 31, warp = m >> 5;
    const int T = w.T, H = w.H, t0 = blockIdx.z * w.chunk, t1 = min(T, t0 + w.chunk);
    const float la = w.loga[h * SLOTS + m];
    const float tau = __expf(rd.log_tau[h]);
    __shared__ float s_wm[2][NWARP], s_ws[2][NWARP], s_wo[2][NWARP][DH];
    float Kb[DH], Vb[DH], Zs;
    int tm, nopen = 0;
    chunk_start(w, b, h, blockIdx.z, m, Kb, Vb, Zs, tm);
    const size_t base = size_t(b) * T;
    uint2 nxt = w.info[(base + t0) * H + h];
    for (int t = t0; t < t1; ++t) {
        const Info si = unpack(nxt);
        if (t + 1 < t1) { nxt = w.info[(base + t + 1) * H + h]; }
        const size_t r = base + t, rh = r * H + h;
        if (rd.zpre && (m == si.m1 || m == si.m2)) {  // the routed slots' mass before this token's write
            rd.zpre[rh * 2 + (m == si.m1 ? 0 : 1)] = (si.flags & 4) ? 0.f : Zs * __expf(la * float(t - tm));
        }
        step_state(Kb, Vb, Zs, tm, la, t, m, si, w, r, h);
        bf16* out = rd.o + r * H * DH + h * DH;
        if (si.flags & rd.gate_bit) {
            const bool valid = Zs * __expf(la * float(t - tm)) > BANK_EPS;
            float sc = -INFINITY;
            if (valid) { sc = tau * dot_load<DH>(rd.q + rh * DH, Kb); }
            const float wmax = warp_max(sc);
            const float e = valid ? __expf(sc - wmax) : 0.f;
            const float wsum = warp_sum(e);
            float od[DH / 32];
            warp_scatter<DH>(Vb, e, lane, od);
            const int buf = nopen++ & 1;
            if (lane == 0) {
                s_wm[buf][warp] = wmax;
                s_ws[buf][warp] = wsum;
            }
#pragma unroll
            for (int c = 0; c < DH / 32; ++c) { s_wo[buf][warp][lane + 32 * c] = od[c]; }
            __syncthreads();
            if (warp == 0) {
                float M = -INFINITY;
#pragma unroll
                for (int i = 0; i < NWARP; ++i) { M = fmaxf(M, s_wm[buf][i]); }
                float o[DH / 32];
#pragma unroll
                for (int c = 0; c < DH / 32; ++c) { o[c] = 0.f; }
                if (M > -INFINITY) {
                    float S = 0.f;
#pragma unroll
                    for (int i = 0; i < NWARP; ++i) {
                        const float f = s_wm[buf][i] > -INFINITY ? __expf(s_wm[buf][i] - M) : 0.f;
                        S += s_ws[buf][i] * f;
#pragma unroll
                        for (int c = 0; c < DH / 32; ++c) { o[c] += s_wo[buf][i][lane + 32 * c] * f; }
                    }
#pragma unroll
                    for (int c = 0; c < DH / 32; ++c) { o[c] /= S; }
                }
#pragma unroll
                for (int c = 0; c < DH / 32; ++c) { out[lane + 32 * c] = __float2bfloat16(o[c]); }
            }
        } else if (warp == 0) {
#pragma unroll
            for (int c = 0; c < DH / 32; ++c) { out[lane + 32 * c] = __float2bfloat16(0.f); }
        }
    }
}

// incremental decoding: one token of row j (blockIdx.x), head h: the state from slot map[j], the write (when
// `write`), then the read; the read's math is k_bank_fwd's
template <int DH>
__global__ void __launch_bounds__(SLOTS)
    k_bank_step(BankWrites w, BankRead rd, const int* map, const int* tpos, float* state, int write) {
    const int j = blockIdx.x, h = blockIdx.y, m = threadIdx.x, lane = m & 31, warp = m >> 5;
    const int H = w.H;
    const float la = w.loga[h * SLOTS + m];
    const float tau = __expf(rd.log_tau[h]);
    __shared__ float s_wm[NWARP], s_ws[NWARP], s_wo[NWARP][DH];
    float* base = state + (size_t(map[j]) * H + h) * size_t(ckpt_fields_t<DH>()) * SLOTS;
    float Kb[DH], Vb[DH], Zs, tmf;
    ckpt_load(base, m, Kb, Vb, Zs, tmf);
    int tm = __float_as_int(tmf);
    const int t = tpos[j];
    const size_t rh = size_t(j) * H + h;
    const Info si = unpack(w.info[rh]);
    if (write) {
        // a reset empties the slot (Zs = 0) and restarts its clock: a reused slot's last write time can lie after t,
        // and the decay a^(t - tm) of that negative interval overflows (0 * inf = NaN)
        if (si.flags & 4) { tm = t; }
        const bool changed = (si.flags & 4) || ((si.flags & 1) && (m == si.m1 || m == si.m2));
        step_state(Kb, Vb, Zs, tm, la, t, m, si, w, size_t(j), h);
        if (changed) { ckpt_store(base, m, Kb, Vb, Zs, __int_as_float(tm)); }
    }
    bf16* out = rd.o + size_t(j) * H * DH + h * DH;
    if (si.flags & rd.gate_bit) {
        const bool valid = Zs * __expf(la * float(t - tm)) > BANK_EPS;
        float sc = -INFINITY;
        if (valid) { sc = tau * dot_load<DH>(rd.q + rh * DH, Kb); }
        const float wmax = warp_max(sc);
        const float e = valid ? __expf(sc - wmax) : 0.f;
        const float wsum = warp_sum(e);
        float od[DH / 32];
        warp_scatter<DH>(Vb, e, lane, od);
        if (lane == 0) {
            s_wm[warp] = wmax;
            s_ws[warp] = wsum;
        }
#pragma unroll
        for (int c = 0; c < DH / 32; ++c) { s_wo[warp][lane + 32 * c] = od[c]; }
        __syncthreads();
        if (warp == 0) {
            float M = -INFINITY;
#pragma unroll
            for (int i = 0; i < NWARP; ++i) { M = fmaxf(M, s_wm[i]); }
            float o[DH / 32];
#pragma unroll
            for (int c = 0; c < DH / 32; ++c) { o[c] = 0.f; }
            if (M > -INFINITY) {
                float S = 0.f;
#pragma unroll
                for (int i = 0; i < NWARP; ++i) {
                    const float f = s_wm[i] > -INFINITY ? __expf(s_wm[i] - M) : 0.f;
                    S += s_ws[i] * f;
#pragma unroll
                    for (int c = 0; c < DH / 32; ++c) { o[c] += s_wo[i][lane + 32 * c] * f; }
                }
#pragma unroll
                for (int c = 0; c < DH / 32; ++c) { o[c] /= S; }
            }
#pragma unroll
            for (int c = 0; c < DH / 32; ++c) { out[lane + 32 * c] = __float2bfloat16(o[c]); }
        }
    } else if (warp == 0) {
#pragma unroll
        for (int c = 0; c < DH / 32; ++c) { out[lane + 32 * c] = __float2bfloat16(0.f); }
    }
}

template <int DH>
__global__ void __launch_bounds__(SLOTS) k_bank_rewalk(BankWrites w, BankRead rd) {
    const int b = blockIdx.x, h = blockIdx.y, m = threadIdx.x, lane = m & 31, warp = m >> 5;
    const int T = w.T, H = w.H, t0 = blockIdx.z * w.chunk, t1 = min(T, t0 + w.chunk);
    const size_t plane = size_t(w.B) * T * H * SLOTS;
    const float la = w.loga[h * SLOTS + m];
    const float tau = __expf(rd.log_tau[h]);
    __shared__ float s_red[2][NWARP][3];
    __shared__ float s_dq[2][NWARP][DH];
    __shared__ float s_tau[NWARP];
    float Kb[DH], Vb[DH], Zs, dtau = 0.f;
    int tm, nopen = 0;
    chunk_start(w, b, h, blockIdx.z, m, Kb, Vb, Zs, tm);
    // the dq of an open read is summed over the warps after the next open read's barrier (or after the loop):
    // one barrier per open read; s_dq is double-buffered by open-read parity
    size_t pending_rh = 0;
    int pending_buf = -1;
    auto flush_dq = [&]() {
        if (warp == 0 && pending_buf >= 0) {
#pragma unroll
            for (int c = 0; c < DH / 32; ++c) {
                float dq = 0.f;
#pragma unroll
                for (int i = 0; i < NWARP; ++i) dq += s_dq[pending_buf][i][lane + 32 * c];
                rd.dq[pending_rh * DH + lane + 32 * c] = dq;
            }
        }
        pending_buf = -1;
    };
    const size_t base = size_t(b) * T;
    uint2 nxt = w.info[(base + t0) * H + h];
    for (int t = t0; t < t1; ++t) {
        const Info si = unpack(nxt);
        if (t + 1 < t1) { nxt = w.info[(base + t + 1) * H + h]; }
        const size_t r = base + t, rh = r * H + h;
        step_state(Kb, Vb, Zs, tm, la, t, m, si, w, r, h);
        if (si.flags & rd.gate_bit) {
            const float Zt = Zs * __expf(la * float(t - tm));
            const bool valid = Zt > BANK_EPS;
            float sc = -INFINITY, dp = 0.f;
            if (valid) {
                sc = tau * dot_load<DH>(rd.q + rh * DH, Kb);
                dp = dot_load<DH>(rd.dy + r * H * DH + h * DH, Vb);  // dy . (the read's value on slot m) = dy . Vb_m
            }
            const float wmax = warp_max(sc);
            const float e = valid ? __expf(sc - wmax) : 0.f;
            const float wsum = warp_sum(e), wpd = warp_sum(e * dp);
            const int buf = nopen++ & 1;
            if (lane == 0) {
                s_red[buf][warp][0] = wmax;
                s_red[buf][warp][1] = wsum;
                s_red[buf][warp][2] = wpd;
            }
            __syncthreads();
            flush_dq();
            float M = -INFINITY;
#pragma unroll
            for (int i = 0; i < NWARP; ++i) { M = fmaxf(M, s_red[buf][i][0]); }
            float S = 0.f, PD = 0.f;
            if (M > -INFINITY) {
#pragma unroll
                for (int i = 0; i < NWARP; ++i) {
                    const float f = s_red[buf][i][0] > -INFINITY ? __expf(s_red[buf][i][0] - M) : 0.f;
                    S += s_red[buf][i][1] * f;
                    PD += s_red[buf][i][2] * f;
                }
            }
            const float sdp = S > 0.f ? PD / S : 0.f;  // sum_m p_m dp_m = dy . read
            float alpha = 0.f, beta = 0.f, gamma = 0.f, tds = 0.f;
            if (valid) {
                const float p = e * __expf(wmax - M) / S;
                const float ds = p * (dp - sdp);
                tds = tau * ds;
                alpha = tds / Zt;
                beta = p / Zt;
                gamma = -(ds * sc + p * dp) / Zt;
                dtau += ds * sc / tau;
            }
            float dqw[DH / 32];
            warp_scatter<DH>(Kb, tds, lane, dqw);  // dq = tau sum_m ds_m Kb_m
#pragma unroll
            for (int c = 0; c < DH / 32; ++c) { s_dq[buf][warp][lane + 32 * c] = dqw[c]; }
            pending_rh = rh;
            pending_buf = buf;
            rd.coef[rh * SLOTS + m] = __float2bfloat16(alpha);
            rd.coef[plane + rh * SLOTS + m] = __float2bfloat16(beta);
            rd.coef[2 * plane + rh * SLOTS + m] = __float2bfloat16(gamma);
            if (m == 0) { rd.dg[rh] = sdp; }
        } else {
            if (warp == 0)
#pragma unroll
                for (int c = 0; c < DH / 32; ++c) { rd.dq[rh * DH + lane + 32 * c] = 0.f; }
            if (m == 0) { rd.dg[rh] = 0.f; }
        }
    }
    dtau = warp_sum(dtau);
    if (lane == 0) { s_tau[warp] = dtau; }
    __syncthreads();
    flush_dq();
    if (m == 0) {
        float tot = 0.f;
#pragma unroll
        for (int i = 0; i < NWARP; ++i) { tot += s_tau[i]; }
        atomicAdd(rd.dlog_tau + h, tot * tau);
    }
}

// LOCAL: the chunk's own contribution, from a zero adjoint at its end; stores the adjoint at the chunk start (after
// that token's local terms) in a.carry and whether a reset lies inside the chunk in a.flags. Otherwise: the full
// walk of the chunk from the incoming adjoint (a.carry, written by the combine), writing the write gradients.
template <int DH, bool LOCAL>
__global__ void __launch_bounds__(SLOTS) k_bank_adjoint(BankWrites w, BankAdjoint a) {
    const int b = blockIdx.x, h = blockIdx.y, c = blockIdx.z, m = threadIdx.x;
    const int T = w.T, H = w.H, nc = bank_chunks(w), t0 = c * w.chunk, t1 = min(T, t0 + w.chunk);
    const size_t plane = size_t(w.B) * T * H * SLOTS;
    const float la = w.loga[h * SLOTS + m];
    float LK[DH], LV[DH], LZ = 0.f;
#pragma unroll
    for (int d = 0; d < DH; ++d) { LK[d] = LV[d] = 0.f; }
    int tl = t1 - 1;  // the time the adjoint state (LK, LV, LZ) refers to
    const size_t base = size_t(b) * T;
    if (!LOCAL && t1 < T && !(w.info[(base + t1) * H + h].x & (4u << 16))) {
        float unused;  // the adjoint of the state at t1 (the next chunk's start); t1 - 1 gets a times it
        ckpt_load(a.carry + ckpt_offset<DH>(b, h, c, H, nc), m, LK, LV, LZ, unused);
        tl = t1;
    }
    bool reset_inside = false;
    uint2 nxt = w.info[(base + t1 - 1) * H + h];
    for (int t = t1 - 1; t >= t0; --t) {
        const Info si = unpack(nxt);
        if (t > t0) { nxt = w.info[(base + t - 1) * H + h]; }
        const size_t r = base + t, rh = r * H + h;
        const bool open[2] = {(si.flags & 2) != 0, a.hops > 1 && (si.flags & 8) != 0};
        const bool routed = !LOCAL && (m == si.m1 || m == si.m2);
        if (open[0] || open[1] || routed) {
            if (tl != t) {  // lambda_t = a^(tl - t) lambda_tl when no local term or reset lies between
                const float f = __expf(la * float(tl - t));
#pragma unroll
                for (int d = 0; d < DH; ++d) {
                    LK[d] *= f;
                    LV[d] *= f;
                }
                LZ *= f;
                tl = t;
            }
#pragma unroll
            for (int hop = 0; hop < 2; ++hop) {
                if (!open[hop]) { continue; }
                const bf16* cf = a.coef[hop];
                const float alpha = __bfloat162float(cf[rh * SLOTS + m]),
                            beta = __bfloat162float(cf[plane + rh * SLOTS + m]),
                            gamma = __bfloat162float(cf[2 * plane + rh * SLOTS + m]);
                axpy_load<DH>(a.q[hop] + rh * DH, alpha, LK);
                axpy_load<DH>(a.dy[hop] + r * H * DH + h * DH, beta, LV);
                LZ += gamma;
            }
            if (!LOCAL && routed) {  // K_m += r_m k, V_m += r_m v, Z_m += r_m
                const int slot = m == si.m1 ? 0 : 1;
                const float dr = LZ + dot_load<DH>(w.k + rh * DH, LK) + dot_load<DH>(w.v + r * w.ldv + h * DH, LV);
                a.dr2[rh * 2 + slot] = dr;
                if (si.flags & 1) {
                    const float rw = slot == 0 ? si.p1 : 1.f - si.p1;
#pragma unroll
                    for (int c = 0; c < DH / 32; ++c) {
                        store32(a.dk2 + (rh * 2 + slot) * DH + 32 * c, LK + 32 * c, rw);
                        store32(a.dv2 + (rh * 2 + slot) * DH + 32 * c, LV + 32 * c, rw);
                    }
                }
            }
        }
        if ((si.flags & 4) && t > t0) {  // state_t does not depend on state_{t-1} (the chunk start is kept)
#pragma unroll
            for (int d = 0; d < DH; ++d) { LK[d] = LV[d] = 0.f; }
            LZ = 0.f;
            reset_inside = true;
        }
    }
    if (LOCAL) {
        if (tl != t0) {
            const float f = __expf(la * float(tl - t0));
#pragma unroll
            for (int d = 0; d < DH; ++d) {
                LK[d] *= f;
                LV[d] *= f;
            }
            LZ *= f;
        }
        ckpt_store(a.carry + ckpt_offset<DH>(b, h, c, H, nc), m, LK, LV, LZ, 0.f);
        if (m == 0) { a.flags[(size_t(b) * H + h) * nc + c] = reset_inside; }
    }
}

// Per slot, from the last chunk back: E_c (the adjoint at chunk c's start) = L_c + a^(t1 - t0) E_{c+1} unless a reset
// lies in (t0, t1]; the chunk's final pass needs E_{c+1}, written over L_c.
template <int DH>
__global__ void __launch_bounds__(SLOTS) k_bank_adjoint_combine(BankWrites w, BankAdjoint a) {
    const int b = blockIdx.x, h = blockIdx.y, m = threadIdx.x;
    const int T = w.T, H = w.H, nc = bank_chunks(w);
    const float la = w.loga[h * SLOTS + m];
    float EK[DH], EV[DH], EZ = 0.f;
#pragma unroll
    for (int d = 0; d < DH; ++d) { EK[d] = EV[d] = 0.f; }
    for (int c = nc - 1; c >= 0; --c) {
        const int t0 = c * w.chunk, t1 = min(T, t0 + w.chunk);
        float* rec = a.carry + ckpt_offset<DH>(b, h, c, H, nc);
        float LK[DH], LV[DH], LZ, unused;
        ckpt_load(rec, m, LK, LV, LZ, unused);
        ckpt_store(rec, m, EK, EV, EZ, 0.f);
        const bool through = !a.flags[(size_t(b) * H + h) * nc + c]
                             && !(t1 < T && (w.info[(size_t(b) * T + t1) * H + h].x & (4u << 16)));
        const float f = through ? __expf(la * float(t1 - t0)) : 0.f;
#pragma unroll
        for (int d = 0; d < DH; ++d) {
            EK[d] = LK[d] + f * EK[d];
            EV[d] = LV[d] + f * EV[d];
        }
        EZ = LZ + f * EZ;
    }
}

}  // namespace

namespace {

template <int DH>
void checkpoint_t(const BankWrites& w, float* ckpt, float* local, int* rflags, cudaStream_t s) {
    k_bank_ckpt_local<DH><<<dim3(w.B, w.H, bank_chunks(w)), SLOTS, 0, s>>>(w, local, rflags);
    LBC_CUDA(cudaGetLastError());
    k_bank_ckpt_merge<DH><<<dim3(w.B, w.H), SLOTS, 0, s>>>(w, local, rflags, ckpt);
    LBC_CUDA(cudaGetLastError());
}

template <int DH>
void adjoint_t(const BankWrites& w, const BankAdjoint& a, cudaStream_t s) {
    const dim3 grid(w.B, w.H, bank_chunks(w));
    if (bank_chunks(w) > 1) {
        k_bank_adjoint<DH, true><<<grid, SLOTS, 0, s>>>(w, a);
        LBC_CUDA(cudaGetLastError());
        k_bank_adjoint_combine<DH><<<dim3(w.B, w.H), SLOTS, 0, s>>>(w, a);
        LBC_CUDA(cudaGetLastError());
    }
    k_bank_adjoint<DH, false><<<grid, SLOTS, 0, s>>>(w, a);
    LBC_CUDA(cudaGetLastError());
}

void check_dh(int dh) {
    if (dh != 32 && dh != 64) { fail("bank", "head size must be 32 or 64", __FILE__, __LINE__); }
}

}  // namespace

void bank_checkpoint(const BankWrites& w, float* ckpt, float* local, int* rflags, cudaStream_t s) {
    check_dh(w.dh);
    w.dh == 64 ? checkpoint_t<64>(w, ckpt, local, rflags, s) : checkpoint_t<32>(w, ckpt, local, rflags, s);
}

void bank_step(const BankWrites& w,
               const BankRead& r,
               const int* map,
               const int* tpos,
               float* state,
               bool write,
               cudaStream_t s) {
    check_dh(w.dh);
    if (w.dh == 64) {
        k_bank_step<64><<<dim3(w.B, w.H), SLOTS, 0, s>>>(w, r, map, tpos, state, write ? 1 : 0);
    } else {
        k_bank_step<32><<<dim3(w.B, w.H), SLOTS, 0, s>>>(w, r, map, tpos, state, write ? 1 : 0);
    }
    LBC_CUDA(cudaGetLastError());
}

void bank_fwd(const BankWrites& w, const BankRead& r, cudaStream_t s) {
    check_dh(w.dh);
    if (w.dh == 64) {
        k_bank_fwd<64><<<dim3(w.B, w.H, bank_chunks(w)), SLOTS, 0, s>>>(w, r);
    } else {
        k_bank_fwd<32><<<dim3(w.B, w.H, bank_chunks(w)), SLOTS, 0, s>>>(w, r);
    }
    LBC_CUDA(cudaGetLastError());
}

void bank_rewalk(const BankWrites& w, const BankRead& r, cudaStream_t s) {
    check_dh(w.dh);
    if (w.dh == 64) {
        k_bank_rewalk<64><<<dim3(w.B, w.H, bank_chunks(w)), SLOTS, 0, s>>>(w, r);
    } else {
        k_bank_rewalk<32><<<dim3(w.B, w.H, bank_chunks(w)), SLOTS, 0, s>>>(w, r);
    }
    LBC_CUDA(cudaGetLastError());
}

void bank_adjoint(const BankWrites& w, const BankAdjoint& a, cudaStream_t s) {
    check_dh(w.dh);
    w.dh == 64 ? adjoint_t<64>(w, a, s) : adjoint_t<32>(w, a, s);
}

}  // namespace lbc
