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
    for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(FULL, v, o);
    return v;
}

__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
    for (int o = 16; o; o >>= 1) v = fmaxf(v, __shfl_xor_sync(FULL, v, o));
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
__device__ __forceinline__ void load32(const bf16* p, float (&x)[DH]) {
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
__device__ __forceinline__ void load32(const float* p, float (&x)[DH]) {
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

__device__ __forceinline__ void store32(float* p, const float (&x)[DH], float scale) {
    float4* q = reinterpret_cast<float4*>(p);
#pragma unroll
    for (int i = 0; i < 8; ++i)
        q[i] = make_float4(scale * x[4 * i], scale * x[4 * i + 1], scale * x[4 * i + 2], scale * x[4 * i + 3]);
}

struct Info {
    int m1, m2, flags;
    float p1;
};

__device__ __forceinline__ Info unpack(uint2 w) {
    return Info{int(w.x & 0xffu), int((w.x >> 8) & 0xffu), int((w.x >> 16) & 0xffu), __uint_as_float(w.y)};
}

// the write of token t into this thread's slot with weight r: running averages of the decayed writes
__device__ __forceinline__ void write_slot(float (&Kb)[DH], float (&Vb)[DH], float& Zs, int& tm, float la, int t,
                                           float r, const bf16* k, const bf16* v) {
    float kk[DH], vv[DH];
    load32(k, kk);
    load32(v, vv);
    const float zn = Zs * __expf(la * float(t - tm)) + r;
    const float c = r / zn;
#pragma unroll
    for (int d = 0; d < DH; ++d) {
        Kb[d] += c * (kk[d] - Kb[d]);
        Vb[d] += c * (vv[d] - Vb[d]);
    }
    Zs = zn;
    tm = t;
}

// the state update of token t (reset, then the write when this thread's slot is routed and the write gate open)
__device__ __forceinline__ void step_state(float (&Kb)[DH], float (&Vb)[DH], float& Zs, int& tm, float la, int t,
                                           int m, const Info& si, const BankWrites& w, size_t r, int h) {
    if (si.flags & 4) Zs = 0.f;
    if ((si.flags & 1) && (m == si.m1 || m == si.m2))
        write_slot(Kb, Vb, Zs, tm, la, t, m == si.m1 ? si.p1 : 1.f - si.p1, w.k + (r * w.H + h) * DH,
                   w.v + r * w.ldv + h * DH);
}

__global__ void __launch_bounds__(SLOTS) k_bank_fwd(BankWrites w, BankRead rd) {
    const int b = blockIdx.x, h = blockIdx.y, m = threadIdx.x, lane = m & 31, warp = m >> 5;
    const int T = w.T, H = w.H;
    const float la = w.loga[h * SLOTS + m];
    const float tau = __expf(rd.log_tau[h]);
    __shared__ float s_wm[2][NWARP], s_ws[2][NWARP], s_wo[2][NWARP][DH];
    float Kb[DH], Vb[DH], Zs = 0.f;
    int tm = 0, nopen = 0;
#pragma unroll
    for (int d = 0; d < DH; ++d) Kb[d] = Vb[d] = 0.f;
    const size_t base = size_t(b) * T;
    uint2 nxt = w.info[base * H + h];
    for (int t = 0; t < T; ++t) {
        const Info si = unpack(nxt);
        if (t + 1 < T) nxt = w.info[(base + t + 1) * H + h];
        const size_t r = base + t, rh = r * H + h;
        step_state(Kb, Vb, Zs, tm, la, t, m, si, w, r, h);
        bf16* out = rd.o + r * H * DH + h * DH;
        if (si.flags & rd.gate_bit) {
            const bool valid = Zs * __expf(la * float(t - tm)) > BANK_EPS;
            float sc = -INFINITY;
            if (valid) {
                float q[DH];
                load32(rd.q + rh * DH, q);
                float dot = 0.f;
#pragma unroll
                for (int d = 0; d < DH; ++d) dot += q[d] * Kb[d];
                sc = tau * dot;
            }
            const float wmax = warp_max(sc);
            const float e = valid ? __expf(sc - wmax) : 0.f;
            const float wsum = warp_sum(e);
            float u[DH];
#pragma unroll
            for (int d = 0; d < DH; ++d) u[d] = e * Vb[d];
            const float od = reduce_scatter32(u, lane);
            const int buf = nopen++ & 1;
            if (lane == 0) {
                s_wm[buf][warp] = wmax;
                s_ws[buf][warp] = wsum;
            }
            s_wo[buf][warp][lane] = od;
            __syncthreads();
            if (warp == 0) {
                float M = -INFINITY;
#pragma unroll
                for (int i = 0; i < NWARP; ++i) M = fmaxf(M, s_wm[buf][i]);
                float o = 0.f;
                if (M > -INFINITY) {
                    float S = 0.f, acc = 0.f;
#pragma unroll
                    for (int i = 0; i < NWARP; ++i) {
                        const float f = s_wm[buf][i] > -INFINITY ? __expf(s_wm[buf][i] - M) : 0.f;
                        S += s_ws[buf][i] * f;
                        acc += s_wo[buf][i][lane] * f;
                    }
                    o = acc / S;
                }
                out[lane] = __float2bfloat16(o);
            }
        } else if (warp == 0) {
            out[lane] = __float2bfloat16(0.f);
        }
    }
}

__global__ void __launch_bounds__(SLOTS) k_bank_rewalk(BankWrites w, BankRead rd) {
    const int b = blockIdx.x, h = blockIdx.y, m = threadIdx.x, lane = m & 31, warp = m >> 5;
    const int T = w.T, H = w.H;
    const size_t plane = size_t(w.B) * T * H * SLOTS;
    const float la = w.loga[h * SLOTS + m];
    const float tau = __expf(rd.log_tau[h]);
    __shared__ float s_red[2][NWARP][3];
    __shared__ float s_dq[2][NWARP][DH];
    __shared__ float s_tau[NWARP];
    float Kb[DH], Vb[DH], Zs = 0.f, dtau = 0.f;
    int tm = 0, nopen = 0;
    // the dq of an open read is summed over the warps after the next open read's barrier (or after the loop):
    // one barrier per open read; s_dq is double-buffered by open-read parity
    size_t pending_rh = 0;
    int pending_buf = -1;
    auto flush_dq = [&]() {
        if (warp == 0 && pending_buf >= 0) {
            float dq = 0.f;
#pragma unroll
            for (int i = 0; i < NWARP; ++i) dq += s_dq[pending_buf][i][lane];
            rd.dq[pending_rh * DH + lane] = dq;
        }
        pending_buf = -1;
    };
#pragma unroll
    for (int d = 0; d < DH; ++d) Kb[d] = Vb[d] = 0.f;
    const size_t base = size_t(b) * T;
    uint2 nxt = w.info[base * H + h];
    for (int t = 0; t < T; ++t) {
        const Info si = unpack(nxt);
        if (t + 1 < T) nxt = w.info[(base + t + 1) * H + h];
        const size_t r = base + t, rh = r * H + h;
        step_state(Kb, Vb, Zs, tm, la, t, m, si, w, r, h);
        if (si.flags & rd.gate_bit) {
            const float Zt = Zs * __expf(la * float(t - tm));
            const bool valid = Zt > BANK_EPS;
            float sc = -INFINITY, dp = 0.f;
            if (valid) {
                float x[DH];
                load32(rd.q + rh * DH, x);
                float dot = 0.f;
#pragma unroll
                for (int d = 0; d < DH; ++d) dot += x[d] * Kb[d];
                sc = tau * dot;
                load32(rd.dy + r * H * DH + h * DH, x);
                float dyv = 0.f;
#pragma unroll
                for (int d = 0; d < DH; ++d) dyv += x[d] * Vb[d];
                dp = dyv;  // dy . (the read's value on slot m) = dy . Vb_m
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
            for (int i = 0; i < NWARP; ++i) M = fmaxf(M, s_red[buf][i][0]);
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
            float u[DH];
#pragma unroll
            for (int d = 0; d < DH; ++d) u[d] = tds * Kb[d];
            s_dq[buf][warp][lane] = reduce_scatter32(u, lane);  // dq = tau sum_m ds_m Kb_m
            pending_rh = rh;
            pending_buf = buf;
            rd.coef[rh * SLOTS + m] = __float2bfloat16(alpha);
            rd.coef[plane + rh * SLOTS + m] = __float2bfloat16(beta);
            rd.coef[2 * plane + rh * SLOTS + m] = __float2bfloat16(gamma);
            if (m == 0) rd.dg[rh] = sdp;
        } else {
            if (warp == 0) rd.dq[rh * DH + lane] = 0.f;
            if (m == 0) rd.dg[rh] = 0.f;
        }
    }
    dtau = warp_sum(dtau);
    if (lane == 0) s_tau[warp] = dtau;
    __syncthreads();
    flush_dq();
    if (m == 0) {
        float tot = 0.f;
#pragma unroll
        for (int i = 0; i < NWARP; ++i) tot += s_tau[i];
        atomicAdd(rd.dlog_tau + h, tot * tau);
    }
}

__global__ void __launch_bounds__(SLOTS) k_bank_adjoint(BankWrites w, BankAdjoint a) {
    const int b = blockIdx.x, h = blockIdx.y, m = threadIdx.x;
    const int T = w.T, H = w.H;
    const size_t plane = size_t(w.B) * T * H * SLOTS;
    const float la = w.loga[h * SLOTS + m];
    float LK[DH], LV[DH], LZ = 0.f;
#pragma unroll
    for (int d = 0; d < DH; ++d) LK[d] = LV[d] = 0.f;
    int tl = T - 1;  // the time the adjoint state (LK, LV, LZ) refers to
    const size_t base = size_t(b) * T;
    uint2 nxt = w.info[(base + T - 1) * H + h];
    for (int t = T - 1; t >= 0; --t) {
        const Info si = unpack(nxt);
        if (t > 0) nxt = w.info[(base + t - 1) * H + h];
        const size_t r = base + t, rh = r * H + h;
        const bool open[2] = {(si.flags & 2) != 0, a.hops > 1 && (si.flags & 8) != 0};
        const bool routed = m == si.m1 || m == si.m2;
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
                if (!open[hop]) continue;
                const bf16* cf = a.coef[hop];
                const float alpha = __bfloat162float(cf[rh * SLOTS + m]),
                            beta = __bfloat162float(cf[plane + rh * SLOTS + m]),
                            gamma = __bfloat162float(cf[2 * plane + rh * SLOTS + m]);
                float x[DH];
                load32(a.q[hop] + rh * DH, x);
#pragma unroll
                for (int d = 0; d < DH; ++d) LK[d] += alpha * x[d];
                load32(a.dy[hop] + r * H * DH + h * DH, x);
#pragma unroll
                for (int d = 0; d < DH; ++d) LV[d] += beta * x[d];
                LZ += gamma;
            }
            if (routed) {  // K_m += r_m k, V_m += r_m v, Z_m += r_m
                const int slot = m == si.m1 ? 0 : 1;
                float x[DH];
                load32(w.k + rh * DH, x);
                float dr = LZ;
#pragma unroll
                for (int d = 0; d < DH; ++d) dr += x[d] * LK[d];
                load32(w.v + r * w.ldv + h * DH, x);
#pragma unroll
                for (int d = 0; d < DH; ++d) dr += x[d] * LV[d];
                a.dr2[rh * 2 + slot] = dr;
                if (si.flags & 1) {
                    const float rw = slot == 0 ? si.p1 : 1.f - si.p1;
                    store32(a.dk2 + (rh * 2 + slot) * DH, LK, rw);
                    store32(a.dv2 + (rh * 2 + slot) * DH, LV, rw);
                }
            }
        }
        if (si.flags & 4) {  // state_t does not depend on state_{t-1}
#pragma unroll
            for (int d = 0; d < DH; ++d) LK[d] = LV[d] = 0.f;
            LZ = 0.f;
        }
    }
}

}  // namespace

void bank_fwd(const BankWrites& w, const BankRead& r, cudaStream_t s) {
    k_bank_fwd<<<dim3(w.B, w.H), SLOTS, 0, s>>>(w, r);
    LBC_CUDA(cudaGetLastError());
}

void bank_rewalk(const BankWrites& w, const BankRead& r, cudaStream_t s) {
    k_bank_rewalk<<<dim3(w.B, w.H), SLOTS, 0, s>>>(w, r);
    LBC_CUDA(cudaGetLastError());
}

void bank_adjoint(const BankWrites& w, const BankAdjoint& a, cudaStream_t s) {
    k_bank_adjoint<<<dim3(w.B, w.H), SLOTS, 0, s>>>(w, a);
    LBC_CUDA(cudaGetLastError());
}

}  // namespace lbc
