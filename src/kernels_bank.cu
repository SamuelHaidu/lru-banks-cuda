// The banks, as an exact sequential walk per (row, head): one block of SLOTS threads, thread m owns slot m's state
// (K_m, V_m: DH floats each, Z_m) in registers. Per token: decay (or clear at a reset), the write into the two
// routed slots (if the write gate is open), then the read (if the read gate is open: closed reads skip the scores,
// the softmax and the output entirely).
//
// Backward, per layer: one re-walk per read (forward recompute; per slot the read's local gradient on the state is
// rank one: dK_m = alpha q, dV_m = beta dy, dZ_m = gamma, stored in a scratch), then one reverse adjoint walk for both
// reads together (lambda_t = local_t + a * lambda_{t+1}) giving the gradients of the writes: k, v, the routing
// weight and the write gate (straight-through: d g_w is computed for closed gates too).
//
// Inputs of step t+1 are loaded into a shared double buffer while step t computes; the warp results go through a
// double-buffered exchange, so each step needs one block barrier (forward) or two (re-walk).

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

// One halving step: lanes whose bit O is set keep the upper half of the current block, the others the lower half,
// and each adds the half its partner (lane ^ O) sends.
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

// Sum over the warp of the lanes' vectors u[0..31]; lane d returns component d (31 shuffles). The steps are spelled
// out (a loop over the offset was not unrolled and put u in local memory).
__device__ __forceinline__ float reduce_scatter32(float (&u)[32], int lane) {
    reduce_scatter_step<16>(u, lane);
    reduce_scatter_step<8>(u, lane);
    reduce_scatter_step<4>(u, lane);
    reduce_scatter_step<2>(u, lane);
    reduce_scatter_step<1>(u, lane);
    return u[0];
}

struct StepInfo {
    int m1, m2, flags;  // flags: 1 write gate, 2 read gate (hop 0), 4 reset, 8 read gate (hop 1)
    float p1;
};

__device__ __forceinline__ void load_info(StepInfo& si, const BankWrites& w, const uint8_t* gr0, const uint8_t* gr1,
                                          size_t r, int h) {
    const size_t rh = r * w.H + h;
    si.m1 = w.ridx[rh * 2];
    si.m2 = w.ridx[rh * 2 + 1];
    si.flags = (w.gw[rh] ? 1 : 0) | ((gr0 && gr0[rh]) ? 2 : 0) | (w.reset[r] ? 4 : 0) | ((gr1 && gr1[rh]) ? 8 : 0);
    si.p1 = w.rp1[rh];
}

// state update for one token: decay or clear, then the write
__device__ __forceinline__ void update_state(float (&K)[DH], float (&V)[DH], float& Z, float am, bool rst, bool gw,
                                             int m, int m1, int m2, float p1, const float* sk, const float* sv) {
    if (rst) {
#pragma unroll
        for (int d = 0; d < DH; ++d) K[d] = V[d] = 0.f;
        Z = 0.f;
    } else {
#pragma unroll
        for (int d = 0; d < DH; ++d) {
            K[d] *= am;
            V[d] *= am;
        }
        Z *= am;
    }
    if (gw && (m == m1 || m == m2)) {
        const float c = (m == m1) ? p1 : 1.f - p1;
#pragma unroll
        for (int d = 0; d < DH; ++d) {
            K[d] += c * sk[d];
            V[d] += c * sv[d];
        }
        Z += c;
    }
}

__global__ void __launch_bounds__(SLOTS) k_bank_fwd(BankWrites w, BankRead rd) {
    const int b = blockIdx.x, h = blockIdx.y, m = threadIdx.x, lane = m & 31, warp = m >> 5;
    const int T = w.T, H = w.H;
    const float am = __expf(w.loga[h * SLOTS + m]);
    const float tau = __expf(rd.log_tau[h]);
    __shared__ float s_k[2][DH], s_v[2][DH], s_q[2][DH];
    __shared__ StepInfo s_i[2];
    __shared__ float s_wm[2][NWARP], s_ws[2][NWARP], s_wo[2][NWARP][DH];

    auto load = [&](int t, int buf) {
        const size_t r = size_t(b) * T + t;
        const size_t rh = r * H + h;
        if (m < 32) s_k[buf][m] = __bfloat162float(w.k[rh * DH + m]);
        else if (m < 64) s_v[buf][m - 32] = __bfloat162float(w.v[r * w.ldv + h * DH + (m - 32)]);
        else if (m < 96) s_q[buf][m - 64] = __bfloat162float(rd.q[rh * DH + (m - 64)]);
        else if (m == 96) load_info(s_i[buf], w, rd.gr, nullptr, r, h);
    };

    float K[DH], V[DH], Z = 0.f;
#pragma unroll
    for (int d = 0; d < DH; ++d) K[d] = V[d] = 0.f;
    load(0, 0);
    __syncthreads();
    for (int t = 0; t < T; ++t) {
        const int cur = t & 1;
        const StepInfo si = s_i[cur];
        const bool gw = si.flags & 1, gr = si.flags & 2, rst = si.flags & 4;
        update_state(K, V, Z, am, rst, gw, m, si.m1, si.m2, si.p1, s_k[cur], s_v[cur]);
        if (gr) {
            const bool valid = Z > BANK_EPS;
            float sc = -INFINITY;
            if (valid) {
                float dot = 0.f;
#pragma unroll
                for (int d = 0; d < DH; ++d) dot += s_q[cur][d] * K[d];
                sc = tau * dot / Z;
            }
            const float wmax = warp_max(sc);
            const float e = valid ? __expf(sc - wmax) : 0.f;
            const float wsum = warp_sum(e);
            const float c = valid ? e / Z : 0.f;
            float u[DH];
#pragma unroll
            for (int d = 0; d < DH; ++d) u[d] = c * V[d];
            const float od = reduce_scatter32(u, lane);
            if (lane == 0) {
                s_wm[cur][warp] = wmax;
                s_ws[cur][warp] = wsum;
            }
            s_wo[cur][warp][lane] = od;
        }
        if (t + 1 < T) load(t + 1, cur ^ 1);
        __syncthreads();
        if (warp == 0) {
            float o = 0.f;
            if (gr) {
                float M = -INFINITY;
#pragma unroll
                for (int i = 0; i < NWARP; ++i) M = fmaxf(M, s_wm[cur][i]);
                if (M > -INFINITY) {
                    float S = 0.f, acc = 0.f;
#pragma unroll
                    for (int i = 0; i < NWARP; ++i) {
                        const float f = s_wm[cur][i] > -INFINITY ? __expf(s_wm[cur][i] - M) : 0.f;
                        S += s_ws[cur][i] * f;
                        acc += s_wo[cur][i][lane] * f;
                    }
                    o = acc / S;
                }
            }
            rd.o[(size_t(b) * T + t) * H * DH + h * DH + lane] = __float2bfloat16(o);
        }
    }
}

__global__ void __launch_bounds__(SLOTS) k_bank_rewalk(BankWrites w, BankRead rd) {
    const int b = blockIdx.x, h = blockIdx.y, m = threadIdx.x, lane = m & 31, warp = m >> 5;
    const int T = w.T, H = w.H;
    const size_t plane = size_t(w.B) * T * H * SLOTS;
    const float am = __expf(w.loga[h * SLOTS + m]);
    const float tau = __expf(rd.log_tau[h]);
    __shared__ float s_k[2][DH], s_v[2][DH], s_q[2][DH], s_dy[2][DH];
    __shared__ StepInfo s_i[2];
    __shared__ float s_red[2][NWARP][3];
    __shared__ float s_dq[2][NWARP][DH];
    __shared__ float s_tau[NWARP];

    auto load = [&](int t, int buf) {
        const size_t r = size_t(b) * T + t;
        const size_t rh = r * H + h;
        if (m < 32) {
            s_k[buf][m] = __bfloat162float(w.k[rh * DH + m]);
            s_dy[buf][m] = rd.dy[r * H * DH + h * DH + m];
        } else if (m < 64) {
            s_v[buf][m - 32] = __bfloat162float(w.v[r * w.ldv + h * DH + (m - 32)]);
        } else if (m < 96) {
            s_q[buf][m - 64] = __bfloat162float(rd.q[rh * DH + (m - 64)]);
        } else if (m == 96) {
            load_info(s_i[buf], w, rd.gr, nullptr, r, h);
        }
    };

    float K[DH], V[DH], Z = 0.f, dtau = 0.f;
#pragma unroll
    for (int d = 0; d < DH; ++d) K[d] = V[d] = 0.f;
    load(0, 0);
    __syncthreads();
    for (int t = 0; t < T; ++t) {
        const int cur = t & 1;
        const size_t r = size_t(b) * T + t;
        const size_t rh = r * H + h;
        const StepInfo si = s_i[cur];
        const bool gw = si.flags & 1, gr = si.flags & 2, rst = si.flags & 4;
        update_state(K, V, Z, am, rst, gw, m, si.m1, si.m2, si.p1, s_k[cur], s_v[cur]);
        const bool valid = gr && Z > BANK_EPS;
        float sc = -INFINITY, e = 0.f, dp = 0.f, wmax = -INFINITY;
        if (gr) {
            if (valid) {
                float dot = 0.f, dyv = 0.f;
#pragma unroll
                for (int d = 0; d < DH; ++d) {
                    dot += s_q[cur][d] * K[d];
                    dyv += s_dy[cur][d] * V[d];
                }
                sc = tau * dot / Z;
                dp = dyv / Z;  // d(read)/d p_m . dy = dy . V_m / Z_m
            }
            wmax = warp_max(sc);
            e = valid ? __expf(sc - wmax) : 0.f;
            const float wsum = warp_sum(e), wpd = warp_sum(e * dp);
            if (lane == 0) {
                s_red[cur][warp][0] = wmax;
                s_red[cur][warp][1] = wsum;
                s_red[cur][warp][2] = wpd;
            }
        }
        __syncthreads();
        float alpha = 0.f, beta = 0.f, gamma = 0.f;
        if (gr) {
            float M = -INFINITY;
#pragma unroll
            for (int i = 0; i < NWARP; ++i) M = fmaxf(M, s_red[cur][i][0]);
            float S = 0.f, PD = 0.f;
            if (M > -INFINITY) {
#pragma unroll
                for (int i = 0; i < NWARP; ++i) {
                    const float f = s_red[cur][i][0] > -INFINITY ? __expf(s_red[cur][i][0] - M) : 0.f;
                    S += s_red[cur][i][1] * f;
                    PD += s_red[cur][i][2] * f;
                }
            }
            const float sdp = S > 0.f ? PD / S : 0.f;  // sum_m p_m dp_m = dy . read
            if (valid) {
                const float p = e * __expf(wmax - M) / S;
                const float ds = p * (dp - sdp);
                alpha = tau * ds / Z;
                beta = p / Z;
                gamma = -(ds * sc + p * dp) / Z;
                dtau += ds * sc / tau;
            }
            float u[DH];
#pragma unroll
            for (int d = 0; d < DH; ++d) u[d] = alpha * K[d];
            s_dq[cur][warp][lane] = reduce_scatter32(u, lane);  // dq = tau sum_m ds_m K_m / Z_m
            rd.coef[rh * SLOTS + m] = alpha;
            rd.coef[plane + rh * SLOTS + m] = beta;
            rd.coef[2 * plane + rh * SLOTS + m] = gamma;
            if (m == 0) rd.dg[rh] = sdp;
        } else if (m == 0) {
            rd.dg[rh] = 0.f;
        }
        if (t + 1 < T) load(t + 1, cur ^ 1);
        __syncthreads();
        if (warp == 0) {
            float dq = 0.f;
            if (gr) {
#pragma unroll
                for (int i = 0; i < NWARP; ++i) dq += s_dq[cur][i][lane];
            }
            rd.dq[rh * DH + lane] = dq;
        }
    }
    dtau = warp_sum(dtau);
    if (lane == 0) s_tau[warp] = dtau;
    __syncthreads();
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
    const float am = __expf(w.loga[h * SLOTS + m]);
    __shared__ float s_k[2][DH], s_v[2][DH], s_q[2][2][DH], s_dy[2][2][DH];
    __shared__ StepInfo s_i[2];
    __shared__ float s_dk[2][2][DH], s_dv[2][2][DH], s_dr[2][2];

    auto load = [&](int t, int buf) {
        const size_t r = size_t(b) * T + t;
        const size_t rh = r * H + h;
        if (m < 32) {
            s_k[buf][m] = __bfloat162float(w.k[rh * DH + m]);
            s_v[buf][m] = __bfloat162float(w.v[r * w.ldv + h * DH + m]);
        } else if (m < 64) {
            const int d = m - 32;
            s_q[buf][0][d] = __bfloat162float(a.q[0][rh * DH + d]);
            s_dy[buf][0][d] = a.dy[0][r * H * DH + h * DH + d];
        } else if (m < 96) {
            const int d = m - 64;
            if (a.hops > 1) {
                s_q[buf][1][d] = __bfloat162float(a.q[1][rh * DH + d]);
                s_dy[buf][1][d] = a.dy[1][r * H * DH + h * DH + d];
            }
        } else if (m == 96) {
            load_info(s_i[buf], w, a.gr[0], a.hops > 1 ? a.gr[1] : nullptr, r, h);
        }
    };

    float LK[DH], LV[DH], LZ = 0.f;
#pragma unroll
    for (int d = 0; d < DH; ++d) LK[d] = LV[d] = 0.f;
    load(T - 1, 0);
    __syncthreads();
    for (int t = T - 1; t >= 0; --t) {
        const int cur = (T - 1 - t) & 1;
        const size_t r = size_t(b) * T + t;
        const size_t rh = r * H + h;
        const StepInfo si = s_i[cur];
        const bool gw = si.flags & 1, rst = si.flags & 4;
        // the reads' local gradients on the state after token t
#pragma unroll
        for (int hop = 0; hop < 2; ++hop) {
            if (hop >= a.hops || !(si.flags & (hop == 0 ? 2 : 8))) continue;
            const float* cf = a.coef[hop];
            const float alpha = cf[rh * SLOTS + m], beta = cf[plane + rh * SLOTS + m],
                        gamma = cf[2 * plane + rh * SLOTS + m];
#pragma unroll
            for (int d = 0; d < DH; ++d) {
                LK[d] += alpha * s_q[cur][hop][d];
                LV[d] += beta * s_dy[cur][hop][d];
            }
            LZ += gamma;
        }
        // the write at t: K_m += r_m k, V_m += r_m v, Z_m += r_m for the two routed slots
        if (m == si.m1 || m == si.m2) {
            const int slot = (m == si.m1) ? 0 : 1;
            float dr = LZ;
#pragma unroll
            for (int d = 0; d < DH; ++d) dr += s_k[cur][d] * LK[d] + s_v[cur][d] * LV[d];
            const float rw = gw ? (slot == 0 ? si.p1 : 1.f - si.p1) : 0.f;
            s_dr[cur][slot] = dr;
#pragma unroll
            for (int d = 0; d < DH; ++d) {
                s_dk[cur][slot][d] = rw * LK[d];
                s_dv[cur][slot][d] = rw * LV[d];
            }
        }
        // to the state after token t - 1
        if (rst) {
#pragma unroll
            for (int d = 0; d < DH; ++d) LK[d] = LV[d] = 0.f;
            LZ = 0.f;
        } else {
#pragma unroll
            for (int d = 0; d < DH; ++d) {
                LK[d] *= am;
                LV[d] *= am;
            }
            LZ *= am;
        }
        if (t > 0) load(t - 1, cur ^ 1);
        __syncthreads();
        if (m < 32) {
            a.dk[rh * DH + m] = s_dk[cur][0][m] + s_dk[cur][1][m];
        } else if (m < 64) {
            a.dv[r * a.lddv + h * DH + (m - 32)] = __float2bfloat16(s_dv[cur][0][m - 32] + s_dv[cur][1][m - 32]);
        } else if (m == 64) {
            const float dr1 = s_dr[cur][0], dr2 = s_dr[cur][1];
            a.dp1[rh] = gw ? dr1 - dr2 : 0.f;
            a.dgw[rh] = si.p1 * dr1 + (1.f - si.p1) * dr2;
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
