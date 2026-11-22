// The LRU scan, and the addressing: keys, queries, top-2 routing, binary read / write gates.

#include <cmath>

#include "kernels.hpp"
#include "lbc/check.hpp"

namespace lbc {
namespace {

constexpr unsigned FULL = 0xffffffffu;
constexpr int LRU_UNROLL = 8;    // timesteps loaded together (independent loads in flight)
constexpr int RPAD = SLOTS + 1;  // router rows padded: lane d reading column m hits bank (d + m) % 32

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o; o >>= 1) { v += __shfl_xor_sync(FULL, v, o); }
    return v;
}

__device__ __forceinline__ float sigmoid(float x) { return 1.f / (1.f + __expf(-x)); }

// (value, index) argmax over the warp, ties to the lower index; every lane gets the result
__device__ __forceinline__ void warp_argmax(float& v, int& i) {
#pragma unroll
    for (int o = 16; o; o >>= 1) {
        const float ov = __shfl_xor_sync(FULL, v, o);
        const int oi = __shfl_xor_sync(FULL, i, o);
        if (ov > v || (ov == v && oi < i)) {
            v = ov;
            i = oi;
        }
    }
}

struct LruConst {
    float rho, lr, li, gam;
};

__device__ __forceinline__ LruConst lru_const(float nu, float theta) {
    LruConst c;
    c.rho = __expf(nu);
    const float mag = __expf(-c.rho);
    c.lr = mag * cosf(theta);
    c.li = mag * sinf(theta);
    c.gam = sqrtf(fmaxf(-expm1f(-2.f * c.rho), 1e-12f));
    return c;
}

__global__ void k_lru_fwd(const bf16* __restrict__ P,
                          int ldp,
                          const float* __restrict__ nu,
                          const float* __restrict__ theta,
                          const uint8_t* __restrict__ reset,
                          float* __restrict__ s,
                          bf16* __restrict__ sb,
                          int B,
                          int T,
                          int N) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * N) { return; }
    const int b = idx / N, n = idx % N;
    const LruConst c = lru_const(nu[n], theta[n]);
    const bf16* p = P + size_t(b) * T * ldp;
    float* so = s + size_t(b) * T * 2 * N;
    bf16* sbo = sb + size_t(b) * T * 2 * N;
    const uint8_t* rs = reset + size_t(b) * T;
    float hr = 0.f, hi = 0.f;
    for (int t0 = 0; t0 < T; t0 += LRU_UNROLL) {
        float xr[LRU_UNROLL], xi[LRU_UNROLL];
        uint8_t rr[LRU_UNROLL];
#pragma unroll
        for (int u = 0; u < LRU_UNROLL; ++u) {
            const size_t row = size_t(t0 + u) * ldp;
            xr[u] = __bfloat162float(p[row + n]);
            xi[u] = __bfloat162float(p[row + N + n]);
            rr[u] = rs[t0 + u];
        }
#pragma unroll
        for (int u = 0; u < LRU_UNROLL; ++u) {
            const float keep = rr[u] ? 0.f : 1.f;
            const float nr = keep * (c.lr * hr - c.li * hi) + c.gam * xr[u];
            hi = keep * (c.lr * hi + c.li * hr) + c.gam * xi[u];
            hr = nr;
            const size_t o = size_t(t0 + u) * 2 * N;
            so[o + n] = hr;
            so[o + N + n] = hi;
            sbo[o + n] = __float2bfloat16(hr);
            sbo[o + N + n] = __float2bfloat16(hi);
        }
    }
}

__global__ void k_lru_bwd(const float* __restrict__ ds,
                          const float* __restrict__ s,
                          const bf16* __restrict__ P,
                          int ldp,
                          const float* __restrict__ nu,
                          const float* __restrict__ theta,
                          const uint8_t* __restrict__ reset,
                          bf16* __restrict__ dP,
                          int lddp,
                          float* __restrict__ dnu,
                          float* __restrict__ dtheta,
                          int B,
                          int T,
                          int N) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * N) { return; }
    const int b = idx / N, n = idx % N;
    const LruConst c = lru_const(nu[n], theta[n]);
    const size_t S2 = 2 * size_t(N);
    const float* dsb = ds + size_t(b) * T * S2;
    const float* sbb = s + size_t(b) * T * S2;
    const bf16* p = P + size_t(b) * T * ldp;
    bf16* dp = dP + size_t(b) * T * lddp;
    const uint8_t* rs = reset + size_t(b) * T;
    float cr = 0.f, ci = 0.f, dlr = 0.f, dli = 0.f, dgam = 0.f;
    for (int t0 = T - LRU_UNROLL; t0 >= 0; t0 -= LRU_UNROLL) {
        float gr_[LRU_UNROLL], gi_[LRU_UNROLL], xr[LRU_UNROLL], xi[LRU_UNROLL], pr[LRU_UNROLL], pi[LRU_UNROLL];
        uint8_t rr[LRU_UNROLL];
#pragma unroll
        for (int u = 0; u < LRU_UNROLL; ++u) {
            const int t = t0 + u;
            gr_[u] = dsb[t * S2 + n];
            gi_[u] = dsb[t * S2 + N + n];
            xr[u] = __bfloat162float(p[size_t(t) * ldp + n]);
            xi[u] = __bfloat162float(p[size_t(t) * ldp + N + n]);
            pr[u] = t > 0 ? sbb[(t - 1) * S2 + n] : 0.f;
            pi[u] = t > 0 ? sbb[(t - 1) * S2 + N + n] : 0.f;
            rr[u] = rs[t];
        }
#pragma unroll
        for (int u = LRU_UNROLL - 1; u >= 0; --u) {
            const int t = t0 + u;
            const float gr = gr_[u] + cr, gi = gi_[u] + ci;
            dp[size_t(t) * lddp + n] = __float2bfloat16(c.gam * gr);
            dp[size_t(t) * lddp + N + n] = __float2bfloat16(c.gam * gi);
            dgam += gr * xr[u] + gi * xi[u];
            const float keep = rr[u] ? 0.f : 1.f;
            const float hpr = keep * pr[u], hpi = keep * pi[u];
            dlr += gr * hpr + gi * hpi;
            dli += -gr * hpi + gi * hpr;
            cr = keep * (c.lr * gr + c.li * gi);
            ci = keep * (c.lr * gi - c.li * gr);
        }
    }
    const float drho = -(c.lr * dlr + c.li * dli) + dgam * __expf(-2.f * c.rho) / c.gam;
    atomicAdd(dnu + n, drho * c.rho);
    atomicAdd(dtheta + n, -c.li * dlr + c.lr * dli);
}

// Head sizes DH = 32 or 64 (templates): lane l of a row's warp holds dimensions l + 32 c, c < DH / 32.
// One block per (head, chunk of rows); the head's router (DH x SLOTS) sits in (dynamic) shared memory.
template <int DH>
__global__ void __launch_bounds__(256) k_addr_fwd(AddrFwd a, int rows_per_block) {
    constexpr int E = DH / 32;
    extern __shared__ float smem[];
    float* Rs = smem;                // [DH][RPAD]
    float* st_s = smem + DH * RPAD;  // [4]
    const int h = blockIdx.x, H = a.H, HD = H * DH;
    for (int i = threadIdx.x; i < DH * SLOTS; i += blockDim.x) {
        Rs[(i / SLOTS) * RPAD + i % SLOTS] = a.R[size_t(h) * DH * SLOTS + i];
    }
    if (threadIdx.x < 4) { st_s[threadIdx.x] = 0.f; }
    __syncthreads();
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nw = blockDim.x >> 5;
    const int r0 = blockIdx.y * rows_per_block, r1 = min(a.M, r0 + rows_per_block);
    float sw = 0.f, cw = 0.f, sr = 0.f, cr = 0.f;
    for (int r = r0 + warp; r < r1; r += nw) {
        const int t = a.tpos ? a.tpos[r] : r % a.T;
        const bool rst = a.reset[r];
        const bf16* gprev = a.Gprev ? a.Gprev + size_t(a.map[r]) * a.ldGprev : a.G + size_t(r - 1) * a.ldG;
        float kp[E], qp[E], ks = 0.f, qs = 0.f;
#pragma unroll
        for (int c = 0; c < E; ++c) {
            const int d = lane + 32 * c;
            kp[c] = a.bk[h * DH + d];
            if (t > 0 && !rst) { kp[c] += __bfloat162float(gprev[h * DH + d]); }
            qp[c] = __bfloat162float(a.G[size_t(r) * a.ldG + HD + h * DH + d]);
            ks += kp[c] * kp[c];
            qs += qp[c] * qp[c];
        }
        const float kn = sqrtf(warp_sum(ks)), qn = sqrtf(warp_sum(qs));
        float k[E];
#pragma unroll
        for (int c = 0; c < E; ++c) { k[c] = kp[c] / fmaxf(kn, NORM_EPS); }
        // router logits l_m = k . R[:, m], lane holds m = lane + 32 j
        float l[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
        for (int c = 0; c < E; ++c) {
#pragma unroll 8
            for (int dd = 0; dd < 32; ++dd) {
                const float kd = __shfl_sync(FULL, k[c], dd);
                const float* row = Rs + (32 * c + dd) * RPAD + lane;
#pragma unroll
                for (int j = 0; j < 4; ++j) { l[j] += kd * row[32 * j]; }
            }
        }
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            if (lane + 32 * j >= a.slots) {
                l[j] = -INFINITY;  // past the layer's bank size
            }
        }
        float v1 = l[0], v2 = -INFINITY;
        int i1 = lane, i2 = -1;
#pragma unroll
        for (int j = 1; j < 4; ++j) {
            if (l[j] > v1) {
                v2 = v1;
                i2 = i1;
                v1 = l[j];
                i1 = lane + 32 * j;
            } else if (l[j] > v2) {
                v2 = l[j];
                i2 = lane + 32 * j;
            }
        }
        float b1 = v1;
        int j1 = i1;
        warp_argmax(b1, j1);
        float b2 = (i1 == j1) ? v2 : v1;
        int j2 = (i1 == j1) ? i2 : i1;
        if (j2 < 0) {
            j2 = SLOTS;  // a lane whose 4 slots all lost (cannot win the argmax below)
        }
        warp_argmax(b2, j2);
        const size_t rh = size_t(r) * H + h;
#pragma unroll
        for (int c = 0; c < E; ++c) {
            a.k[rh * DH + lane + 32 * c] = __float2bfloat16(k[c]);
            a.q[rh * DH + lane + 32 * c] = __float2bfloat16(qp[c] / fmaxf(qn, NORM_EPS));
        }
        if (lane == 0) {
            const float aw = __bfloat162float(a.G[size_t(r) * a.ldG + 2 * HD + H + h]) + a.bwg[h];
            const float ar = __bfloat162float(a.G[size_t(r) * a.ldG + 2 * HD + h]) + a.brg[h];
            const bool gw = !rst && t > 0 && aw > 0.f, gr = ar > 0.f;  // no write without a previous state
            a.kn[rh] = kn;
            a.qn[rh] = qn;
            a.ridx[rh * 2] = uint8_t(j1);
            a.ridx[rh * 2 + 1] = uint8_t(j2);
            a.rp1[rh] = sigmoid(b1 - b2);
            a.gw[rh] = gw;
            a.gr[rh] = gr;
            a.info[rh] = make_uint2(
                unsigned(j1) | unsigned(j2) << 8 | unsigned((gw ? 1 : 0) | (gr ? 2 : 0) | (rst ? 4 : 0)) << 16,
                __float_as_uint(sigmoid(b1 - b2)));
            a.aw[rh] = aw;
            a.ar[rh] = ar;
            if (!rst && t > 0) { sw += sigmoid(aw); }
            cw += gw;
            sr += sigmoid(ar);
            cr += gr;
        }
    }
    if (lane == 0) {  // per block, not per warp: these four addresses are shared by the whole grid
        atomicAdd(&st_s[0], sw);
        atomicAdd(&st_s[1], cw);
        atomicAdd(&st_s[2], sr);
        atomicAdd(&st_s[3], cr);
    }
    __syncthreads();
    if (threadIdx.x < 4) { atomicAdd(a.stats + threadIdx.x, st_s[threadIdx.x]); }
}

__global__ void k_lru_step(const bf16* __restrict__ P,
                           int ldp,
                           const float* __restrict__ nu,
                           const float* __restrict__ theta,
                           const uint8_t* __restrict__ reset,
                           const int* __restrict__ map,
                           float* __restrict__ state,
                           bf16* __restrict__ sb,
                           int R,
                           int N) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= R * N) { return; }
    const int r = idx / N, n = idx % N;
    const LruConst c = lru_const(nu[n], theta[n]);
    float* h = state + size_t(map[r]) * 2 * N;
    const float keep = reset[r] ? 0.f : 1.f;
    const float xr = __bfloat162float(P[size_t(r) * ldp + n]), xi = __bfloat162float(P[size_t(r) * ldp + N + n]);
    const float hr = h[n], hi = h[N + n];
    const float nr = keep * (c.lr * hr - c.li * hi) + c.gam * xr;  // as k_lru_fwd
    const float ni = keep * (c.lr * hi + c.li * hr) + c.gam * xi;
    h[n] = nr;
    h[N + n] = ni;
    sb[size_t(r) * 2 * N + n] = __float2bfloat16(nr);
    sb[size_t(r) * 2 * N + N + n] = __float2bfloat16(ni);
}

__global__ void k_store_rows(const bf16* __restrict__ G,
                             int ldG,
                             int cols,
                             const int* __restrict__ map,
                             bf16* __restrict__ Gp,
                             int ldp,
                             int R) {
    const size_t idx = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= size_t(R) * cols) { return; }
    const int r = int(idx / size_t(cols)), c = int(idx % size_t(cols));
    Gp[size_t(map[r]) * ldp + c] = G[size_t(r) * ldG + c];
}

template <int DH>
__global__ void __launch_bounds__(256) k_addr_bwd(AddrBwd a, int rows_per_block) {
    constexpr int E = DH / 32;
    extern __shared__ float smem[];
    float* Rs = smem;                     // [DH][RPAD]
    float* dRs = smem + DH * RPAD;        // [DH][RPAD]
    float* dbk_s = smem + 2 * DH * RPAD;  // [DH]
    float* dg_s = dbk_s + DH;             // [2]
    const int h = blockIdx.x, H = a.H, HD = H * DH;
    for (int i = threadIdx.x; i < DH * SLOTS; i += blockDim.x) {
        Rs[(i / SLOTS) * RPAD + i % SLOTS] = a.R[size_t(h) * DH * SLOTS + i];
        dRs[(i / SLOTS) * RPAD + i % SLOTS] = 0.f;
    }
    for (int i = threadIdx.x; i < DH; i += blockDim.x) { dbk_s[i] = 0.f; }
    if (threadIdx.x < 2) { dg_s[threadIdx.x] = 0.f; }
    __syncthreads();
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nw = blockDim.x >> 5;
    const int r0 = blockIdx.y * rows_per_block, r1 = min(a.M, r0 + rows_per_block);
    for (int r = r0 + warp; r < r1; r += nw) {
        const int t = r % a.T;
        const bool rst = a.reset[r];
        const size_t rh = size_t(r) * H + h;
        const bool gw = a.gw[rh];
        float k[E], dk[E];
        // the write's gradients from the two routed slots (bank adjoint): key, value, slot weights
#pragma unroll
        for (int c = 0; c < E; ++c) {
            const int d = lane + 32 * c;
            k[c] = __bfloat162float(a.k[rh * DH + d]);
            dk[c] =
                gw ? __bfloat162float(a.dk2[rh * 2 * DH + d]) + __bfloat162float(a.dk2[(rh * 2 + 1) * DH + d]) : 0.f;
            a.dv[size_t(r) * a.lddv + h * DH + d] = __float2bfloat16(
                gw ? __bfloat162float(a.dv2[rh * 2 * DH + d]) + __bfloat162float(a.dv2[(rh * 2 + 1) * DH + d]) : 0.f);
        }
        const float dr1 = a.dr2[rh * 2], dr2 = a.dr2[rh * 2 + 1];
        // routing: weights gw p1, gw (1 - p1) with p1 = sigmoid(l1 - l2), l_m = k . R[:, m]
        const int m1 = a.ridx[rh * 2], m2 = a.ridx[rh * 2 + 1];
        const float p1 = a.rp1[rh];
        const float dl = (gw ? dr1 - dr2 : 0.f) * p1 * (1.f - p1);
        // straight-through, for closed write gates too. A closed gate's slot gradient dr is the tangent at weight 0,
        // (k - average) / Z: unbounded as the slot's mass Z decays. Its secant to the open state (the weight r the
        // write would have had) is the tangent times Z / (Z + r), Z the slot's mass just before the write (zpre,
        // from the hop-1 forward walk); bounded, and exact when no other write follows before the reads.
        float s1 = 1.f, s2 = 1.f;
        if (!gw && a.zpre) {
            const float z1 = a.zpre[rh * 2], z2 = a.zpre[rh * 2 + 1];
            s1 = z1 / fmaxf(z1 + p1, 1e-30f);  // (1 - p1 can round to 0)
            s2 = z2 / fmaxf(z2 + (1.f - p1), 1e-30f);
        }
        const float dgw = p1 * dr1 * s1 + (1.f - p1) * dr2 * s2;
        float kdk = 0.f;
#pragma unroll
        for (int c = 0; c < E; ++c) {
            const int d = lane + 32 * c;
            dk[c] += dl * (Rs[d * RPAD + m1] - Rs[d * RPAD + m2]);
            atomicAdd(&dRs[d * RPAD + m1], dl * k[c]);
            atomicAdd(&dRs[d * RPAD + m2], -dl * k[c]);
            kdk += k[c] * dk[c];
        }
        kdk = warp_sum(kdk);
        float qdq = 0.f, q[E], dq[E];
#pragma unroll
        for (int c = 0; c < E; ++c) {
            q[c] = __bfloat162float(a.q[rh * DH + lane + 32 * c]);
            dq[c] = a.dq[rh * DH + lane + 32 * c];
            qdq += q[c] * dq[c];
        }
        qdq = warp_sum(qdq);
#pragma unroll
        for (int c = 0; c < E; ++c) {
            const int d = lane + 32 * c;
            // k = k_pre / |k_pre|
            const float dkp = (dk[c] - k[c] * kdk) / fmaxf(a.kn[rh], NORM_EPS);
            // k_pre at row r is G_k at row r - 1 (zero without a previous state); every G_k row gets written here:
            // row r - 1 by row r, a row's last token by itself (nothing reads it)
            if (t > 0) { a.dG[size_t(r - 1) * a.ldG + h * DH + d] = __float2bfloat16(rst ? 0.f : dkp); }
            if (t == a.T - 1) { a.dG[size_t(r) * a.ldG + h * DH + d] = __float2bfloat16(0.f); }
            atomicAdd(&dbk_s[d], dkp);
            const float dqp = (dq[c] - q[c] * qdq) / fmaxf(a.qn[rh], NORM_EPS);
            a.dG[size_t(r) * a.ldG + HD + h * DH + d] = __float2bfloat16(dqp);
        }
        if (lane == 0) {
            const float bw = a.budget > 0.f ? 2.f * a.budget * (a.stats[0] / a.count - a.target) / a.count : 0.f;
            const float br = a.budget > 0.f ? 2.f * a.budget * (a.stats[2] / a.count - a.target) / a.count : 0.f;
            const float sr = sigmoid(a.ar[rh]);
            const float dar = (a.dgr[rh] + br) * sr * (1.f - sr);
            const float sw = sigmoid(a.aw[rh]);
            const float daw = (rst || t == 0) ? 0.f : (dgw + bw) * sw * (1.f - sw);
            a.dG[size_t(r) * a.ldG + 2 * HD + h] = __float2bfloat16(dar);
            a.dG[size_t(r) * a.ldG + 2 * HD + H + h] = __float2bfloat16(daw);
            atomicAdd(&dg_s[0], dar);
            atomicAdd(&dg_s[1], daw);
        }
    }
    __syncthreads();
    for (int i = threadIdx.x; i < DH * SLOTS; i += blockDim.x) {
        atomicAdd(a.dR + size_t(h) * DH * SLOTS + i, dRs[(i / SLOTS) * RPAD + i % SLOTS]);
    }
    for (int i = threadIdx.x; i < DH; i += blockDim.x) { atomicAdd(a.dbk + h * DH + i, dbk_s[i]); }
    if (threadIdx.x == 0) {
        atomicAdd(a.dbrg + h, dg_s[0]);
        atomicAdd(a.dbwg + h, dg_s[1]);
    }
}

// one block per (head, chunk of rows), a warp per row; sums reduced per block before the global atomics
template <int DH>
__global__ void __launch_bounds__(256) k_q2_fwd(Q2Fwd a, int rows_per_block) {
    constexpr int E = DH / 32;
    __shared__ float st_s[2];
    if (threadIdx.x < 2) { st_s[threadIdx.x] = 0.f; }
    __syncthreads();
    const int h = blockIdx.x, HD = a.H * DH;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nw = blockDim.x >> 5;
    const int r0 = blockIdx.y * rows_per_block, r1 = min(a.M, r0 + rows_per_block);
    float ss = 0.f, so = 0.f;
    for (int r = r0 + warp; r < r1; r += nw) {
        const size_t w = size_t(r) * a.H + h;
        float qp[E], qs = 0.f;
#pragma unroll
        for (int c = 0; c < E; ++c) {
            qp[c] = __bfloat162float(a.G2[size_t(r) * a.ldG2 + h * DH + lane + 32 * c]);
            qs += qp[c] * qp[c];
        }
        const float qn = sqrtf(warp_sum(qs));
#pragma unroll
        for (int c = 0; c < E; ++c) { a.q[w * DH + lane + 32 * c] = __float2bfloat16(qp[c] / fmaxf(qn, NORM_EPS)); }
        if (lane == 0) {
            const float ar = __bfloat162float(a.G2[size_t(r) * a.ldG2 + HD + h]) + a.brg[h];
            a.qn[w] = qn;
            a.ar[w] = ar;
            a.gr[w] = ar > 0.f;
            if (ar > 0.f) { a.info[w].x |= 8u << 16; }
            ss += sigmoid(ar);
            so += ar > 0.f ? 1.f : 0.f;
        }
    }
    if (lane == 0) {
        atomicAdd(&st_s[0], ss);
        atomicAdd(&st_s[1], so);
    }
    __syncthreads();
    if (threadIdx.x < 2) { atomicAdd(a.stats + threadIdx.x, st_s[threadIdx.x]); }
}

template <int DH>
__global__ void __launch_bounds__(256) k_q2_bwd(Q2Bwd a, int rows_per_block) {
    constexpr int E = DH / 32;
    __shared__ float db_s;
    if (threadIdx.x == 0) { db_s = 0.f; }
    __syncthreads();
    const int h = blockIdx.x, HD = a.H * DH;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nw = blockDim.x >> 5;
    const int r0 = blockIdx.y * rows_per_block, r1 = min(a.M, r0 + rows_per_block);
    const float br = a.budget > 0.f ? 2.f * a.budget * (a.stats[0] / a.count - a.target) / a.count : 0.f;
    float db = 0.f;
    for (int r = r0 + warp; r < r1; r += nw) {
        const size_t w = size_t(r) * a.H + h;
        float q[E], dq[E], qdq = 0.f;
#pragma unroll
        for (int c = 0; c < E; ++c) {
            q[c] = __bfloat162float(a.q[w * DH + lane + 32 * c]);
            dq[c] = a.dq[w * DH + lane + 32 * c];
            qdq += q[c] * dq[c];
        }
        qdq = warp_sum(qdq);
#pragma unroll
        for (int c = 0; c < E; ++c) {
            a.dG2[size_t(r) * a.ldG2 + h * DH + lane + 32 * c] =
                __float2bfloat16((dq[c] - q[c] * qdq) / fmaxf(a.qn[w], NORM_EPS));
        }
        if (lane == 0) {
            const float sr = sigmoid(a.ar[w]);
            const float dar = (a.dgr[w] + br) * sr * (1.f - sr);
            a.dG2[size_t(r) * a.ldG2 + HD + h] = __float2bfloat16(dar);
            db += dar;
        }
    }
    if (lane == 0) { atomicAdd(&db_s, db); }
    __syncthreads();
    if (threadIdx.x == 0) { atomicAdd(a.dbrg + h, db_s); }
}

void check_dh(int dh) {
    if (dh != 32 && dh != 64) { fail("addressing", "head size must be 32 or 64", __FILE__, __LINE__); }
}

template <int DH>
size_t addr_fwd_smem() {
    return size_t(DH * RPAD + 4) * sizeof(float);
}

template <int DH>
size_t addr_bwd_smem() {
    return size_t(2 * DH * RPAD + DH + 2) * sizeof(float);
}

// the router tables exceed the 48 KB default for dynamic shared memory at DH = 64 (the backward's two)
template <int DH>
void allow_smem() {
    static bool done = false;
    if (done) { return; }
    LBC_CUDA(
        cudaFuncSetAttribute(k_addr_fwd<DH>, cudaFuncAttributeMaxDynamicSharedMemorySize, int(addr_fwd_smem<DH>())));
    LBC_CUDA(
        cudaFuncSetAttribute(k_addr_bwd<DH>, cudaFuncAttributeMaxDynamicSharedMemorySize, int(addr_bwd_smem<DH>())));
    done = true;
}

}  // namespace

void lru_fwd(const bf16* P,
             int ldp,
             const float* nu,
             const float* theta,
             const uint8_t* reset,
             float* s,
             bf16* sb,
             int B,
             int T,
             int N,
             cudaStream_t st) {
    if (T % LRU_UNROLL) { fail("lru_fwd", "T must be a multiple of 8", __FILE__, __LINE__); }
    k_lru_fwd<<<(B * N + 127) / 128, 128, 0, st>>>(P, ldp, nu, theta, reset, s, sb, B, T, N);
    LBC_CUDA(cudaGetLastError());
}

void lru_bwd(const float* ds,
             const float* s,
             const bf16* P,
             int ldp,
             const float* nu,
             const float* theta,
             const uint8_t* reset,
             bf16* dP,
             int lddp,
             float* dnu,
             float* dtheta,
             int B,
             int T,
             int N,
             cudaStream_t st) {
    k_lru_bwd<<<(B * N + 127) / 128, 128, 0, st>>>(ds, s, P, ldp, nu, theta, reset, dP, lddp, dnu, dtheta, B, T, N);
    LBC_CUDA(cudaGetLastError());
}

// 512 rows per block: each block flushes its head's router gradient (DH x SLOTS) once, so fewer blocks mean fewer
// global atomics (64 rows per block was 33.5M per layer)
constexpr int ADDR_ROWS = 512;

void lru_step(const bf16* P,
              int ldp,
              const float* nu,
              const float* theta,
              const uint8_t* reset,
              const int* map,
              float* state,
              bf16* sb,
              int R,
              int N,
              cudaStream_t st) {
    k_lru_step<<<(R * N + 127) / 128, 128, 0, st>>>(P, ldp, nu, theta, reset, map, state, sb, R, N);
    LBC_CUDA(cudaGetLastError());
}

void store_rows(const bf16* G, int ldG, int cols, const int* map, bf16* Gprev, int ldGprev, int R, cudaStream_t st) {
    const size_t n = size_t(R) * cols;
    k_store_rows<<<unsigned((n + 255) / 256), 256, 0, st>>>(G, ldG, cols, map, Gprev, ldGprev, R);
    LBC_CUDA(cudaGetLastError());
}

void addr_fwd(const AddrFwd& a, cudaStream_t s) {
    const int rows = ADDR_ROWS;
    const dim3 grid(a.H, (a.M + rows - 1) / rows);
    check_dh(a.dh);
    if (a.dh == 64) {
        allow_smem<64>();
        k_addr_fwd<64><<<grid, 256, addr_fwd_smem<64>(), s>>>(a, rows);
    } else {
        allow_smem<32>();
        k_addr_fwd<32><<<grid, 256, addr_fwd_smem<32>(), s>>>(a, rows);
    }
    LBC_CUDA(cudaGetLastError());
}

void addr_bwd(const AddrBwd& a, cudaStream_t s) {
    const int rows = ADDR_ROWS;
    const dim3 grid(a.H, (a.M + rows - 1) / rows);
    check_dh(a.dh);
    if (a.dh == 64) {
        allow_smem<64>();
        k_addr_bwd<64><<<grid, 256, addr_bwd_smem<64>(), s>>>(a, rows);
    } else {
        allow_smem<32>();
        k_addr_bwd<32><<<grid, 256, addr_bwd_smem<32>(), s>>>(a, rows);
    }
    LBC_CUDA(cudaGetLastError());
}

void q2_fwd(const Q2Fwd& a, cudaStream_t s) {
    const int rows = ADDR_ROWS;
    const dim3 grid(a.H, (a.M + rows - 1) / rows);
    check_dh(a.dh);
    if (a.dh == 64) {
        k_q2_fwd<64><<<grid, 256, 0, s>>>(a, rows);
    } else {
        k_q2_fwd<32><<<grid, 256, 0, s>>>(a, rows);
    }
    LBC_CUDA(cudaGetLastError());
}

void q2_bwd(const Q2Bwd& a, cudaStream_t s) {
    const int rows = ADDR_ROWS;
    const dim3 grid(a.H, (a.M + rows - 1) / rows);
    check_dh(a.dh);
    if (a.dh == 64) {
        k_q2_bwd<64><<<grid, 256, 0, s>>>(a, rows);
    } else {
        k_q2_bwd<32><<<grid, 256, 0, s>>>(a, rows);
    }
    LBC_CUDA(cudaGetLastError());
}

}  // namespace lbc
