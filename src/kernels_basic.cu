// Norms, elementwise passes, embedding, the loss rows and the optimizer.

#include <cmath>

#include "kernels.hpp"
#include "lbc/check.hpp"

namespace lbc {
namespace {

constexpr unsigned FULL = 0xffffffffu;

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

// block of 256 threads
__device__ float block_sum(float v, float* sh) {
    v = warp_sum(v);
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    __syncthreads();
    if (lane == 0) sh[warp] = v;
    __syncthreads();
    v = (threadIdx.x < (blockDim.x >> 5)) ? sh[threadIdx.x] : 0.f;
    if (warp == 0) v = warp_sum(v);
    if (threadIdx.x == 0) sh[0] = v;
    __syncthreads();
    return sh[0];
}

__device__ float block_max(float v, float* sh) {
    v = warp_max(v);
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    __syncthreads();
    if (lane == 0) sh[warp] = v;
    __syncthreads();
    v = (threadIdx.x < (blockDim.x >> 5)) ? sh[threadIdx.x] : -INFINITY;
    if (warp == 0) v = warp_max(v);
    if (threadIdx.x == 0) sh[0] = v;
    __syncthreads();
    return sh[0];
}

__global__ void k_rmsnorm_fwd(const float* __restrict__ h, bf16* __restrict__ xh, float* __restrict__ rstd, int M,
                              int D) {
    const int row = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    if (row >= M) return;
    const float* x = h + size_t(row) * D;
    float ss = 0.f;
    for (int j = lane; j < D; j += 32) ss += x[j] * x[j];
    const float r = rsqrtf(warp_sum(ss) / D + RMS_EPS);
    bf16* y = xh + size_t(row) * D;
    for (int j = lane; j < D; j += 32) y[j] = __float2bfloat16(x[j] * r);
    if (lane == 0) rstd[row] = r;
}

// a warp per row, RMS_BWD_ROWS rows per block; the column sums (if any) kept per lane in registers, then summed
// per block in shared memory: one global atomic per column per block
constexpr int RMS_BWD_ROWS = 64;

__global__ void __launch_bounds__(256) k_rmsnorm_bwd(const float* __restrict__ dxh, const bf16* __restrict__ xh,
                                                     const float* __restrict__ rstd, float* __restrict__ dh,
                                                     bf16* __restrict__ dhb, float* __restrict__ colsum, int M,
                                                     int D, int acc) {
    __shared__ float s_col[1024];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nw = blockDim.x >> 5, nd = D / 32;
    if (colsum)
        for (int j = threadIdx.x; j < D; j += blockDim.x) s_col[j] = 0.f;
    float cs[32];
#pragma unroll
    for (int i = 0; i < 32; ++i) cs[i] = 0.f;
    const int r0 = blockIdx.x * RMS_BWD_ROWS, r1 = min(M, r0 + RMS_BWD_ROWS);
    for (int row = r0 + warp; row < r1; row += nw) {
        const float* g = dxh + size_t(row) * D;
        const bf16* y = xh + size_t(row) * D;
        float dot = 0.f;
#pragma unroll
        for (int i = 0; i < 32; ++i)
            if (i < nd) dot += g[lane + 32 * i] * __bfloat162float(y[lane + 32 * i]);
        dot = warp_sum(dot) / D;
        const float r = rstd[row];
        float* out = dh + size_t(row) * D;
        bf16* outb = dhb + size_t(row) * D;
#pragma unroll
        for (int i = 0; i < 32; ++i) {
            if (i < nd) {
                const int j = lane + 32 * i;
                float v = r * (g[j] - __bfloat162float(y[j]) * dot);
                if (acc) v += out[j];
                out[j] = v;
                outb[j] = __float2bfloat16(v);
                cs[i] += v;
            }
        }
    }
    if (!colsum) return;
    __syncthreads();
#pragma unroll
    for (int i = 0; i < 32; ++i)
        if (i < nd) atomicAdd(&s_col[lane + 32 * i], cs[i]);
    __syncthreads();
    for (int j = threadIdx.x; j < D; j += blockDim.x) atomicAdd(colsum + j, s_col[j]);
}

__global__ void k_to_bf16(const float* __restrict__ x, bf16* __restrict__ y, size_t n) {
    for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < n; i += size_t(gridDim.x) * blockDim.x)
        y[i] = __float2bfloat16(x[i]);
}

__global__ void k_add_to_f32(const bf16* __restrict__ x, float* __restrict__ y, size_t n) {
    for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < n; i += size_t(gridDim.x) * blockDim.x)
        y[i] += __bfloat162float(x[i]);
}

__global__ void k_embed_gather(const int* __restrict__ ids, const bf16* __restrict__ E, bf16* __restrict__ x,
                               size_t n, int E0) {
    for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < n; i += size_t(gridDim.x) * blockDim.x)
        x[i] = E[size_t(ids[i / E0]) * E0 + i % E0];
}

__global__ void k_embed_scatter(const int* __restrict__ ids, const float* __restrict__ dx, float* __restrict__ dE,
                                size_t n, int E0) {
    for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < n; i += size_t(gridDim.x) * blockDim.x)
        atomicAdd(dE + size_t(ids[i / E0]) * E0 + i % E0, dx[i]);
}

__global__ void k_embed_gather_f32(const int* __restrict__ ids, const float* __restrict__ E,
                                   float* __restrict__ x, size_t n, int E0) {
    for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < n; i += size_t(gridDim.x) * blockDim.x)
        x[i] = E[size_t(ids[i / E0]) * E0 + i % E0];
}

__global__ void k_scale_cols(const bf16* __restrict__ x, const float* __restrict__ g, bf16* __restrict__ y, size_t n,
                             int D) {
    for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < n; i += size_t(gridDim.x) * blockDim.x)
        y[i] = __float2bfloat16(__bfloat162float(x[i]) * g[i % D]);
}

// SCALE_ROWS rows per block, a thread per column (strided): the column sums in registers, one atomic per column
// per block
constexpr int SCALE_ROWS = 64;

__global__ void __launch_bounds__(256) k_scale_cols_bwd(const bf16* __restrict__ dy, const bf16* __restrict__ x,
                                                        const float* __restrict__ g, float* __restrict__ dx,
                                                        float* __restrict__ dg, int M, int D) {
    const int r0 = blockIdx.x * SCALE_ROWS, r1 = min(M, r0 + SCALE_ROWS);
    for (int j = threadIdx.x; j < D; j += blockDim.x) {
        const float gj = g[j];
        float acc = 0.f;
        for (int r = r0; r < r1; ++r) {
            const size_t i = size_t(r) * D + j;
            const float d = __bfloat162float(dy[i]);
            dx[i] = d * gj;
            acc += d * __bfloat162float(x[i]);
        }
        atomicAdd(dg + j, acc);
    }
}

__global__ void __launch_bounds__(256) k_ce_rows(const float* __restrict__ logits, const int* __restrict__ y,
                                                 bf16* __restrict__ dlogits, float* __restrict__ loss_sum,
                                                 float* __restrict__ correct_sum, int V,
                                                 const float* __restrict__ inv_n_ptr, float* __restrict__ row_loss) {
    const float inv_n = *inv_n_ptr;
    __shared__ float sh[32];
    const size_t row = blockIdx.x;
    const float* l = logits + row * V;
    bf16* d = dlogits ? dlogits + row * V : nullptr;
    const int target = y[row];
    if (target < 0) {
        if (row_loss && threadIdx.x == 0) row_loss[row] = 0.f;
        if (d)
            for (int j = threadIdx.x; j < V; j += blockDim.x) d[j] = __float2bfloat16(0.f);
        return;
    }
    // one pass for the max and the sum (online softmax), merged over the block
    float m = -INFINITY, s = 0.f;
    for (int j = threadIdx.x; j < V; j += blockDim.x) {
        const float x = l[j];
        if (x > m) {
            s = s * __expf(m - x) + 1.f;
            m = x;
        } else {
            s += __expf(x - m);
        }
    }
    const float mb = block_max(m, sh);
    s = block_sum(m > -INFINITY ? s * __expf(m - mb) : 0.f, sh);
    m = mb;
    const float inv = 1.f / s;
    if (d)
        for (int j = threadIdx.x; j < V; j += blockDim.x) {
            float p = __expf(l[j] - m) * inv;
            if (j == target) p -= 1.f;
            d[j] = __float2bfloat16(p * inv_n);
        }
    if (threadIdx.x == 0) {
        atomicAdd(loss_sum, (logf(s) + m - l[target]) * inv_n);
        if (row_loss) row_loss[row] = logf(s) + m - l[target];
        if (l[target] >= m) atomicAdd(correct_sum, inv_n);  // the target is the row's (first) maximum
    }
}

__global__ void k_sumsq(const float* __restrict__ x, size_t n, float* __restrict__ out) {
    float s = 0.f;
    for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < n; i += size_t(gridDim.x) * blockDim.x)
        s += x[i] * x[i];
    s = warp_sum(s);
    if ((threadIdx.x & 31) == 0) atomicAdd(out, s);
}

__global__ void k_adamw(float* __restrict__ p, const float* __restrict__ g, float* __restrict__ m,
                        float* __restrict__ v, const uint8_t* __restrict__ decay, size_t n, Schedule sched, float b1,
                        float b2, float eps, float wd, const int* __restrict__ step, const float* __restrict__ gsq,
                        float clip, float* __restrict__ lr_out) {
    const float bc1 = 1.f - powf(b1, float(*step)), bc2 = 1.f - powf(b2, float(*step));
    const float lr = schedule_lr(sched, *step - 1);  // *step counts this update already
    if (blockIdx.x == 0 && threadIdx.x == 0 && lr_out) *lr_out = lr;
    const float norm = sqrtf(*gsq);
    const float scale = (clip > 0.f && norm > clip) ? clip / (norm + 1e-6f) : 1.f;
    for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < n; i += size_t(gridDim.x) * blockDim.x) {
        const float gi = g[i] * scale;
        const float mi = b1 * m[i] + (1.f - b1) * gi;
        const float vi = b2 * v[i] + (1.f - b2) * gi * gi;
        m[i] = mi;
        v[i] = vi;
        float pi = p[i];
        if (decay[i]) pi -= lr * wd * pi;
        p[i] = pi - lr * (mi / bc1) / (sqrtf(vi / bc2) + eps);
    }
}

__global__ void k_increment(int* x) { ++*x; }

__global__ void k_prep_weight(const float* __restrict__ W, const float* __restrict__ gain, bf16* __restrict__ Wb,
                              int rows, int cols) {
    const size_t n = size_t(rows) * cols;
    for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < n; i += size_t(gridDim.x) * blockDim.x)
        Wb[i] = __float2bfloat16(gain ? gain[i / cols] * W[i] : W[i]);
}

__global__ void k_unfold_grad(const float* __restrict__ dWp, const float* __restrict__ W,
                              const float* __restrict__ gain, float* __restrict__ dW, float* __restrict__ dgain,
                              int rows, int cols) {
    const int row = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    if (row >= rows) return;
    const float g = gain ? gain[row] : 1.f;
    float acc = 0.f;
    for (int j = lane; j < cols; j += 32) {
        const size_t i = size_t(row) * cols + j;
        dW[i] += g * dWp[i];
        acc += W[i] * dWp[i];
    }
    if (gain) {
        acc = warp_sum(acc);
        if (lane == 0) dgain[row] += acc;
    }
}

int blocks_for(size_t n, int threads = 256) {
    const size_t b = (n + threads - 1) / threads;
    return int(b < 65536 ? b : 65536);
}

}  // namespace

void rmsnorm_fwd(const float* h, bf16* xh, float* rstd, int M, int D, cudaStream_t s) {
    k_rmsnorm_fwd<<<(M + 7) / 8, 256, 0, s>>>(h, xh, rstd, M, D);
    LBC_CUDA(cudaGetLastError());
}

void rmsnorm_bwd(const float* dxh, const bf16* xh, const float* rstd, float* dh, bf16* dhb, float* colsum, int M,
                 int D, bool accumulate, cudaStream_t s) {
    if (D % 32 || D > 1024) fail("rmsnorm_bwd", "D must be a multiple of 32, at most 1024", __FILE__, __LINE__);
    k_rmsnorm_bwd<<<(M + RMS_BWD_ROWS - 1) / RMS_BWD_ROWS, 256, 0, s>>>(dxh, xh, rstd, dh, dhb, colsum, M, D,
                                                                      accumulate ? 1 : 0);
    LBC_CUDA(cudaGetLastError());
}

void to_bf16(const float* x, bf16* y, size_t n, cudaStream_t s) {
    k_to_bf16<<<blocks_for(n), 256, 0, s>>>(x, y, n);
    LBC_CUDA(cudaGetLastError());
}

void add_to_f32(const bf16* x, float* y, size_t n, cudaStream_t s) {
    k_add_to_f32<<<blocks_for(n), 256, 0, s>>>(x, y, n);
    LBC_CUDA(cudaGetLastError());
}

void embed_gather(const int* ids, const bf16* E, bf16* x, int M, int E0, cudaStream_t s) {
    const size_t n = size_t(M) * E0;
    k_embed_gather<<<blocks_for(n), 256, 0, s>>>(ids, E, x, n, E0);
    LBC_CUDA(cudaGetLastError());
}

void embed_gather_f32(const int* ids, const float* E, float* x, int M, int E0, cudaStream_t s) {
    const size_t n = size_t(M) * E0;
    k_embed_gather_f32<<<blocks_for(n), 256, 0, s>>>(ids, E, x, n, E0);
    LBC_CUDA(cudaGetLastError());
}

void scale_cols(const bf16* x, const float* g, bf16* y, int M, int D, cudaStream_t s) {
    const size_t n = size_t(M) * D;
    k_scale_cols<<<blocks_for(n), 256, 0, s>>>(x, g, y, n, D);
    LBC_CUDA(cudaGetLastError());
}

void scale_cols_bwd(const bf16* dy, const bf16* x, const float* g, float* dx, float* dg, int M, int D,
                    cudaStream_t s) {
    k_scale_cols_bwd<<<(M + SCALE_ROWS - 1) / SCALE_ROWS, 256, 0, s>>>(dy, x, g, dx, dg, M, D);
    LBC_CUDA(cudaGetLastError());
}

void embed_scatter(const int* ids, const float* dx, float* dE, int M, int E0, cudaStream_t s) {
    const size_t n = size_t(M) * E0;
    k_embed_scatter<<<blocks_for(n), 256, 0, s>>>(ids, dx, dE, n, E0);
    LBC_CUDA(cudaGetLastError());
}

void ce_rows(const float* logits, const int* y, bf16* dlogits, float* loss_sum, float* correct_sum, int rows, int V,
             const float* inv_n, cudaStream_t s, float* row_loss) {
    k_ce_rows<<<rows, 256, 0, s>>>(logits, y, dlogits, loss_sum, correct_sum, V, inv_n, row_loss);
    LBC_CUDA(cudaGetLastError());
}

void sumsq(const float* x, size_t n, float* out, cudaStream_t s) {
    k_sumsq<<<blocks_for(n), 256, 0, s>>>(x, n, out);
    LBC_CUDA(cudaGetLastError());
}

void adamw(float* p, const float* g, float* m, float* v, const uint8_t* decay, size_t n, const Schedule& sched,
           float beta1, float beta2, float eps, float wd, const int* step, const float* gsq, float clip,
           float* lr_out, cudaStream_t s) {
    k_adamw<<<blocks_for(n), 256, 0, s>>>(p, g, m, v, decay, n, sched, beta1, beta2, eps, wd, step, gsq, clip,
                                          lr_out);
    LBC_CUDA(cudaGetLastError());
}

void increment(int* x, cudaStream_t s) {
    k_increment<<<1, 1, 0, s>>>(x);
    LBC_CUDA(cudaGetLastError());
}

void prep_weight(const float* W, const float* gain, bf16* Wb, int rows, int cols, cudaStream_t s) {
    k_prep_weight<<<blocks_for(size_t(rows) * cols), 256, 0, s>>>(W, gain, Wb, rows, cols);
    LBC_CUDA(cudaGetLastError());
}

void unfold_grad(const float* dWp, const float* W, const float* gain, float* dW, float* dgain, int rows, int cols,
                 cudaStream_t s) {
    k_unfold_grad<<<(rows + 7) / 8, 256, 0, s>>>(dWp, W, gain, dW, dgain, rows, cols);
    LBC_CUDA(cudaGetLastError());
}

}  // namespace lbc
