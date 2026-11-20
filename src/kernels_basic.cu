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

__device__ __forceinline__ float to_f(float x) { return x; }
__device__ __forceinline__ float to_f(bf16 x) { return __bfloat162float(x); }

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

__device__ __forceinline__ float gelu_tanh(float z) {
    const float c = 0.7978845608028654f;
    return 0.5f * z * (1.f + tanhf(c * (z + 0.044715f * z * z * z)));
}

__device__ __forceinline__ float gelu_tanh_grad(float z) {
    const float c = 0.7978845608028654f;
    const float z2 = z * z;
    const float t = tanhf(c * (z + 0.044715f * z * z2));
    return 0.5f * (1.f + t) + 0.5f * z * (1.f - t * t) * c * (1.f + 3.f * 0.044715f * z2);
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

__global__ void k_rmsnorm_bwd(const float* __restrict__ dxh, const bf16* __restrict__ xh,
                              const float* __restrict__ rstd, float* __restrict__ dh, int M, int D, int acc) {
    const int row = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    if (row >= M) return;
    const float* g = dxh + size_t(row) * D;
    const bf16* y = xh + size_t(row) * D;
    float dot = 0.f;
    for (int j = lane; j < D; j += 32) dot += g[j] * __bfloat162float(y[j]);
    dot = warp_sum(dot) / D;
    const float r = rstd[row];
    float* out = dh + size_t(row) * D;
    for (int j = lane; j < D; j += 32) {
        const float v = r * (g[j] - __bfloat162float(y[j]) * dot);
        out[j] = acc ? out[j] + v : v;
    }
}

__global__ void k_to_bf16(const float* __restrict__ x, bf16* __restrict__ y, size_t n) {
    for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < n; i += size_t(gridDim.x) * blockDim.x)
        y[i] = __float2bfloat16(x[i]);
}

__global__ void k_bias_gelu(bf16* __restrict__ z, const float* __restrict__ b, bf16* __restrict__ u, size_t n,
                            int F) {
    for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < n; i += size_t(gridDim.x) * blockDim.x) {
        const float v = __bfloat162float(z[i]) + b[i % F];
        z[i] = __float2bfloat16(v);
        u[i] = __float2bfloat16(gelu_tanh(v));
    }
}

__global__ void k_gelu_bwd(const float* __restrict__ du, const bf16* __restrict__ z, bf16* __restrict__ dz,
                           size_t n) {
    for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < n; i += size_t(gridDim.x) * blockDim.x)
        dz[i] = __float2bfloat16(du[i] * gelu_tanh_grad(__bfloat162float(z[i])));
}

__global__ void k_add_bias(float* __restrict__ h, const float* __restrict__ b, size_t n, int D) {
    for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < n; i += size_t(gridDim.x) * blockDim.x)
        h[i] += b[i % D];
}

template <class T>
__global__ void k_colsum(const T* __restrict__ x, int M, int N, int ld, float* __restrict__ out, int rows) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= N) return;
    const int r0 = blockIdx.y * rows, r1 = min(M, r0 + rows);
    float s = 0.f;
    for (int r = r0; r < r1; ++r) s += to_f(x[size_t(r) * ld + col]);
    atomicAdd(out + col, s);
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

__global__ void __launch_bounds__(256) k_ce_rows(const float* __restrict__ logits, const int* __restrict__ y,
                                                 bf16* __restrict__ dlogits, float* __restrict__ loss_sum, int V,
                                                 float inv_n) {
    __shared__ float sh[32];
    const size_t row = blockIdx.x;
    const float* l = logits + row * V;
    bf16* d = dlogits + row * V;
    const int target = y[row];
    if (target < 0) {
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
    for (int j = threadIdx.x; j < V; j += blockDim.x) {
        float p = __expf(l[j] - m) * inv;
        if (j == target) p -= 1.f;
        d[j] = __float2bfloat16(p * inv_n);
    }
    if (threadIdx.x == 0) atomicAdd(loss_sum, (logf(s) + m - l[target]) * inv_n);
}

__global__ void k_sumsq(const float* __restrict__ x, size_t n, float* __restrict__ out) {
    float s = 0.f;
    for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < n; i += size_t(gridDim.x) * blockDim.x)
        s += x[i] * x[i];
    s = warp_sum(s);
    if ((threadIdx.x & 31) == 0) atomicAdd(out, s);
}

__global__ void k_adamw(float* __restrict__ p, const float* __restrict__ g, float* __restrict__ m,
                        float* __restrict__ v, const uint8_t* __restrict__ decay, size_t n, float lr, float b1,
                        float b2, float eps, float wd, float bc1, float bc2, const float* __restrict__ gsq,
                        float clip) {
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

void rmsnorm_bwd(const float* dxh, const bf16* xh, const float* rstd, float* dh, int M, int D, bool accumulate,
                 cudaStream_t s) {
    k_rmsnorm_bwd<<<(M + 7) / 8, 256, 0, s>>>(dxh, xh, rstd, dh, M, D, accumulate ? 1 : 0);
    LBC_CUDA(cudaGetLastError());
}

void to_bf16(const float* x, bf16* y, size_t n, cudaStream_t s) {
    k_to_bf16<<<blocks_for(n), 256, 0, s>>>(x, y, n);
    LBC_CUDA(cudaGetLastError());
}

void bias_gelu_fwd(bf16* z, const float* b, bf16* u, int M, int F, cudaStream_t s) {
    const size_t n = size_t(M) * F;
    k_bias_gelu<<<blocks_for(n), 256, 0, s>>>(z, b, u, n, F);
    LBC_CUDA(cudaGetLastError());
}

void gelu_bwd(const float* du, const bf16* z, bf16* dz, int M, int F, cudaStream_t s) {
    const size_t n = size_t(M) * F;
    k_gelu_bwd<<<blocks_for(n), 256, 0, s>>>(du, z, dz, n);
    LBC_CUDA(cudaGetLastError());
}

void add_bias(float* h, const float* b, int M, int D, cudaStream_t s) {
    const size_t n = size_t(M) * D;
    k_add_bias<<<blocks_for(n), 256, 0, s>>>(h, b, n, D);
    LBC_CUDA(cudaGetLastError());
}

void colsum(const float* x, int M, int N, int ld, float* out, cudaStream_t s) {
    const int rows = 256;
    k_colsum<float><<<dim3((N + 255) / 256, (M + rows - 1) / rows), 256, 0, s>>>(x, M, N, ld, out, rows);
    LBC_CUDA(cudaGetLastError());
}

void colsum(const bf16* x, int M, int N, int ld, float* out, cudaStream_t s) {
    const int rows = 256;
    k_colsum<bf16><<<dim3((N + 255) / 256, (M + rows - 1) / rows), 256, 0, s>>>(x, M, N, ld, out, rows);
    LBC_CUDA(cudaGetLastError());
}

void embed_gather(const int* ids, const bf16* E, bf16* x, int M, int E0, cudaStream_t s) {
    const size_t n = size_t(M) * E0;
    k_embed_gather<<<blocks_for(n), 256, 0, s>>>(ids, E, x, n, E0);
    LBC_CUDA(cudaGetLastError());
}

void embed_scatter(const int* ids, const float* dx, float* dE, int M, int E0, cudaStream_t s) {
    const size_t n = size_t(M) * E0;
    k_embed_scatter<<<blocks_for(n), 256, 0, s>>>(ids, dx, dE, n, E0);
    LBC_CUDA(cudaGetLastError());
}

void ce_rows(const float* logits, const int* y, bf16* dlogits, float* loss_sum, int rows, int V, float inv_n,
             cudaStream_t s) {
    k_ce_rows<<<rows, 256, 0, s>>>(logits, y, dlogits, loss_sum, V, inv_n);
    LBC_CUDA(cudaGetLastError());
}

void sumsq(const float* x, size_t n, float* out, cudaStream_t s) {
    k_sumsq<<<blocks_for(n), 256, 0, s>>>(x, n, out);
    LBC_CUDA(cudaGetLastError());
}

void adamw(float* p, const float* g, float* m, float* v, const uint8_t* decay, size_t n, float lr, float beta1,
           float beta2, float eps, float wd, float bc1, float bc2, const float* gsq, float clip, cudaStream_t s) {
    k_adamw<<<blocks_for(n), 256, 0, s>>>(p, g, m, v, decay, n, lr, beta1, beta2, eps, wd, bc1, bc2, gsq, clip);
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
