// Microbenchmark for P22 / optimization 6: the FFN GEMMs in FP8 (E4M3, per-tensor scales, fp32 accumulation) against
// bf16, on cuBLASLt, plus the activation quantization pass FP8 would add (amax + cast, the cheapest form).
//
//   bench_fp8
//
// FP8 GEMMs on sm_89 need the "TN" layout (K contiguous in both operands): row-major D[M x N] = X[M x K] W[K x N]
// is run as column-major D^T = (W^T) X^T with W^T stored row-major N x K (a transposed weight copy) and X as is.

#include <cstdio>
#include <string>
#include <vector>

#include <cublasLt.h>
#include <cuda_fp8.h>

#include "../src/gemm.hpp"
#include "lbc/bench.hpp"

using namespace lbc;

namespace {

struct Fp8Gemm {
    cublasLtMatmulDesc_t op;
    cublasLtMatrixLayout_t a, b, d;
    cublasLtMatmulAlgo_t algo;
};

// D (bf16, row-major M x N) = X (fp8, row-major M x K) W, with Wt = W^T (fp8, row-major N x K)
Fp8Gemm make_fp8(cublasLtHandle_t lt, int M, int N, int K, const float* sa, const float* sb, size_t ws) {
    Fp8Gemm g{};
    LBC_CUBLAS(cublasLtMatmulDescCreate(&g.op, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    const cublasOperation_t T = CUBLAS_OP_T, Nn = CUBLAS_OP_N;
    LBC_CUBLAS(cublasLtMatmulDescSetAttribute(g.op, CUBLASLT_MATMUL_DESC_TRANSA, &T, sizeof(T)));
    LBC_CUBLAS(cublasLtMatmulDescSetAttribute(g.op, CUBLASLT_MATMUL_DESC_TRANSB, &Nn, sizeof(Nn)));
    LBC_CUBLAS(cublasLtMatmulDescSetAttribute(g.op, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &sa, sizeof(sa)));
    LBC_CUBLAS(cublasLtMatmulDescSetAttribute(g.op, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &sb, sizeof(sb)));
    LBC_CUBLAS(cublasLtMatrixLayoutCreate(&g.a, CUDA_R_8F_E4M3, K, N, K));  // Wt: column-major K x N (op T: N x K)
    LBC_CUBLAS(cublasLtMatrixLayoutCreate(&g.b, CUDA_R_8F_E4M3, K, M, K));  // X: column-major K x M
    LBC_CUBLAS(cublasLtMatrixLayoutCreate(&g.d, CUDA_R_16BF, N, M, N));
    cublasLtMatmulPreference_t pref;
    LBC_CUBLAS(cublasLtMatmulPreferenceCreate(&pref));
    LBC_CUBLAS(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws, sizeof(ws)));
    cublasLtMatmulHeuristicResult_t res{};
    int found = 0;
    LBC_CUBLAS(cublasLtMatmulAlgoGetHeuristic(lt, g.op, g.a, g.b, g.d, g.d, pref, 1, &res, &found));
    if (!found) fail("fp8 heuristic", "no algorithm", __FILE__, __LINE__);
    g.algo = res.algo;
    cublasLtMatmulPreferenceDestroy(pref);
    return g;
}

__global__ void k_amax(const bf16* x, size_t n, float* amax) {
    float m = 0.f;
    for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < n; i += size_t(gridDim.x) * blockDim.x)
        m = fmaxf(m, fabsf(__bfloat162float(x[i])));
    for (int o = 16; o; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
    if ((threadIdx.x & 31) == 0) atomicMax(reinterpret_cast<int*>(amax), __float_as_int(m));  // m >= 0
}

__global__ void k_cast_fp8(const bf16* x, __nv_fp8_e4m3* y, size_t n, const float* amax) {
    const float scale = 448.f / fmaxf(*amax, 1e-12f);
    for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < n; i += size_t(gridDim.x) * blockDim.x)
        y[i] = __nv_fp8_e4m3(__bfloat162float(x[i]) * scale);
}

}  // namespace

int main() {
    cudaStream_t s;
    LBC_CUDA(cudaStreamCreate(&s));
    cublasLtHandle_t lt;
    LBC_CUBLAS(cublasLtCreate(&lt));
    const size_t ws_bytes = size_t(32) << 20;
    void* ws;
    LBC_CUDA(cudaMalloc(&ws, ws_bytes));
    float* one;
    LBC_CUDA(cudaMalloc(&one, sizeof(float)));
    const float h1 = 1.f;
    LBC_CUDA(cudaMemcpy(one, &h1, sizeof(float), cudaMemcpyHostToDevice));
    Gemm bfg;
    L2Flusher flush;

    struct Shape {
        const char* name;
        int M, N, K;
    } shapes[] = {{"FFN up   (32768 x 512 -> 2048)", 32768, 2048, 512}, {"FFN down (32768 x 2048 -> 512)", 32768, 512, 2048}};
    std::printf("%-34s %10s %10s %8s %12s\n", "shape", "bf16 ms", "fp8 ms", "speedup", "quantize ms");
    for (const auto& sh : shapes) {
        const size_t nx = size_t(sh.M) * sh.K, nw = size_t(sh.K) * sh.N, nd = size_t(sh.M) * sh.N;
        bf16 *X, *W, *Wt, *D;
        __nv_fp8_e4m3 *X8, *Wt8;
        float* amax;
        LBC_CUDA(cudaMalloc(&X, nx * 2));
        LBC_CUDA(cudaMalloc(&W, nw * 2));
        LBC_CUDA(cudaMalloc(&Wt, nw * 2));
        LBC_CUDA(cudaMalloc(&D, nd * 2));
        LBC_CUDA(cudaMalloc(&X8, nx));
        LBC_CUDA(cudaMalloc(&Wt8, nw));
        LBC_CUDA(cudaMalloc(&amax, sizeof(float)));
        LBC_CUDA(cudaMemset(X, 0x3c, nx * 2));  // small bf16 values; content does not change the timing
        LBC_CUDA(cudaMemset(W, 0x3c, nw * 2));
        LBC_CUDA(cudaMemset(Wt, 0x3c, nw * 2));
        LBC_CUDA(cudaMemset(X8, 0x30, nx));
        LBC_CUDA(cudaMemset(Wt8, 0x30, nw));
        const double flops = 2.0 * sh.M * sh.N * sh.K;
        // bf16, the layout the model uses (W row-major K x N) and the TN layout (W^T) for a like-for-like comparison
        const Stats sb = time_it([&](cudaStream_t st) {
            bfg.run(st, false, false, sh.M, sh.N, sh.K, X, sh.K, W, sh.N, D, sh.N, Out::BF16);
        }, s, 5, 30, &flush);
        const Stats sbt = time_it([&](cudaStream_t st) {
            bfg.run(st, false, true, sh.M, sh.N, sh.K, X, sh.K, Wt, sh.K, D, sh.N, Out::BF16);
        }, s, 5, 30, &flush);
        Fp8Gemm g = make_fp8(lt, sh.M, sh.N, sh.K, one, one, ws_bytes);
        const float alpha = 1.f, beta = 0.f;
        const Stats s8 = time_it([&](cudaStream_t st) {
            LBC_CUBLAS(cublasLtMatmul(lt, g.op, &alpha, Wt8, g.a, X8, g.b, &beta, D, g.d, D, g.d, &g.algo, ws,
                                      ws_bytes, st));
        }, s, 5, 30, &flush);
        const Stats sq = time_it([&](cudaStream_t st) {
            LBC_CUDA(cudaMemsetAsync(amax, 0, sizeof(float), st));
            k_amax<<<1024, 256, 0, st>>>(X, nx, amax);
            k_cast_fp8<<<1024, 256, 0, st>>>(X, X8, nx, amax);
        }, s, 5, 30, &flush);
        report((std::string(sh.name) + " bf16").c_str(), sb, flops);
        report((std::string(sh.name) + " bf16 TN").c_str(), sbt, flops);
        report((std::string(sh.name) + " fp8 TN").c_str(), s8, flops);
        report((std::string(sh.name) + " quantize X").c_str(), sq, 0.0, nx * 3.0);
        std::printf("  -> fp8 speedup over bf16 %.2fx; with X quantized every call %.2fx\n",
                    sb.median_ms / s8.median_ms, sb.median_ms / (s8.median_ms + sq.median_ms));
    }
    return 0;
}
