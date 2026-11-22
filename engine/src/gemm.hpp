#pragma once
// Row-major GEMMs on cuBLASLt: bf16 operands, fp32 accumulation, fp32 or bf16 output. One plan (descriptors and the
// heuristic's algorithm) per shape, built on first use and reused: every shape of the step is static.

#include <map>
#include <tuple>

#include <cublasLt.h>
#include <cuda_runtime.h>

#include "common.hpp"
#include "lbc/check.hpp"

#define LBC_CUBLAS(call)                                                              \
    do {                                                                              \
        const cublasStatus_t lbc_st_ = (call);                                        \
        if (lbc_st_ != CUBLAS_STATUS_SUCCESS)                                         \
            ::lbc::fail(#call, cublasLtGetStatusString(lbc_st_), __FILE__, __LINE__); \
    } while (0)

namespace lbc {

enum class Out { F32, BF16 };

// Fused epilogues (cuBLASLt), in row-major terms: the bias is a vector over D's columns, in D's type (cuBLASLt
// refuses an fp32 bias with a bf16 output for the GELU epilogues on sm_89).
//   Bias:         D = AB (+ beta D) + bias
//   GeluAuxBias:  aux = AB + bias (bf16, kept for the backward), D = gelu(aux) (cuBLASLt's GELU is the tanh form:
//                 tools/check_epilogue)
//   DGeluBgrad:   D = AB * gelu'(aux), bias_grad = column sums of D (written)
struct Epilogue {
    enum Kind { None, Bias, GeluAuxBias, DGeluBgrad, GeluBias } kind = None;  // GeluBias: D = gelu(AB + bias), no aux

    const void* bias = nullptr;  // Bias, GeluAuxBias
    void* bias_grad = nullptr;   // DGeluBgrad
    bf16* aux = nullptr;         // GeluAuxBias (written), DGeluBgrad (read): same shape as D
    int ld_aux = 0;
};

class Gemm {
  public:
    Gemm();
    ~Gemm();
    Gemm(const Gemm&) = delete;
    Gemm& operator=(const Gemm&) = delete;

    // D[M x N] = alpha * op(A)[M x K] * op(B)[K x N] + beta * D, all row-major; op(X) = X^T when tX. lda, ldb, ldd
    // are the row strides of the stored matrices (A stored M x K, or K x M when tA).
    void run(cudaStream_t stream,
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
             float beta = 0.f,
             float alpha = 1.f,
             const Epilogue& epi = {});

  private:
    struct Plan {
        cublasLtMatmulDesc_t op;
        cublasLtMatrixLayout_t a, b, d;
        cublasLtMatmulAlgo_t algo;
    };

    using Key = std::tuple<bool, bool, int, int, int, int, int, int, int, int, int>;
    Plan& plan(bool tA, bool tB, int M, int N, int K, int lda, int ldb, int ldd, Out out, int epi, int ld_aux);

    std::map<Key, Plan> plans_;
    cublasLtHandle_t lt_ = nullptr;
    void* workspace_ = nullptr;
    size_t workspace_bytes_ = size_t(32) << 20;
};

}  // namespace lbc
