#include "gemm.hpp"

#include <cstdio>
#include <cstdlib>

namespace lbc {

Gemm::Gemm() {
    LBC_CUBLAS(cublasLtCreate(&lt_));
    LBC_CUDA(cudaMalloc(&workspace_, workspace_bytes_));
}

Gemm::~Gemm() {
    for (auto& [key, p] : plans_) {
        cublasLtMatrixLayoutDestroy(p.a);
        cublasLtMatrixLayoutDestroy(p.b);
        cublasLtMatrixLayoutDestroy(p.d);
        cublasLtMatmulDescDestroy(p.op);
    }
    cudaFree(workspace_);
    cublasLtDestroy(lt_);
}

// cuBLASLt is column-major: a row-major X (r x c, stride ld) is the column-major X^T (c x r, ld). So row-major
// D = op(A) op(B) is computed as column-major D^T = op(B)^T op(A)^T: the first operand is the stored B, the second
// the stored A, and D^T is N x M.
Gemm::Plan& Gemm::plan(bool tA, bool tB, int M, int N, int K, int lda, int ldb, int ldd, Out out) {
    const Key key{tA, tB, M, N, K, lda, ldb, ldd, int(out)};
    auto it = plans_.find(key);
    if (it != plans_.end()) return it->second;

    Plan p{};
    LBC_CUBLAS(cublasLtMatmulDescCreate(&p.op, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    const cublasOperation_t opa = tB ? CUBLAS_OP_T : CUBLAS_OP_N;
    const cublasOperation_t opb = tA ? CUBLAS_OP_T : CUBLAS_OP_N;
    LBC_CUBLAS(cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_TRANSA, &opa, sizeof(opa)));
    LBC_CUBLAS(cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_TRANSB, &opb, sizeof(opb)));
    // stored B: K x N row-major (column-major N x K), or N x K row-major (column-major K x N) when tB
    LBC_CUBLAS(cublasLtMatrixLayoutCreate(&p.a, CUDA_R_16BF, tB ? K : N, tB ? N : K, ldb));
    // stored A: M x K row-major (column-major K x M), or K x M row-major (column-major M x K) when tA
    LBC_CUBLAS(cublasLtMatrixLayoutCreate(&p.b, CUDA_R_16BF, tA ? M : K, tA ? K : M, lda));
    LBC_CUBLAS(cublasLtMatrixLayoutCreate(&p.d, out == Out::F32 ? CUDA_R_32F : CUDA_R_16BF, N, M, ldd));

    cublasLtMatmulPreference_t pref;
    LBC_CUBLAS(cublasLtMatmulPreferenceCreate(&pref));
    LBC_CUBLAS(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace_bytes_,
                                                    sizeof(workspace_bytes_)));
    cublasLtMatmulHeuristicResult_t res{};
    int found = 0;
    LBC_CUBLAS(cublasLtMatmulAlgoGetHeuristic(lt_, p.op, p.a, p.b, p.d, p.d, pref, 1, &res, &found));
    cublasLtMatmulPreferenceDestroy(pref);
    if (found == 0) {
        std::fprintf(stderr, "no cuBLASLt algorithm for M=%d N=%d K=%d tA=%d tB=%d\n", M, N, K, tA, tB);
        std::abort();
    }
    p.algo = res.algo;
    return plans_.emplace(key, p).first->second;
}

void Gemm::run(cudaStream_t stream, bool tA, bool tB, int M, int N, int K, const bf16* A, int lda, const bf16* B,
               int ldb, void* D, int ldd, Out out, float beta, float alpha) {
    Plan& p = plan(tA, tB, M, N, K, lda, ldb, ldd, out);
    LBC_CUBLAS(cublasLtMatmul(lt_, p.op, &alpha, B, p.a, A, p.b, &beta, D, p.d, D, p.d, &p.algo, workspace_,
                              workspace_bytes_, stream));
}

}  // namespace lbc
