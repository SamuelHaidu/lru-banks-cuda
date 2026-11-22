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
        LBC_CUBLAS_CLEANUP(cublasLtMatrixLayoutDestroy(p.a));
        LBC_CUBLAS_CLEANUP(cublasLtMatrixLayoutDestroy(p.b));
        LBC_CUBLAS_CLEANUP(cublasLtMatrixLayoutDestroy(p.d));
        LBC_CUBLAS_CLEANUP(cublasLtMatmulDescDestroy(p.op));
    }
    LBC_CUDA_CLEANUP(cudaFree(workspace_));
    LBC_CUBLAS_CLEANUP(cublasLtDestroy(lt_));
}

// cuBLASLt is column-major: a row-major X (r x c, stride ld) is the column-major X^T (c x r, ld). So row-major
// D = op(A) op(B) is computed as column-major D^T = op(B)^T op(A)^T: the first operand is the stored B, the second
// the stored A, and D^T is N x M.
Gemm::Plan& Gemm::plan(bool tA, bool tB, int M, int N, int K, int lda, int ldb, int ldd, Out out, int epi, int ld_aux) {
    const Key key{tA, tB, M, N, K, lda, ldb, ldd, int(out), epi, ld_aux};
    auto it = plans_.find(key);
    if (it != plans_.end()) { return it->second; }

    Plan p{};
    LBC_CUBLAS(cublasLtMatmulDescCreate(&p.op, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    const cublasOperation_t opa = tB ? CUBLAS_OP_T : CUBLAS_OP_N;
    const cublasOperation_t opb = tA ? CUBLAS_OP_T : CUBLAS_OP_N;
    LBC_CUBLAS(cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_TRANSA, &opa, sizeof(opa)));
    LBC_CUBLAS(cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_TRANSB, &opb, sizeof(opb)));
    if (epi != Epilogue::None) {
        // column-major D^T: the row-major bias over D's columns is cuBLASLt's bias over D^T's rows
        cublasLtEpilogue_t e = CUBLASLT_EPILOGUE_DGELU_BGRAD;
        switch (epi) {
            case Epilogue::Bias: e = CUBLASLT_EPILOGUE_BIAS; break;
            case Epilogue::GeluAuxBias: e = CUBLASLT_EPILOGUE_GELU_AUX_BIAS; break;
            case Epilogue::GeluBias: e = CUBLASLT_EPILOGUE_GELU_BIAS; break;
            default: break;
        }
        LBC_CUBLAS(cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_EPILOGUE, &e, sizeof(e)));
        const cudaDataType_t bias_type = out == Out::F32 ? CUDA_R_32F : CUDA_R_16BF;
        LBC_CUBLAS(
            cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE, &bias_type, sizeof(bias_type)));
        if (epi == Epilogue::GeluAuxBias || epi == Epilogue::DGeluBgrad) {
            const int64_t ld = ld_aux;
            LBC_CUBLAS(cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_LD, &ld, sizeof(ld)));
            const cudaDataType_t aux_type = CUDA_R_16BF;
            LBC_CUBLAS(cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_DATA_TYPE, &aux_type,
                                                      sizeof(aux_type)));
        }
    }
    // stored B: K x N row-major (column-major N x K), or N x K row-major (column-major K x N) when tB
    LBC_CUBLAS(cublasLtMatrixLayoutCreate(&p.a, CUDA_R_16BF, tB ? K : N, tB ? N : K, ldb));
    // stored A: M x K row-major (column-major K x M), or K x M row-major (column-major M x K) when tA
    LBC_CUBLAS(cublasLtMatrixLayoutCreate(&p.b, CUDA_R_16BF, tA ? M : K, tA ? K : M, lda));
    LBC_CUBLAS(cublasLtMatrixLayoutCreate(&p.d, out == Out::F32 ? CUDA_R_32F : CUDA_R_16BF, N, M, ldd));

    cublasLtMatmulPreference_t pref = nullptr;
    LBC_CUBLAS(cublasLtMatmulPreferenceCreate(&pref));
    LBC_CUBLAS(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace_bytes_,
                                                    sizeof(workspace_bytes_)));
    cublasLtMatmulHeuristicResult_t res{};
    int found = 0;
    cublasStatus_t const st = cublasLtMatmulAlgoGetHeuristic(lt_, p.op, p.a, p.b, p.d, p.d, pref, 1, &res, &found);
    LBC_CUBLAS(st);
    LBC_CUBLAS_CLEANUP(cublasLtMatmulPreferenceDestroy(pref));
    if (found == 0) {
        std::fprintf(stderr, "no cuBLASLt algorithm for M=%d N=%d K=%d tA=%d tB=%d\n", M, N, K, static_cast<int>(tA),
                     static_cast<int>(tB));
        std::abort();
    }
    p.algo = res.algo;
    return plans_.emplace(key, p).first->second;
}

void Gemm::run(cudaStream_t stream,
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
               float beta,
               float alpha,
               const Epilogue& epi) {
    Plan const& p = plan(tA, tB, M, N, K, lda, ldb, ldd, out, int(epi.kind), epi.ld_aux);
    if (epi.kind == Epilogue::Bias || epi.kind == Epilogue::GeluAuxBias || epi.kind == Epilogue::GeluBias) {
        set_attribute(p.op, CUBLASLT_MATMUL_DESC_BIAS_POINTER, epi.bias);
    }
    if (epi.kind == Epilogue::DGeluBgrad) { set_attribute(p.op, CUBLASLT_MATMUL_DESC_BIAS_POINTER, epi.bias_grad); }
    if (epi.kind == Epilogue::GeluAuxBias || epi.kind == Epilogue::DGeluBgrad) {
        set_attribute(p.op, CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER, epi.aux);
    }
    LBC_CUBLAS(cublasLtMatmul(lt_, p.op, &alpha, B, p.a, A, p.b, &beta, D, p.d, D, p.d, &p.algo, workspace_,
                              workspace_bytes_, stream));
}

}  // namespace lbc
