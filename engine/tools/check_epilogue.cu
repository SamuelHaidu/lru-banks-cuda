// Checks the cuBLASLt epilogues against host references, in our row-major convention: which GELU (tanh form or erf),
// the bias / bias-gradient orientation, and the aux layout.
//
//   check_epilogue        (prints the max errors; exit status 1 if an epilogue is off)

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "../src/gemm.hpp"

using namespace lbc;

namespace {

double gelu_tanh(double z) { return 0.5 * z * (1 + std::tanh(0.7978845608028654 * (z + 0.044715 * z * z * z))); }

double gelu_erf(double z) { return 0.5 * z * (1 + std::erf(z / std::sqrt(2.0))); }

double gelu_tanh_grad(double z) {
    const double t = std::tanh(0.7978845608028654 * (z + 0.044715 * z * z * z));
    return 0.5 * (1 + t) + 0.5 * z * (1 - t * t) * 0.7978845608028654 * (1 + 3 * 0.044715 * z * z);
}

double bf(float x) { return double(__bfloat162float(__float2bfloat16(x))); }

template <class T>
T* dev(const std::vector<T>& h) {
    T* d;
    LBC_CUDA(cudaMalloc(&d, h.size() * sizeof(T)));
    LBC_CUDA(cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice));
    return d;
}

template <class T>
std::vector<T> host(const T* d, size_t n) {
    std::vector<T> h(n);
    LBC_CUDA(cudaMemcpy(h.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost));
    return h;
}

}  // namespace

int main() {
    const int M = 256, N = 128, K = 64;
    std::mt19937 rng(0);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<bf16> A(size_t(M) * K), B(size_t(K) * N);
    std::vector<float> bias(N);
    std::vector<bf16> biasb(N);
    for (auto& x : A) { x = __float2bfloat16(nd(rng)); }
    for (auto& x : B) { x = __float2bfloat16(0.3f * nd(rng)); }
    for (int j = 0; j < N; ++j) {
        biasb[j] = __float2bfloat16(nd(rng));
        bias[j] = __bfloat162float(biasb[j]);  // the bf16 bias of the bf16-output epilogues, exactly
    }
    // host AB (row-major)
    std::vector<double> AB(size_t(M) * N, 0.0);
    for (int i = 0; i < M; ++i) {
        for (int k = 0; k < K; ++k) {
            for (int j = 0; j < N; ++j) {
                AB[size_t(i) * N + j] +=
                    double(__bfloat162float(A[size_t(i) * K + k])) * __bfloat162float(B[size_t(k) * N + j]);
            }
        }
    }

    Gemm g;
    bf16 *dA = dev(A), *dB = dev(B);
    float* dbias = dev(bias);
    bf16* dbiasb = dev(biasb);
    bf16 *dD, *daux;
    LBC_CUDA(cudaMalloc(&dD, size_t(M) * N * sizeof(bf16)));
    LBC_CUDA(cudaMalloc(&daux, size_t(M) * N * sizeof(bf16)));
    bool ok = true;

    // 1. GeluAuxBias: aux = AB + bias, D = gelu(aux)
    Epilogue e1{Epilogue::GeluAuxBias, dbiasb, nullptr, daux, N};
    g.run(nullptr, false, false, M, N, K, dA, K, dB, N, dD, N, Out::BF16, 0.f, 1.f, e1);
    LBC_CUDA(cudaDeviceSynchronize());
    const auto D1 = host(dD, size_t(M) * N), aux = host(daux, size_t(M) * N);
    double e_aux = 0, e_tanh = 0, e_erf = 0;
    for (size_t i = 0; i < D1.size(); ++i) {
        const double z = AB[i] + bias[i % N];
        e_aux = std::max(e_aux, std::abs(double(__bfloat162float(aux[i])) - z) / (1 + std::abs(z)));
        e_tanh = std::max(e_tanh, std::abs(double(__bfloat162float(D1[i])) - gelu_tanh(z)));
        e_erf = std::max(e_erf, std::abs(double(__bfloat162float(D1[i])) - gelu_erf(z)));
    }
    std::printf("GeluAuxBias (bf16 out): aux vs AB + bias (rel) %.2e; D vs gelu_tanh %.2e, vs gelu_erf %.2e (abs)\n",
                e_aux, e_tanh, e_erf);
    ok &= e_aux < 2e-2 && e_tanh < 6e-2;
    // which GELU: an fp32 output (fp32 bias) resolves the tanh / erf difference (up to ~5e-4)
    {
        float* dD32g;
        LBC_CUDA(cudaMalloc(&dD32g, size_t(M) * N * sizeof(float)));
        Epilogue eg{Epilogue::GeluBias, dbias, nullptr, nullptr, 0};
        g.run(nullptr, false, false, M, N, K, dA, K, dB, N, dD32g, N, Out::F32, 0.f, 1.f, eg);
        LBC_CUDA(cudaDeviceSynchronize());
        const auto Dg = host(dD32g, size_t(M) * N);
        double et = 0, ee = 0;
        for (size_t i = 0; i < Dg.size(); ++i) {
            const double z = AB[i] + bias[i % N];
            et = std::max(et, std::abs(Dg[i] - gelu_tanh(z)));
            ee = std::max(ee, std::abs(Dg[i] - gelu_erf(z)));
        }
        std::printf("GeluAuxBias (fp32 out): D vs gelu_tanh %.2e, vs gelu_erf %.2e -> cuBLASLt GELU is the %s form\n",
                    et, ee, et < ee ? "tanh" : "erf");
        ok &= et < ee;
    }

    // 2. DGeluBgrad: D = AB * gelu'(aux), bias_grad[j] = sum_i D[i][j]
    bf16* dbg;
    LBC_CUDA(cudaMalloc(&dbg, N * sizeof(bf16)));
    Epilogue e2{Epilogue::DGeluBgrad, nullptr, dbg, daux, N};
    g.run(nullptr, false, false, M, N, K, dA, K, dB, N, dD, N, Out::BF16, 0.f, 1.f, e2);
    LBC_CUDA(cudaDeviceSynchronize());
    const auto D2 = host(dD, size_t(M) * N);
    const auto bg = host(dbg, N);
    double e_d = 0, e_bg = 0;
    std::vector<double> colsum(N, 0.0);
    for (size_t i = 0; i < D2.size(); ++i) {
        const double ref = AB[i] * gelu_tanh_grad(double(__bfloat162float(aux[i])));
        e_d = std::max(e_d, std::abs(double(__bfloat162float(D2[i])) - ref) / (1 + std::abs(ref)));
        colsum[i % N] += double(__bfloat162float(D2[i]));
    }
    for (int j = 0; j < N; ++j) {
        e_bg = std::max(e_bg, std::abs(double(__bfloat162float(bg[j])) - colsum[j]) / (1 + std::abs(colsum[j])));
    }
    std::printf("DGeluBgrad: D vs AB * gelu_tanh'(aux) (rel) %.2e; bias grad vs column sums of D (rel) %.2e\n", e_d,
                e_bg);
    ok &= e_d < 2e-2 && e_bg < 2e-2;

    // 3. Bias on an fp32 output accumulated into (beta 1): D = D + AB + bias
    std::vector<float> D0(size_t(M) * N);
    for (auto& x : D0) { x = nd(rng); }
    float* dD32 = dev(D0);
    Epilogue e3{Epilogue::Bias, dbias, nullptr, nullptr, 0};
    g.run(nullptr, false, false, M, N, K, dA, K, dB, N, dD32, N, Out::F32, 1.f, 1.f, e3);
    LBC_CUDA(cudaDeviceSynchronize());
    const auto D3 = host(dD32, size_t(M) * N);
    double e_b = 0;
    for (size_t i = 0; i < D3.size(); ++i) {
        const double ref = D0[i] + AB[i] + bias[i % N];
        e_b = std::max(e_b, std::abs(D3[i] - ref) / (1 + std::abs(ref)));
    }
    std::printf("Bias (fp32, beta 1): D vs D0 + AB + bias (rel) %.2e\n", e_b);
    ok &= e_b < 1e-3;
    (void)bf;
    std::printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
