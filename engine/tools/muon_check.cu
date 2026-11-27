// Checks newton_schulz (Muon's orthogonalization, gemm.hpp) against a float64 host reference of the same iteration on
// random matrices of the model's block shapes (wide, tall, square), and times it. The result's spread around the
// orthogonal factor is summarized by the diagonal of O^T O (short side): close to 1 for an orthogonalized update.
//
//   muon_check [--steps 5]   (prints the relative errors and times; exit status 1 if one is off)

#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "../src/gemm.hpp"

using namespace lbc;

namespace {

// X <- a X + (b A + c A^2) X with A = X X^T (rows <= cols) or X <- a X + X (b A + c A^2) with A = X^T X, in float64
void reference(std::vector<double>& x, int r, int c, int steps) {
    const double a = 3.4445, b = -4.7750, cc = 2.0315;
    const bool wide = r <= c;
    const int k = wide ? r : c;
    std::vector<double> A(size_t(k) * k), B(size_t(k) * k), y(x.size());
    for (int it = 0; it < steps; ++it) {
        for (int i = 0; i < k; ++i) {
            for (int j = 0; j < k; ++j) {
                double s = 0;
                if (wide) {
                    for (int t = 0; t < c; ++t) { s += x[size_t(i) * c + t] * x[size_t(j) * c + t]; }
                } else {
                    for (int t = 0; t < r; ++t) { s += x[size_t(t) * c + i] * x[size_t(t) * c + j]; }
                }
                A[size_t(i) * k + j] = s;
            }
        }
        for (int i = 0; i < k; ++i) {
            for (int j = 0; j < k; ++j) {
                double s = 0;
                for (int t = 0; t < k; ++t) { s += A[size_t(i) * k + t] * A[size_t(t) * k + j]; }
                B[size_t(i) * k + j] = b * A[size_t(i) * k + j] + cc * s;
            }
        }
        for (int i = 0; i < r; ++i) {
            for (int j = 0; j < c; ++j) {
                double s = 0;
                if (wide) {
                    for (int t = 0; t < r; ++t) { s += B[size_t(i) * k + t] * x[size_t(t) * c + j]; }
                } else {
                    for (int t = 0; t < c; ++t) { s += x[size_t(i) * c + t] * B[size_t(t) * k + j]; }
                }
                y[size_t(i) * c + j] = a * x[size_t(i) * c + j] + s;
            }
        }
        x.swap(y);
    }
}

}  // namespace

int main(int argc, char** argv) {
    int steps = 5;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--steps") == 0 && i + 1 < argc) { steps = std::stoi(argv[++i]); }
    }
    const int shapes[][2] = {{512, 2048}, {2048, 512}, {512, 512}, {512, 1024}};
    Gemm g;
    cudaStream_t s = nullptr;
    LBC_CUDA(cudaStreamCreate(&s));
    std::mt19937_64 rng(7);
    std::normal_distribution<float> nd(0.f, 1.f);
    bool ok = true;
    for (const auto& sh : shapes) {
        const int r = sh[0], c = sh[1], k = std::min(r, c);
        const size_t n = size_t(r) * c;
        std::vector<float> xf(n);
        double ss = 0;
        for (float& v : xf) {
            v = nd(rng);
            ss += double(v) * v;
        }
        std::vector<bf16> xb(n);
        std::vector<double> xd(n);
        for (size_t i = 0; i < n; ++i) {
            xb[i] = __float2bfloat16(float(xf[i] / std::sqrt(ss)));
            xd[i] = double(__bfloat162float(xb[i]));
        }
        bf16 *x = nullptr, *xn = nullptr, *wa = nullptr, *wb = nullptr;
        LBC_CUDA(cudaMalloc(&x, n * sizeof(bf16)));
        LBC_CUDA(cudaMalloc(&xn, n * sizeof(bf16)));
        LBC_CUDA(cudaMalloc(&wa, size_t(k) * k * sizeof(bf16)));
        LBC_CUDA(cudaMalloc(&wb, size_t(k) * k * sizeof(bf16)));
        LBC_CUDA(cudaMemcpy(x, xb.data(), n * sizeof(bf16), cudaMemcpyHostToDevice));
        const bf16* o = newton_schulz(g, x, xn, wa, wb, r, c, steps, s);
        std::vector<bf16> ob(n);
        LBC_CUDA(cudaMemcpyAsync(ob.data(), o, n * sizeof(bf16), cudaMemcpyDeviceToHost, s));
        LBC_CUDA(cudaStreamSynchronize(s));
        // timing: 50 calls (each on the previous result: near-orthogonal, so the values stay bounded)
        cudaEvent_t e0 = nullptr, e1 = nullptr;
        LBC_CUDA(cudaEventCreate(&e0));
        LBC_CUDA(cudaEventCreate(&e1));
        LBC_CUDA(cudaEventRecord(e0, s));
        for (int t = 0; t < 50; ++t) { (void)newton_schulz(g, x, xn, wa, wb, r, c, steps, s); }
        LBC_CUDA(cudaEventRecord(e1, s));
        LBC_CUDA(cudaEventSynchronize(e1));
        float ms = 0.f;
        LBC_CUDA(cudaEventElapsedTime(&ms, e0, e1));
        reference(xd, r, c, steps);
        double num = 0, den = 0, dmin = 1e30, dmax = 0;
        for (size_t i = 0; i < n; ++i) {
            const double d = double(__bfloat162float(ob[i])) - xd[i];
            num += d * d;
            den += xd[i] * xd[i];
        }
        // diag of O O^T (wide) or O^T O (tall): the squared norms of the short side's vectors
        for (int i = 0; i < k; ++i) {
            double q = 0;
            if (r <= c) {
                for (int t = 0; t < c; ++t) { q += std::pow(double(__bfloat162float(ob[size_t(i) * c + t])), 2); }
            } else {
                for (int t = 0; t < r; ++t) { q += std::pow(double(__bfloat162float(ob[size_t(t) * c + i])), 2); }
            }
            dmin = std::min(dmin, q);
            dmax = std::max(dmax, q);
        }
        const double rel = std::sqrt(num / den);
        const bool good = rel < 0.05 && dmin > 0.3 && dmax < 1.6;
        ok = ok && good;
        std::printf("%4d x %4d: rel err vs float64 %.4f, diag(O O^T) in [%.3f, %.3f], %.1f us per call  %s\n", r, c,
                    rel, dmin, dmax, 1000.0 * ms / 50, good ? "ok" : "OFF");
        LBC_CUDA(cudaFree(x));
        LBC_CUDA(cudaFree(xn));
        LBC_CUDA(cudaFree(wa));
        LBC_CUDA(cudaFree(wb));
    }
    return ok ? 0 : 1;
}
