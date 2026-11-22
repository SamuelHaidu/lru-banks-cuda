// Prototype fp16 GEMM for Xe-LP (tgllp): C[M,N] (f32) = A[M,K] (f16) . B[K,N] (f16), row-major.
//
// oneDNN's layout (see its ISA): a sub-group computes MT rows x 16*NC columns, lane l owns columns l + 16 c; per
// k, one B row vector per column block times an A element broadcast through a register region. Here the mads are
// half (2x the f32 rate on Xe-LP): products are summed in half over KC steps, then added into f32 accumulators.
//
//   icpx -fsycl -fsycl-targets=spir64_gen -Xs "-device tgllp" -O3 -std=c++20 hgemm.cpp -o hgemm
//   LD_LIBRARY_PATH=<repo>/.venv/lib ./hgemm
#include <sycl/sycl.hpp>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

using namespace sycl;
constexpr int SG = 16;

// sub-group block read of 16 halves (lane l gets p[l]); p 4-byte aligned
static inline half blk(const sub_group&, const half* p) {
#ifdef __SYCL_DEVICE_ONLY__
  using gp = const __attribute__((opencl_global)) uint16_t*;
  return bit_cast<half>(__spirv_SubgroupBlockReadINTEL<uint16_t>((gp)(const void*)p));
#else
  return *p;
#endif
}

template <int MT, int NC, int KB, int KC>
struct HGemm {
  // MT rows x 16 NC columns per sub-group; A read KB columns at a time (KB / 16 block reads per row), products
  // summed in half over KC k steps, then added into the f32 accumulators. Offsets are 32-bit, pointers advance.
  static_assert(KB % SG == 0 && KB % KC == 0);
  const half *A, *B;
  float* C;
  int M, N, K;
  [[sycl::reqd_sub_group_size(SG)]] void operator()(nd_item<1> it) const {
    auto sg = it.get_sub_group();
    const int lane = sg.get_local_linear_id();
    const int tiles_n = N / (SG * NC);
    // the tile from uniform values (work-group id, sub-group id): a tile from get_global_id / 16 is not provably
    // uniform, which makes every offset per-lane and the k loop SIMD control flow (sends predicated any16h)
    const int t = it.get_group(0) * (it.get_local_range(0) / SG) + sg.get_group_linear_id();
    const int m0 = (t / tiles_n) * MT, n0 = (t % tiles_n) * SG * NC;
    float acc[MT][NC];
#pragma unroll
    for (int r = 0; r < MT; r++)
#pragma unroll
      for (int c = 0; c < NC; c++) acc[r][c] = 0.f;
    int oa = m0 * K, ob = n0;  // offsets from the kernel arguments (32-bit: stateful, binding-table reads)
    for (int k0 = 0; k0 < K; k0 += KB) {
      half a[KB / SG][MT];  // lane l: A[m0 + r, k0 + 16 j + l]
#pragma unroll
      for (int j = 0; j < KB / SG; j++)
#pragma unroll
        for (int r = 0; r < MT; r++) a[j][r] = blk(sg, A + (oa + r * K + SG * j));
#pragma unroll
      for (int kc = 0; kc < KB; kc += KC) {
        half h[MT][NC];
#pragma unroll
        for (int r = 0; r < MT; r++)
#pragma unroll
          for (int c = 0; c < NC; c++) h[r][c] = 0;
#pragma unroll
        for (int kk = kc; kk < kc + KC; kk++) {
          half b[NC];
#pragma unroll
          for (int c = 0; c < NC; c++) b[c] = blk(sg, B + (ob + SG * c));
          ob += N;
#pragma unroll
          for (int r = 0; r < MT; r++) {
            const half ar = select_from_group(sg, a[kk / SG][r], kk % SG);
#pragma unroll
            for (int c = 0; c < NC; c++) h[r][c] = fma(ar, b[c], h[r][c]);
          }
        }
#pragma unroll
        for (int r = 0; r < MT; r++)
#pragma unroll
          for (int c = 0; c < NC; c++) acc[r][c] += (float)h[r][c];
      }
      oa += KB;
    }
#pragma unroll
    for (int r = 0; r < MT; r++)
#pragma unroll
      for (int c = 0; c < NC; c++) C[(m0 + r) * N + n0 + lane + SG * c] = acc[r][c];
  }
};

template <int MT, int NC, int KB, int KC>
void run(queue& q, const half* A, const half* B, float* C, int M, int N, int K, const std::vector<float>& ref,
         const char* label) {
  const size_t items = (size_t)(M / MT) * (N / (SG * NC)) * SG;
  auto go = [&] { q.parallel_for(nd_range<1>(items, 64), HGemm<MT, NC, KB, KC>{A, B, C, M, N, K}).wait(); };
  go();
  auto t = std::chrono::steady_clock::now();
  const int reps = 5;
  for (int r = 0; r < reps; r++) go();
  const double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t).count() / reps;
  std::vector<float> c((size_t)M * N);
  q.memcpy(c.data(), C, c.size() * 4).wait();
  double err = 0, mag = 0;  // vs the f64 reference on the first 64 rows
  for (size_t i = 0; i < ref.size(); i++) err = std::max(err, (double)std::fabs(c[i] - ref[i])), mag = std::max(mag, (double)std::fabs(ref[i]));
  printf("%-22s M%-6d N%-5d K%-5d %7.2f ms  %.2f TFLOPS  max err %.2e (|ref| %.2f)\n", label, M, N, K, sec * 1e3,
         2.0 * M * N * K / sec / 1e12, err, mag);
}

int main() {
  queue q{gpu_selector_v};
  printf("%s\n", q.get_device().get_info<info::device::name>().c_str());
  const int shapes[3][3] = {{32768, 3072, 768}, {32768, 768, 3072}, {32768, 768, 768}};
  for (auto& s : shapes) {
    const int M = s[0], N = s[1], K = s[2];
    std::mt19937 g(0);
    std::normal_distribution<float> nd(0, 1);
    std::vector<half> a((size_t)M * K), b((size_t)K * N);
    std::vector<float> af(64 * (size_t)K), bf((size_t)K * N);
    for (size_t i = 0; i < a.size(); i++) { a[i] = half(nd(g)); if (i < af.size()) af[i] = (float)a[i]; }
    const float ws = 1.f / std::sqrt((float)K);  // weights at init scale: outputs ~ N(0, 1)
    for (size_t i = 0; i < b.size(); i++) { b[i] = half(nd(g) * ws); bf[i] = (float)b[i]; }
    std::vector<float> ref(64 * (size_t)N);
    for (int i = 0; i < 64; i++)
      for (int j = 0; j < N; j++) {
        double acc = 0;
        for (int k = 0; k < K; k++) acc += (double)af[(size_t)i * K + k] * bf[(size_t)k * N + j];
        ref[(size_t)i * N + j] = (float)acc;
      }
    half* A = malloc_device<half>(a.size(), q);
    half* B = malloc_device<half>(b.size(), q);
    float* C = malloc_device<float>((size_t)M * N, q);
    q.memcpy(A, a.data(), a.size() * 2).wait();
    q.memcpy(B, b.data(), b.size() * 2).wait();
    run<12, 2, 16, 16>(q, A, B, C, M, N, K, ref, "MT12 NC2 KB16 KC16");
    run<12, 2, 32, 32>(q, A, B, C, M, N, K, ref, "MT12 NC2 KB32 KC32");
    run<10, 2, 32, 32>(q, A, B, C, M, N, K, ref, "MT10 NC2 KB32 KC32");
    run<8, 3, 32, 32>(q, A, B, C, M, N, K, ref, "MT8 NC3 KB32 KC32");
    free(A, q), free(B, q), free(C, q);
  }
}
