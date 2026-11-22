// Runs the assembly hgemm_f16 (model/kernels/hgemm/gen_hgemm.py) on a SYCL queue through Level Zero: checks it
// against f64 and times it at the MLP's shapes.
//
//   hgemm_bench <hstub.bin> <hgemm_f16.bin> [kc]
#include <level_zero/ze_api.h>
#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/backend/level_zero.hpp>
#include <elf.h>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <random>
#include <vector>

using namespace sycl;

static std::vector<unsigned char> read_file(const char* path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) { fprintf(stderr, "cannot read %s\n", path); exit(1); }
  return {std::istreambuf_iterator<char>(f), {}};
}

static std::vector<unsigned char> replace_text(std::vector<unsigned char> zebin, const char* kernel,
                                               const std::vector<unsigned char>& isa) {
  const size_t aligned = (zebin.size() + 63) & ~size_t(63);
  zebin.resize(aligned + isa.size());
  memcpy(zebin.data() + aligned, isa.data(), isa.size());
  auto* eh = reinterpret_cast<Elf64_Ehdr*>(zebin.data());
  auto* sh = reinterpret_cast<Elf64_Shdr*>(zebin.data() + eh->e_shoff);
  const char* names = reinterpret_cast<const char*>(zebin.data()) + sh[eh->e_shstrndx].sh_offset;
  const std::string want = std::string(".text.") + kernel;
  for (int i = 0; i < eh->e_shnum; i++)
    if (want == names + sh[i].sh_name) {
      sh[i].sh_offset = aligned, sh[i].sh_size = isa.size();
      return zebin;
    }
  fprintf(stderr, "section %s not found\n", want.c_str());
  exit(1);
}

#define ZE(x) do { ze_result_t r_ = (x); if (r_ != ZE_RESULT_SUCCESS) { fprintf(stderr, "%s: 0x%x\n", #x, r_); exit(1); } } while (0)

int main(int argc, char** argv) {
  if (argc < 3) { fprintf(stderr, "hgemm_bench <hstub.bin> <hgemm_f16.bin> [kc]\n"); return 1; }
  const int KC = argc > 3 ? atoi(argv[3]) : 16;
  queue q{gpu_selector_v, property::queue::in_order()};
  const auto isa = read_file(argv[2]);
  auto zebin = replace_text(read_file(argv[1]), "hgemm_f16", isa);
  auto zctx = get_native<backend::ext_oneapi_level_zero>(q.get_context());
  auto zdev = get_native<backend::ext_oneapi_level_zero>(q.get_device());
  ze_module_desc_t md{ZE_STRUCTURE_TYPE_MODULE_DESC};
  md.format = ZE_MODULE_FORMAT_NATIVE, md.inputSize = zebin.size(), md.pInputModule = zebin.data();
  ze_module_handle_t mod;
  ZE(zeModuleCreate(zctx, zdev, &md, &mod, nullptr));
  ze_kernel_desc_t kd{ZE_STRUCTURE_TYPE_KERNEL_DESC};
  kd.pKernelName = "hgemm_f16";
  ze_kernel_handle_t zk;
  ZE(zeKernelCreate(mod, &kd, &zk));
  size_t sz = 0;
  ZE(zeKernelGetBinaryExp(zk, &sz, nullptr));
  std::vector<unsigned char> loaded(sz);
  ZE(zeKernelGetBinaryExp(zk, &sz, loaded.data()));
  if (sz < isa.size() || memcmp(loaded.data(), isa.data(), isa.size())) { fprintf(stderr, "driver did not load our ISA\n"); return 1; }
  auto bundle = make_kernel_bundle<backend::ext_oneapi_level_zero, bundle_state::executable>(
      {mod, ext::oneapi::level_zero::ownership::keep}, q.get_context());
  kernel k = make_kernel<backend::ext_oneapi_level_zero>({bundle, zk, ext::oneapi::level_zero::ownership::keep},
                                                          q.get_context());

  const int shapes[][3] = {{32768, 3072, 768}, {32768, 768, 3072}, {32768, 768, 768}, {1000, 768, 200}};
  for (auto& s : shapes) {
    const int M = s[0], N = s[1], K = s[2];
    const int Mp = (M + 15) / 16 * 16, Kp = (K + KC - 1) / KC * KC;
    std::mt19937 g(0);
    std::normal_distribution<float> nd(0, 1);
    std::vector<half> A((size_t)M * K), B((size_t)K * N);
    const float ws = 1.f / std::sqrt((float)K);
    for (auto& v : A) v = half(nd(g));
    for (auto& v : B) v = half(nd(g) * ws);
    std::vector<half> ap((size_t)Mp * Kp, half(0)), bp((size_t)N * Kp, half(0));
    for (int m = 0; m < M; m++)
      for (int kk = 0; kk < K; kk++) ap[((size_t)(m / 16) * Kp + kk) * 16 + m % 16] = A[(size_t)m * K + kk];
    for (int kk = 0; kk < K; kk++)
      for (int n = 0; n < N; n++) bp[((size_t)(n / 32) * Kp + kk) * 32 + n % 32] = B[(size_t)kk * N + n];
    half* da = malloc_device<half>(ap.size(), q);
    half* db = malloc_device<half>(bp.size(), q);
    float* dc = malloc_device<float>((size_t)M * N, q);
    q.memcpy(da, ap.data(), ap.size() * 2);
    q.memcpy(db, bp.data(), bp.size() * 2);
    q.memset(dc, 0, (size_t)M * N * 4).wait();
    const float alpha = 0.5f, sa = 2.f, sb = 4.f;  // C = alpha / (sa sb) A B
    float* dscal = malloc_device<float>(16, q);
    std::vector<float> scal(16, 0.f);
    scal[0] = sa, scal[1] = sb;
    q.memcpy(dscal, scal.data(), 64).wait();
    const size_t groups = (size_t)(Mp / 16) * (N / 32);
    auto launch = [&] {
      return q.submit([&](handler& h) {
        h.set_args(da, db, dc, dscal, (unsigned)M, (unsigned)N, (unsigned)Kp, alpha);
        h.parallel_for(nd_range<1>(groups * 16, 16), k);
      });
    };
    launch().wait();
    std::vector<float> c((size_t)M * N);
    q.memcpy(c.data(), dc, c.size() * 4).wait();
    double err = 0, mag = 0;
    std::vector<int> rows;
    for (int i = 0; i < M; i += 97) rows.push_back(i);
    rows.push_back(M - 1);
    for (int i : rows)
      for (int j = 0; j < N; j++) {
        double ref = 0;
        for (int kk = 0; kk < K; kk++) ref += (double)(float)A[(size_t)i * K + kk] * (float)B[(size_t)kk * N + j];
        ref *= alpha / (sa * sb);
        err = std::max(err, std::fabs(ref - c[(size_t)i * N + j]));
        mag = std::max(mag, std::fabs(ref));
      }
    const int reps = 10;
    auto t0 = std::chrono::steady_clock::now();
    for (int r = 0; r < reps; r++) launch();
    q.wait();
    const double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() / reps;
    printf("M%-6d N%-5d K%-5d kc%-3d %8.2f ms  %.2f TFLOPS  max err %.2e (|ref| %.2f)\n", M, N, K, KC, sec * 1e3,
           2.0 * M * N * K / sec / 1e12, err, mag);
    free(da, q), free(db, q), free(dc, q), free(dscal, q);
  }
}
