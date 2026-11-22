// Feasibility: run a hand-assembled Xe-LP kernel (tiny-lm's matmul_f16) on a SYCL queue through Level Zero.
//
// The zebin comes from ocloc compiling the OpenCL stub; its `.text.<kernel>` section is pointed at the assembled
// machine code (as tiny-lm's zebin_replace_text), and the patched zebin is loaded as a native L0 module, wrapped
// into a sycl::kernel and launched with USM pointers. Checks C = A B against the CPU.
//
//   zeload <stub.zebin> <kernel> <isa.bin>
#include <level_zero/ze_api.h>
#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/backend/level_zero.hpp>
#include <elf.h>
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
  if (argc < 4) { fprintf(stderr, "zeload <stub.zebin> <kernel> <isa.bin>\n"); return 1; }
  queue q{gpu_selector_v, property::queue::in_order()};
  auto zebin = replace_text(read_file(argv[1]), argv[2], read_file(argv[3]));
  auto isa = read_file(argv[3]);

  auto zctx = get_native<backend::ext_oneapi_level_zero>(q.get_context());
  auto zdev = get_native<backend::ext_oneapi_level_zero>(q.get_device());
  ze_module_desc_t md{ZE_STRUCTURE_TYPE_MODULE_DESC};
  md.format = ZE_MODULE_FORMAT_NATIVE;
  md.inputSize = zebin.size();
  md.pInputModule = zebin.data();
  ze_module_handle_t mod;
  ze_module_build_log_handle_t log;
  ZE(zeModuleCreate(zctx, zdev, &md, &mod, &log));
  ze_kernel_desc_t kd{ZE_STRUCTURE_TYPE_KERNEL_DESC};
  kd.pKernelName = argv[2];
  ze_kernel_handle_t zk;
  ZE(zeKernelCreate(mod, &kd, &zk));

  // what the driver loaded must be our code (the zebin also carries SPIR-V it could recompile)
  size_t sz = 0;
  ZE(zeKernelGetBinaryExp(zk, &sz, nullptr));
  std::vector<unsigned char> loaded(sz);
  ZE(zeKernelGetBinaryExp(zk, &sz, loaded.data()));
  const bool ours = sz >= isa.size() && !memcmp(loaded.data(), isa.data(), isa.size());
  printf("loaded %zu bytes, our ISA: %s\n", sz, ours ? "yes" : "NO");

  auto bundle = make_kernel_bundle<backend::ext_oneapi_level_zero, bundle_state::executable>(
      {mod, ext::oneapi::level_zero::ownership::keep}, q.get_context());
  kernel k = make_kernel<backend::ext_oneapi_level_zero>({bundle, zk, ext::oneapi::level_zero::ownership::keep},
                                                          q.get_context());

  // tiny-lm matmul_f16: C (n x n, half) = A B; a[panel][k][16], b[panel][k][64] packed, one 16-lane thread per tile
  const int n = 1024, kk = 512, MR = 16, NC = 64, rep = 1;
  std::mt19937 g(0);
  std::uniform_int_distribution<int> ud(-2, 2);  // small integers: exact in half
  std::vector<float> A((size_t)n * kk), B((size_t)kk * n);
  for (auto& v : A) v = ud(g);
  for (auto& v : B) v = ud(g);
  std::vector<half> ap((size_t)n * kk), bp((size_t)kk * n);
  for (int p = 0; p < n / MR; p++)
    for (int s = 0; s < kk; s++)
      for (int m = 0; m < MR; m++) ap[((size_t)p * kk + s) * MR + m] = A[(size_t)(p * MR + m) * kk + s];
  for (int p = 0; p < n / NC; p++)
    for (int s = 0; s < kk; s++)
      for (int c = 0; c < NC; c++) bp[((size_t)p * kk + s) * NC + c] = B[(size_t)s * n + p * NC + c];
  half* da = malloc_device<half>(ap.size(), q);
  half* db = malloc_device<half>(bp.size(), q);
  half* dc = malloc_device<half>((size_t)n * n, q);
  q.memcpy(da, ap.data(), ap.size() * 2);
  q.memcpy(db, bp.data(), bp.size() * 2).wait();
  const size_t tiles = (size_t)(n / MR) * (n / NC);
  auto launch = [&] {
    return q.submit([&](handler& h) {
      h.set_args(da, db, dc, (unsigned)n, (unsigned)kk, (unsigned)rep);
      h.parallel_for(nd_range<1>(tiles * rep * 16, 16), k);
    });
  };
  launch().wait();
  std::vector<half> c((size_t)n * n);
  q.memcpy(c.data(), dc, c.size() * 2).wait();
  double err = 0;
  for (int i = 0; i < n; i += 37)
    for (int j = 0; j < n; j++) {
      double ref = 0;
      for (int s = 0; s < kk; s++) ref += A[(size_t)i * kk + s] * B[(size_t)s * n + j];
      err = std::max(err, std::fabs(ref - (float)c[(size_t)i * n + j]));
    }
  printf("max |C - ref| = %g\n", err);
  auto t0 = std::chrono::steady_clock::now();
  const int reps = 20;
  for (int r = 0; r < reps; r++) launch();
  q.wait();
  const double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() / reps;
  printf("%.3f ms  %.2f TFLOPS\n", sec * 1e3, 2.0 * n * n * kk / sec / 1e12);
}
