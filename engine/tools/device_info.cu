// Prints the GPU properties that matter for kernel design and for the research log's hardware column.

#include <cstdio>

#include <cuda_runtime.h>

#include "lbc/check.hpp"

static int attr(cudaDeviceAttr a, int dev) {
    int v = 0;
    LBC_CUDA(cudaDeviceGetAttribute(&v, a, dev));
    return v;
}

int main() {
    int count = 0, driver = 0, runtime = 0;
    LBC_CUDA(cudaGetDeviceCount(&count));
    LBC_CUDA(cudaDriverGetVersion(&driver));
    LBC_CUDA(cudaRuntimeGetVersion(&runtime));
    std::printf("CUDA driver %d.%d, runtime %d.%d, %d device(s)\n", driver / 1000, (driver % 1000) / 10, runtime / 1000,
                (runtime % 1000) / 10, count);
    for (int d = 0; d < count; ++d) {
        cudaDeviceProp p{};
        LBC_CUDA(cudaGetDeviceProperties(&p, d));
        const double sm_mhz = attr(cudaDevAttrClockRate, d) / 1e3;
        const double mem_mhz = attr(cudaDevAttrMemoryClockRate, d) / 1e3;
        const int bus_bits = attr(cudaDevAttrGlobalMemoryBusWidth, d);
        const double peak_gbs = 2.0 * mem_mhz * 1e6 * (bus_bits / 8.0) / 1e9;  // double data rate
        std::printf("[%d] %s, sm_%d%d, %d SMs, max SM clock %.0f MHz\n", d, p.name, p.major, p.minor,
                    p.multiProcessorCount, sm_mhz);
        std::printf("    memory %.1f GiB, %d-bit bus at %.0f MHz: %.0f GB/s theoretical\n",
                    p.totalGlobalMem / double(1 << 30), bus_bits, mem_mhz, peak_gbs);
        std::printf("    L2 %.1f MiB (persisting up to %.1f MiB)\n", p.l2CacheSize / double(1 << 20),
                    p.persistingL2CacheMaxSize / double(1 << 20));
        std::printf("    shared memory %zu KiB per SM, %zu KiB per block (opt-in); %d registers per SM\n",
                    p.sharedMemPerMultiprocessor / 1024, p.sharedMemPerBlockOptin / 1024, p.regsPerMultiprocessor);
        std::printf("    %d threads per SM, warp %d, ECC %s\n", p.maxThreadsPerMultiProcessor, p.warpSize,
                    (p.ECCEnabled != 0) ? "on" : "off");
    }
    return 0;
}
