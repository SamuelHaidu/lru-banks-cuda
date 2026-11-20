#pragma once
// Error checks: any failing CUDA runtime call aborts with the call, the message, the file and the line.

#include <cstdio>
#include <cstdlib>

#include <cuda_runtime.h>

namespace lbc {

[[noreturn]] inline void fail(const char* call, const char* message, const char* file, int line) {
    std::fprintf(stderr, "%s failed: %s (%s:%d)\n", call, message, file, line);
    std::abort();
}

}  // namespace lbc

#define LBC_CUDA(call)                                                                                    \
    do {                                                                                                  \
        const cudaError_t lbc_err_ = (call);                                                              \
        if (lbc_err_ != cudaSuccess) ::lbc::fail(#call, cudaGetErrorString(lbc_err_), __FILE__, __LINE__); \
    } while (0)
