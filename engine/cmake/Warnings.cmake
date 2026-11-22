# Compiler warnings, hardening and sanitizers for every target, through the lbc_common interface library.
#
# The same code is compiled by two CUDA compilers: nvcc (the production build, GCC for the host side) and clang (the
# lint build, preset "lint"). Each catches bugs the other misses, so the gate builds both with warnings as errors.

option(LBC_WERROR "Treat compiler warnings as errors" OFF)
option(LBC_HARDEN "Host hardening: _FORTIFY_SOURCE, libstdc++ assertions, stack protector, full RELRO" ON)
set(LBC_SANITIZE "" CACHE STRING "Host sanitizers for the C++ side, e.g. address,undefined (preset asan)")

# Warnings for C++ code, on the host compiler (GCC or clang) and on clang's CUDA device code.
set(LBC_CXX_WARNINGS
  -Wall
  -Wextra
  -Wshadow                 # a local hiding another variable
  -Wnon-virtual-dtor
  -Woverloaded-virtual
  -Wcast-qual              # casting const away
  -Wformat=2               # printf formats against their arguments, non-literal formats
  -Wimplicit-fallthrough
  -Wnull-dereference
  -Wvla                    # variable-length arrays (stack overflow)
  -Wundef                  # #if on an undefined macro
)

# GCC-only and clang-only additions.
set(LBC_GCC_WARNINGS -Wduplicated-cond -Wduplicated-branches -Wlogical-op -Wuseless-cast)
set(LBC_CLANG_WARNINGS -Wunreachable-code-aggressive -Wloop-analysis -Wcomma -Wno-unknown-cuda-version)

string(REPLACE "," ";" _lbc_sanitizers "${LBC_SANITIZE}")

function(lbc_host_flags out compiler_id)
  set(flags ${LBC_CXX_WARNINGS})
  if(compiler_id STREQUAL "GNU")
    list(APPEND flags ${LBC_GCC_WARNINGS})
  elseif(compiler_id MATCHES "Clang")
    list(APPEND flags ${LBC_CLANG_WARNINGS})
  endif()
  if(LBC_WERROR)
    list(APPEND flags -Werror)
  endif()
  if(LBC_HARDEN)
    # _FORTIFY_SOURCE needs optimization; libstdc++ assertions check operator[] bounds, front() on empty, ...
    list(APPEND flags -D_GLIBCXX_ASSERTIONS -fstack-protector-strong -fstack-clash-protection -fcf-protection)
    if(NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
      list(APPEND flags -U_FORTIFY_SOURCE -D_FORTIFY_SOURCE=3)
    endif()
  endif()
  # one -fsanitize= per sanitizer: nvcc splits -Xcompiler values at commas
  foreach(san IN LISTS _lbc_sanitizers)
    list(APPEND flags -fsanitize=${san})
  endforeach()
  if(LBC_SANITIZE)
    list(APPEND flags -fno-omit-frame-pointer -fno-sanitize-recover=all)
  endif()
  set(${out} ${flags} PARENT_SCOPE)
endfunction()

lbc_host_flags(_lbc_cxx_flags "${CMAKE_CXX_COMPILER_ID}")
target_compile_options(lbc_common INTERFACE $<$<COMPILE_LANGUAGE:CXX>:${_lbc_cxx_flags}>)

if(CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA")
  # nvcc: its own front end's warnings, then the host flags for the GCC that compiles the host side.
  lbc_host_flags(_lbc_host_flags "${CMAKE_CUDA_HOST_COMPILER_ID}")
  if(NOT CMAKE_CUDA_HOST_COMPILER_ID)
    lbc_host_flags(_lbc_host_flags "GNU")
  endif()
  # -Wuseless-cast and -Wduplicated-branches fire on nvcc's generated host stubs, not on our code.
  list(REMOVE_ITEM _lbc_host_flags -Wuseless-cast -Wduplicated-branches)
  list(JOIN _lbc_host_flags "," _lbc_host_csv)
  set(_lbc_cuda_flags
    -lineinfo
    --expt-relaxed-constexpr
    -Xcompiler=${_lbc_host_csv}
    --Werror=cross-execution-space-call)  # a __device__ function calling a __host__ one
  if(LBC_WERROR)
    list(APPEND _lbc_cuda_flags --Werror=all-warnings)
  endif()
elseif(CMAKE_CUDA_COMPILER_ID STREQUAL "Clang")
  lbc_host_flags(_lbc_cuda_flags "Clang")
  # clang compiles each .cu twice (host and device); the control-flow and stack-clash protections exist for the
  # host only. The production build (nvcc) keeps them.
  list(REMOVE_ITEM _lbc_cuda_flags -fcf-protection -fstack-clash-protection)
endif()
target_compile_options(lbc_common INTERFACE $<$<COMPILE_LANGUAGE:CUDA>:${_lbc_cuda_flags}>)

if(LBC_HARDEN)
  target_link_options(lbc_common INTERFACE -Wl,-z,relro,-z,now)
endif()
if(LBC_SANITIZE)
  foreach(san IN LISTS _lbc_sanitizers)
    target_link_options(lbc_common INTERFACE -fsanitize=${san})
  endforeach()
endif()

# Kernel sources: a float silently promoted to double is slow on GPUs (1/64 rate on consumer cards) and almost
# never intended. Host-side tools (printf of floats, statistics) are exempt.
set(LBC_KERNEL_WARNINGS -Wdouble-promotion)
if(LBC_WERROR)
  set(LBC_KERNEL_WARNINGS -Werror=double-promotion)
endif()
if(CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA")
  list(JOIN LBC_KERNEL_WARNINGS "," _lbc_kernel_csv)
  set(LBC_KERNEL_WARNINGS -Xcompiler=${_lbc_kernel_csv})
endif()
