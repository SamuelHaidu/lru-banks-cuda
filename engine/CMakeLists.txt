cmake_minimum_required(VERSION 3.26)

# nvcc is often installed but not on PATH (RunPod images): use the standard location unless told otherwise.
if(NOT DEFINED CMAKE_CUDA_COMPILER AND NOT DEFINED ENV{CUDACXX} AND EXISTS /usr/local/cuda/bin/nvcc)
  set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
endif()

# The GPU of the machine that builds (override with -DCMAKE_CUDA_ARCHITECTURES=89 or "86;89"). Must be set before
# project(): it would otherwise fill in the compiler's default (sm_52).
set(CMAKE_CUDA_ARCHITECTURES native CACHE STRING "CUDA architectures to build for")

project(lru_banks_cuda LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(CUDAToolkit REQUIRED)

# Shared headers (include/lbc) and flags for every target.
add_library(lbc_common INTERFACE)
target_include_directories(lbc_common INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(lbc_common INTERFACE CUDA::cudart)
target_compile_options(lbc_common INTERFACE
  $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra>
  $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo --expt-relaxed-constexpr -Xcompiler=-Wall,-Wextra>)

find_package(Threads REQUIRED)

# The model: kernels, cuBLASLt GEMMs, the data loader, run configs and checkpoints.
add_library(lbc STATIC
  src/json.cpp
  src/run_config.cpp
  src/checkpoint.cpp
  src/gemm.cpp
  src/data.cpp
  src/tokenizer.cpp
  src/calcx.cpp
  src/kernels_basic.cu
  src/kernels_lru_addr.cu
  src/kernels_bank.cu
  src/model.cu
  src/decode.cu)
target_include_directories(lbc PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(lbc PUBLIC lbc_common CUDA::cublasLt Threads::Threads)

add_executable(device_info tools/device_info.cu)
target_link_libraries(device_info PRIVATE lbc_common)

# The trainer: `lbc create / config / train` (the target is lbc_cli: `lbc` is the library)
add_executable(lbc_cli tools/lbc.cu tools/lbc_rl.cu)
target_link_libraries(lbc_cli PRIVATE lbc)
set_target_properties(lbc_cli PROPERTIES OUTPUT_NAME lbc)

add_executable(train_bench tools/train_bench.cu)
target_link_libraries(train_bench PRIVATE lbc)

add_executable(check_epilogue tools/check_epilogue.cu)
target_link_libraries(check_epilogue PRIVATE lbc)

add_executable(bench_fp8 tools/bench_fp8.cu)
target_link_libraries(bench_fp8 PRIVATE lbc)
