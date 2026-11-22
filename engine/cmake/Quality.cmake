# The quality gate: formatting, static analysis and tests, as CMake targets run by Ninja.
#
#   format        rewrite every C++ / CUDA source with clang-format
#   format-check  fail if any source is not formatted
#   tidy          clang-tidy: host profile (.clang-tidy) on host code, CUDA profile (cuda.clang-tidy) on the kernel
#                 library and on the device half of every .cu. Only in a tree compiled by clang (preset lint): it
#                 reads that tree's compile_commands.json, which nvcc's flags would not fit.
#   cppcheck      a second analyzer on the host sources (lint tree)
#   lint          from any other tree: configure and build the lint tree (clang -Werror, tidy, cppcheck)
#   gate          format-check + lint + this tree's build; then `ctest` (cmake --workflow --preset gate does all)
#
# Each source is linted by its own Ninja edge with a stamp file, so a second run re-checks only what changed (any
# header or profile change re-checks everything).

set(LBC_LLVM_VERSION 20 CACHE STRING "LLVM major version of clang-format / clang-tidy (results differ across versions)")
find_program(LBC_CLANG_FORMAT NAMES clang-format-${LBC_LLVM_VERSION} clang-format)
find_program(LBC_CLANG_TIDY NAMES clang-tidy-${LBC_LLVM_VERSION} clang-tidy)
find_program(LBC_CPPCHECK NAMES cppcheck)

set(_src ${CMAKE_CURRENT_SOURCE_DIR})
file(GLOB _lbc_cpp CONFIGURE_DEPENDS ${_src}/src/*.cpp)
file(GLOB _lbc_cu CONFIGURE_DEPENDS ${_src}/src/*.cu ${_src}/tools/*.cu)
file(GLOB _lbc_headers CONFIGURE_DEPENDS ${_src}/src/*.hpp ${_src}/src/*.cuh ${_src}/include/lbc/*.hpp
  ${_src}/tools/*.hpp)
set(LBC_FORMAT_FILES ${_lbc_cpp} ${_lbc_cu} ${_lbc_headers})

function(lbc_require_tool var what)
  if(NOT ${var})
    message(FATAL_ERROR "${what} not found. Install LLVM ${LBC_LLVM_VERSION} (clang-format-${LBC_LLVM_VERSION}, "
      "clang-tidy-${LBC_LLVM_VERSION}) and cppcheck, or run scripts/gate.sh (it uses the toolchain container).")
  endif()
endfunction()

# --- formatting ---------------------------------------------------------------------------------------------------
if(LBC_CLANG_FORMAT)
  add_custom_target(format
    COMMAND ${LBC_CLANG_FORMAT} -i ${LBC_FORMAT_FILES}
    WORKING_DIRECTORY ${_src} COMMENT "clang-format: rewriting sources" VERBATIM)
  add_custom_target(format-check
    COMMAND ${LBC_CLANG_FORMAT} --dry-run --Werror ${LBC_FORMAT_FILES}
    WORKING_DIRECTORY ${_src} COMMENT "clang-format: checking (fix with: cmake --build --preset lint --target format)"
    VERBATIM)
else()
  add_custom_target(format COMMAND ${CMAKE_COMMAND} -E false COMMENT "clang-format not found")
  add_custom_target(format-check COMMAND ${CMAKE_COMMAND} -E false COMMENT "clang-format not found")
endif()

# --- static analysis (lint tree: clang compiles C++ and CUDA) -----------------------------------------------------
if(CMAKE_CUDA_COMPILER_ID STREQUAL "Clang")
  lbc_require_tool(LBC_CLANG_TIDY clang-tidy)
  lbc_require_tool(LBC_CPPCHECK cppcheck)
  set(_stamps ${CMAKE_CURRENT_BINARY_DIR}/tidy)
  set(_tidy_outputs "")

  # one clang-tidy run: source, profile, extra arguments; a stamp records success
  function(lbc_tidy src profile tag)
    file(RELATIVE_PATH rel ${_src} ${src})
    string(MAKE_C_IDENTIFIER "${rel}" id)
    set(stamp ${_stamps}/${id}.${tag}.ok)
    add_custom_command(OUTPUT ${stamp}
      COMMAND ${LBC_CLANG_TIDY} --quiet --config-file=${profile} -p ${CMAKE_BINARY_DIR}
        --extra-arg=-Wno-unused-command-line-argument ${ARGN} ${src}
      COMMAND ${CMAKE_COMMAND} -E touch ${stamp}
      DEPENDS ${src} ${profile} ${_lbc_headers}
      WORKING_DIRECTORY ${_src}
      COMMENT "clang-tidy (${tag}) ${rel}" VERBATIM)
    set(_tidy_outputs ${_tidy_outputs} ${stamp} PARENT_SCOPE)
  endfunction()

  # Which profile: the kernel library (src/*.cu) is low-level CUDA, so both of its halves get cuda.clang-tidy. The
  # programs in tools/*.cu are mostly host code (CLIs, benchmarks): host profile, plus the CUDA profile on their
  # device half. (A host-only pass still analyzes __device__ bodies, so the profile follows the file, not the half.)
  file(MAKE_DIRECTORY ${_stamps})
  foreach(f IN LISTS _lbc_cpp)
    lbc_tidy(${f} ${_src}/.clang-tidy host)
  endforeach()
  foreach(f IN LISTS _lbc_cu)
    if(f MATCHES "/src/[^/]+\\.cu$")
      lbc_tidy(${f} ${_src}/cuda.clang-tidy host --extra-arg=--cuda-host-only)
    else()
      lbc_tidy(${f} ${_src}/.clang-tidy host --extra-arg=--cuda-host-only)
    endif()
    lbc_tidy(${f} ${_src}/cuda.clang-tidy device --extra-arg=--cuda-device-only)
  endforeach()
  add_custom_target(tidy DEPENDS ${_tidy_outputs})

  add_custom_target(cppcheck
    COMMAND ${LBC_CPPCHECK} --project=${CMAKE_BINARY_DIR}/compile_commands.json "--file-filter=*.cpp"
      --enable=warning,performance,portability --std=c++20 --inline-suppr --quiet --error-exitcode=1
      --suppress=missingIncludeSystem --cppcheck-build-dir=${CMAKE_CURRENT_BINARY_DIR}/cppcheck
    COMMAND_EXPAND_LISTS
    WORKING_DIRECTORY ${_src} COMMENT "cppcheck: host sources" VERBATIM)
  file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/cppcheck)
else()
  # any other tree delegates to the lint tree (its own configure, build directory build/lint)
  add_custom_target(lint
    COMMAND ${CMAKE_COMMAND} --preset lint
    COMMAND ${CMAKE_COMMAND} --build --preset lint
    WORKING_DIRECTORY ${_src} COMMENT "lint: clang build with -Werror, clang-tidy, cppcheck (build/lint)" USES_TERMINAL
    VERBATIM)
  # everything the gate builds: the checks and this tree's programs (not part of `all`: the edit loop stays fast)
  add_custom_target(gate)
  add_dependencies(gate format-check lint lbc_cli device_info train_bench check_epilogue bench_fp8)
endif()

# --- tests --------------------------------------------------------------------------------------------------------
# The engine's checks need data (a Calc-X dataset, a trained run) and most need a GPU, so they are registered only
# when pointed at them. compute-sanitizer then runs the GPU check under each of its tools: memcheck (out-of-bounds
# and misaligned accesses, leaks), racecheck (shared-memory races), synccheck (invalid __syncthreads use), initcheck
# (reads of uninitialized device memory).
set(LBC_TEST_CALCX_DIR "" CACHE PATH "Calc-X dataset directory for `lbc test-calcx` (empty: not registered)")
set(LBC_TEST_TOKENIZER "" CACHE FILEPATH "tokenizer.json for `lbc test-calcx`")
set(LBC_TEST_RUN "" CACHE PATH "a run directory with model.lbc for `lbc decode-check` (GPU; empty: not registered)")
option(LBC_SANITIZER_TESTS "Also run the GPU checks under compute-sanitizer (memcheck, racecheck, synccheck, initcheck)"
  OFF)

if(LBC_TEST_CALCX_DIR AND LBC_TEST_TOKENIZER)
  add_test(NAME calcx-oracles COMMAND lbc_cli test-calcx ${LBC_TEST_CALCX_DIR} ${LBC_TEST_TOKENIZER} --split test)
endif()
if(LBC_TEST_RUN)
  add_test(NAME decode-check COMMAND lbc_cli decode-check ${LBC_TEST_RUN} --rows 2)
  set_tests_properties(decode-check PROPERTIES LABELS gpu)
  if(LBC_SANITIZER_TESTS)
    find_program(LBC_COMPUTE_SANITIZER compute-sanitizer HINTS ${CUDAToolkit_BIN_DIR} REQUIRED)
    foreach(tool memcheck racecheck synccheck initcheck)
      add_test(NAME decode-check-${tool}
        COMMAND ${LBC_COMPUTE_SANITIZER} --tool ${tool} --error-exitcode 1 $<TARGET_FILE:lbc_cli> decode-check
          ${LBC_TEST_RUN} --rows 2)
      set_tests_properties(decode-check-${tool} PROPERTIES LABELS "gpu;sanitizer")
    endforeach()
  endif()
endif()
