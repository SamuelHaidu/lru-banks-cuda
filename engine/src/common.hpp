#pragma once
// Types and fixed sizes shared by every kernel.

#include <cstdint>

#include <cuda_bf16.h>

namespace lbc {

using bf16 = __nv_bfloat16;

// the bank head size (keys and values) is a model setting, 32 or 64 (Config::dh; the kernels are templates)
constexpr int SLOTS = 128;         // bank slots per head (one thread per slot in the bank kernels)
constexpr float BANK_EPS = 1e-6f;  // a slot is readable when its written mass exceeds this
constexpr float NORM_EPS = 1e-12f; // F.normalize's epsilon
constexpr float RMS_EPS = 1e-6f;

}  // namespace lbc
