#pragma once
// Launchers of every custom kernel. Layout conventions: activations are row-major with one row per token
// (r = b * T + t, M = B * T rows); per-head tensors are [M][H][DH]; gates and routing are [M][H].

#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

#include "common.hpp"

namespace lbc {

// -- norms and elementwise (kernels_basic.cu) ---------------------------------------------------------------------

// xh = h / rms(h) (bf16), rstd = 1 / rms(h). The gain is folded into the next GEMM's weight.
void rmsnorm_fwd(const float* h, bf16* xh, float* rstd, int M, int D, cudaStream_t s);
// dh (+)= rstd (dxh - xh * mean(dxh * xh))
void rmsnorm_bwd(const float* dxh, const bf16* xh, const float* rstd, float* dh, int M, int D, bool accumulate,
                 cudaStream_t s);
void to_bf16(const float* x, bf16* y, size_t n, cudaStream_t s);
// z <- z + b (the pre-activation, kept for the backward), u = gelu_tanh(z)
void bias_gelu_fwd(bf16* z, const float* b, bf16* u, int M, int F, cudaStream_t s);
// dz = du * gelu_tanh'(z)
void gelu_bwd(const float* du, const bf16* z, bf16* dz, int M, int F, cudaStream_t s);
void add_bias(float* h, const float* b, int M, int D, cudaStream_t s);
// out[c] += sum over the M rows of x[r][c] (row stride ld)
void colsum(const float* x, int M, int N, int ld, float* out, cudaStream_t s);
void colsum(const bf16* x, int M, int N, int ld, float* out, cudaStream_t s);
void embed_gather(const int* ids, const bf16* E, bf16* x, int M, int E0, cudaStream_t s);
void embed_scatter(const int* ids, const float* dx, float* dE, int M, int E0, cudaStream_t s);
// Rows of logits [rows][V]: loss_sum += sum (lse - logit[y]) * inv_n; dlogits = (softmax - onehot) * inv_n.
// Rows with y < 0 add nothing and get zero gradients.
void ce_rows(const float* logits, const int* y, bf16* dlogits, float* loss_sum, int rows, int V, float inv_n,
             cudaStream_t s);

// -- optimizer (kernels_basic.cu) ---------------------------------------------------------------------------------

void sumsq(const float* x, size_t n, float* out, cudaStream_t s);
// AdamW with the update scaled by min(1, clip / ||g||) (||g||^2 read from gsq on the device: no host sync).
void adamw(float* p, const float* g, float* m, float* v, const uint8_t* decay, size_t n, float lr, float beta1,
           float beta2, float eps, float wd, float bc1, float bc2, const float* gsq, float clip, cudaStream_t s);
// Wb = bf16(diag(gain) W) (gain may be null)
void prep_weight(const float* W, const float* gain, bf16* Wb, int rows, int cols, cudaStream_t s);
// From dWp = dL/d(diag(gain) W): dW += diag(gain) dWp, dgain += rowsum(W * dWp) (gain may be null: dW += dWp)
void unfold_grad(const float* dWp, const float* W, const float* gain, float* dW, float* dgain, int rows, int cols,
                 cudaStream_t s);

// -- LRU (kernels_lru_addr.cu) ------------------------------------------------------------------------------------

// h_t = lambda h_{t-1} + gamma (P_re + i P_im), restarting at resets; lambda = exp(-exp(nu) + i theta),
// gamma = sqrt(1 - |lambda|^2). P: columns [0, N) real, [N, 2N) imaginary, row stride ldp. s = [Re h, Im h].
void lru_fwd(const bf16* P, int ldp, const float* nu, const float* theta, const uint8_t* reset, float* s, bf16* sb,
             int B, int T, int N, cudaStream_t st);
// dP (bf16, row stride lddp) from ds; dnu, dtheta accumulated
void lru_bwd(const float* ds, const float* s, const bf16* P, int ldp, const float* nu, const float* theta,
             const uint8_t* reset, bf16* dP, int lddp, float* dnu, float* dtheta, int B, int T, int N,
             cudaStream_t st);

// -- addressing: keys, queries, routing, gates (kernels_lru_addr.cu) ----------------------------------------------

struct AddrFwd {
    const float* G;  // [M][ldG]: [k_pre (H*DH) | q_pre (H*DH) | read gate (H) | write gate (H)], k_pre unshifted
    int ldG;
    const float *bk, *brg, *bwg;  // key bias [H*DH], gate biases [H]
    const float* R;               // router [H][DH][SLOTS]
    const uint8_t* reset;
    bf16 *k, *q;                  // [M][H][DH], normalized
    float *kn, *qn;               // [M][H] norms before normalizing
    uint8_t* ridx;                // [M][H][2] the two slots
    float* rp1;                   // [M][H] weight of the first slot (the second gets 1 - p1)
    uint8_t *gw, *gr;             // [M][H] binary write / read gates (no write without a previous state:
                                  // at a reset or a row's first token the key would be normalize(b_k) alone)
    float *aw, *ar;               // [M][H] gate logits
    float* stats;                 // [4]: sum sigmoid(aw), open writes, sum sigmoid(ar), open reads
    uint2* info;                  // [M][H] packed for the bank walks: x = m1 | m2 << 8 | flags << 16 (flags: 1 write
                                  // gate, 2 read gate, 4 reset, 8 hop-2 read gate, set by q2_fwd), y = bits of p1
    int M, T, H;
};
void addr_fwd(const AddrFwd& a, cudaStream_t s);

struct AddrBwd {
    // from the bank adjoint, per (row, head) and routed slot s = 0, 1: dk2 / dv2 [M][H][2][DH] (written only when
    // the write gate is open), dr2 [M][H][2] the gradient of the slot's write weight
    const float *dk2, *dv2, *dr2;
    const float* dq;                 // [M][H][DH] gradient of the normalized query
    const float* dgr;                // [M][H] read gate (dy . o)
    const uint8_t* gw;
    bf16* dv;                        // the values' gradient, row r head h at dv + r * lddv + h * DH
    int lddv;
    const bf16 *k, *q;
    const float *kn, *qn, *rp1, *aw, *ar;
    const uint8_t *ridx, *reset;
    const float* R;
    float *dR, *dbk, *dbrg, *dbwg;
    // budget loss budget * (mean sigmoid(a) - target)^2 per gate group, the means from the forward's stats
    // (stats[0] / count: write gates, stats[2] / count: read gates; count = M * H)
    const float* stats;
    float budget, target, count;
    bf16* dG;  // [M][ldG], zeroed before
    int ldG, M, T, H;
};
void addr_bwd(const AddrBwd& a, cudaStream_t s);

// Hop-2 query: G2 [M][ldG2] = [q2_pre (H*DH) | read gate (H)]
struct Q2Fwd {
    const float* G2;
    int ldG2;
    const float* brg;
    bf16* q;
    float *qn, *ar;
    uint8_t* gr;
    float* stats;  // [2]: sum sigmoid(ar), open reads
    uint2* info;   // the layer's packed step words: the hop-2 read gate is OR-ed in (flag 8)
    int M, H;
};
void q2_fwd(const Q2Fwd& a, cudaStream_t s);
struct Q2Bwd {
    const float *dq, *dgr;
    const bf16* q;
    const float *qn, *ar;
    float* dbrg;
    const float* stats;  // the hop-2 read gates' stats (stats[0] / count is the mean sigmoid)
    float budget, target, count;
    bf16* dG2;
    int ldG2, M, H;
};
void q2_bwd(const Q2Bwd& a, cudaStream_t s);

// -- banks (kernels_bank.cu) --------------------------------------------------------------------------------------

// What a layer's banks are built from (shared by both reads). Per (row, head) one slot table: each slot holds the
// decayed sums K (DH), V (DH) and the mass Z of its writes; the read is softmax_m(tau q.K_m / Z_m) over the slots
// with Z_m > BANK_EPS, then sum_m p_m V_m / Z_m.
struct BankWrites {
    const bf16* k;  // [M][H][DH]
    const bf16* v;  // row r, head h at v + r * ldv + h * DH
    int ldv;
    const uint2* info;   // [M][H] routing, gates, reset, p1 (AddrFwd::info)
    const float* loga;   // [H][SLOTS]: log of each slot's decay per token
    int B, T, H;
};

struct BankRead {
    const bf16* q;          // [M][H][DH]
    const uint8_t* gr;      // [M][H]
    const float* log_tau;   // [H]
    int gate_bit;           // this read's gate in the step word's flags: 2 (hop 1) or 8 (hop 2)
    bf16* o;                // forward: [M][H*DH], g_r * read (closed reads: 0, not computed)
    const float* dy;        // backward: [M][H*DH] gradient of o
    float* dq;              // [M][H][DH]
    float* dg;              // [M][H] dy . read (open reads), 0 (closed)
    float* coef;            // scratch [3][M*H*SLOTS]: the read's local gradients on each slot's K, V, Z
    float* dlog_tau;        // [H] accumulated
};
void bank_fwd(const BankWrites& w, const BankRead& r, cudaStream_t s);
void bank_rewalk(const BankWrites& w, const BankRead& r, cudaStream_t s);

struct BankAdjoint {
    const bf16* q[2];
    const float* dy[2];
    const float* coef[2];
    int hops;                 // the reads' gates come from the step words (flags 2 and 8)
    float *dk2, *dv2, *dr2;   // see AddrBwd
};
void bank_adjoint(const BankWrites& w, const BankAdjoint& a, cudaStream_t s);

}  // namespace lbc
