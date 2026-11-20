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
// dh (+)= rstd (dxh - xh * mean(dxh * xh)); also dhb = bf16(dh) (the next GEMMs' operand) and, if colsum is not
// null, colsum[c] += sum over the rows of the new dh (a bias gradient). D <= 1024.
void rmsnorm_bwd(const float* dxh, const bf16* xh, const float* rstd, float* dh, bf16* dhb, float* colsum, int M,
                 int D, bool accumulate, cudaStream_t s);
void to_bf16(const float* x, bf16* y, size_t n, cudaStream_t s);
void add_to_f32(const bf16* x, float* y, size_t n, cudaStream_t s);  // y += x
void embed_gather(const int* ids, const bf16* E, bf16* x, int M, int E0, cudaStream_t s);
void embed_gather_f32(const int* ids, const float* E, float* x, int M, int E0, cudaStream_t s);  // unfactorized: h
void embed_scatter(const int* ids, const float* dx, float* dE, int M, int E0, cudaStream_t s);
// y = bf16(x * g) per column (the unfactorized readout's norm gain), and its backward: dx = dy * g (written),
// dg += column sums of dy * x
void scale_cols(const bf16* x, const float* g, bf16* y, int M, int D, cudaStream_t s);
void scale_cols_bwd(const bf16* dy, const bf16* x, const float* g, float* dx, float* dg, int M, int D,
                    cudaStream_t s);
// Rows of logits [rows][V]: loss_sum += sum (lse - logit[y]) * inv_n; correct_sum += [logit[y] is the row's max] *
// inv_n (the accuracy); dlogits = (softmax - onehot) * inv_n, with inv_n = 1 / targets read from the device (the
// step can be a CUDA graph). Rows with y < 0 add nothing. dlogits may be null (validation: no gradient).
void ce_rows(const float* logits, const int* y, bf16* dlogits, float* loss_sum, float* correct_sum, int rows, int V,
             const float* inv_n, cudaStream_t s);

// -- optimizer (kernels_basic.cu) ---------------------------------------------------------------------------------

// The learning rate of update `step` (0-based: updates done before it), as the Python pipeline's LinearWarmup:
// linear warmup to lr over `warmup` updates, then constant; with decay_steps > 0, from update decay_start a linear
// fall to min_ratio x lr over decay_steps updates.
struct Schedule {
    float lr;
    int warmup, decay_start, decay_steps;
    float min_ratio;
};
__host__ __device__ inline float schedule_lr(const Schedule& s, int step) {
    float frac = float(step + 1) / float(s.warmup > 1 ? s.warmup : 1);
    frac = frac < 1.f ? frac : 1.f;
    if (s.decay_steps > 0) {
        float done = float(step - s.decay_start) / float(s.decay_steps);
        done = done < 0.f ? 0.f : (done > 1.f ? 1.f : done);
        frac *= 1.f - (1.f - s.min_ratio) * done;
    }
    return s.lr * frac;
}

void sumsq(const float* x, size_t n, float* out, cudaStream_t s);
// AdamW with the update scaled by min(1, clip / ||g||) (||g||^2 read from gsq on the device: no host sync); the
// step count (already incremented: the bias corrections and the schedule) is read from the device as well, so the
// step can be a CUDA graph. The learning rate used is written to lr_out.
void adamw(float* p, const float* g, float* m, float* v, const uint8_t* decay, size_t n, const Schedule& sched,
           float beta1, float beta2, float eps, float wd, const int* step, const float* gsq, float clip,
           float* lr_out, cudaStream_t s);
void increment(int* x, cudaStream_t s);
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
    const bf16* G;   // [M][ldG]: [k_pre (H*DH) | q_pre (H*DH) | read gate (H) | write gate (H)], k_pre unshifted
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
    const bf16 *dk2, *dv2;
    const float* dr2;
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
    bf16* dG;  // [M][ldG] (every used column written; padding columns must be zero)
    int ldG, M, T, H;
    // [M][H][2] each routed slot's mass just before the write (BankRead::zpre): a closed write gate's gradient is
    // the secant to the open state (see addr_bwd); null: the tangent at weight 0
    const float* zpre = nullptr;
};
void addr_bwd(const AddrBwd& a, cudaStream_t s);

// Hop-2 query: G2 [M][ldG2] = [q2_pre (H*DH) | read gate (H)]
struct Q2Fwd {
    const bf16* G2;
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
    // Chunks: the walks run one block per (row, head, chunk of `chunk` tokens). With more than one chunk per row the
    // forward and the re-walk start from per-chunk checkpoints of the slot states (bank_checkpoint), and the adjoint
    // runs in two passes around a sequential combine (exact: the adjoint is linear).
    int chunk;
    const float* ckpt;   // [B][H][chunks][CKPT_FIELDS][SLOTS], or null with one chunk per row
};
constexpr int CKPT_FIELDS = 2 * DH + 2;  // Kb, Vb, Zs, tm (forward); LK, LV, LZ (adjoint)
__host__ __device__ inline int bank_chunks(const BankWrites& w) { return (w.T + w.chunk - 1) / w.chunk; }
// The slot states at every chunk start but the first: local chunk states in parallel (writes only), then a per-slot
// merge (local, rflags: scratch of the checkpoint's size and [B][H][chunks]).
void bank_checkpoint(const BankWrites& w, float* ckpt, float* local, int* rflags, cudaStream_t s);

struct BankRead {
    const bf16* q;          // [M][H][DH]
    const uint8_t* gr;      // [M][H]
    const float* log_tau;   // [H]
    int gate_bit;           // this read's gate in the step word's flags: 2 (hop 1) or 8 (hop 2)
    bf16* o;                // forward: [M][H*DH], g_r * read (closed reads: 0, not computed)
    const float* dy;        // backward: [M][H*DH] gradient of o
    float* dq;              // [M][H][DH]
    float* dg;              // [M][H] dy . read (open reads), 0 (closed)
    bf16* coef;             // scratch [3][M*H*SLOTS]: the read's local gradients on each slot's K, V, Z
    float* dlog_tau;        // [H] accumulated
    float* zpre = nullptr;  // forward (hop 1): [M][H][2] each routed slot's mass just before the write (for AddrBwd)
};
void bank_fwd(const BankWrites& w, const BankRead& r, cudaStream_t s);
void bank_rewalk(const BankWrites& w, const BankRead& r, cudaStream_t s);

struct BankAdjoint {
    const bf16* q[2];
    const float* dy[2];
    const bf16* coef[2];
    int hops;                 // the reads' gates come from the step words (flags 2 and 8)
    bf16 *dk2, *dv2;          // see AddrBwd
    float* dr2;
    float* carry;             // with more than one chunk: [B][H][chunks][CKPT_FIELDS][SLOTS] scratch
    int* flags;               // [B][H][chunks] scratch
};
void bank_adjoint(const BankWrites& w, const BankAdjoint& a, cudaStream_t s);

}  // namespace lbc
