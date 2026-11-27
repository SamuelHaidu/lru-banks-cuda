#pragma once
// Launchers of every custom kernel. Layout conventions: activations are row-major with one row per token
// (r = b * T + t, M = B * T rows); per-head tensors are [M][H][DH]; gates and routing are [M][H].

#include <cstddef>
#include <cstdint>
#include <numbers>

#include <cuda_runtime.h>

#include "common.hpp"

namespace lbc {

// -- norms and elementwise (kernels_basic.cu) ---------------------------------------------------------------------

// xh = h / rms(h) (bf16), rstd = 1 / rms(h). The gain is folded into the next GEMM's weight.
void rmsnorm_fwd(const float* h, bf16* xh, float* rstd, int M, int D, cudaStream_t s);
// dh (+)= rstd (dxh - xh * mean(dxh * xh)); also dhb = bf16(dh) (the next GEMMs' operand) and, if colsum is not
// null, colsum[c] += sum over the rows of the new dh (a bias gradient). D <= 1024.
void rmsnorm_bwd(const float* dxh,
                 const bf16* xh,
                 const float* rstd,
                 float* dh,
                 bf16* dhb,
                 float* colsum,
                 int M,
                 int D,
                 bool accumulate,
                 cudaStream_t s);
void to_bf16(const float* x, bf16* y, size_t n, cudaStream_t s);
void add_to_f32(const bf16* x, float* y, size_t n, cudaStream_t s);  // y += x
void embed_gather(const int* ids, const bf16* E, bf16* x, int M, int E0, cudaStream_t s);
void embed_gather_f32(const int* ids, const float* E, float* x, int M, int E0, cudaStream_t s);  // unfactorized: h
void embed_scatter(const int* ids, const float* dx, float* dE, int M, int E0, cudaStream_t s);
// y = bf16(x * g) per column (the unfactorized readout's norm gain), and its backward: dx = dy * g (written),
// dg += column sums of dy * x
void scale_cols(const bf16* x, const float* g, bf16* y, int M, int D, cudaStream_t s);
void scale_cols_bwd(const bf16* dy, const bf16* x, const float* g, float* dx, float* dg, int M, int D, cudaStream_t s);
// Rows of logits [rows][V]: loss_sum += sum (lse - logit[y]) * inv_n; correct_sum += [logit[y] is the row's max] *
// inv_n (the accuracy); dlogits = (softmax - onehot) * inv_n, with inv_n = 1 / targets read from the device (the
// step can be a CUDA graph). Rows with y < 0 add nothing. dlogits may be null (validation: no gradient).
void ce_rows(const float* logits,
             const int* y,
             bf16* dlogits,
             float* loss_sum,
             float* correct_sum,
             int rows,
             int V,
             const float* inv_n,
             cudaStream_t s,
             float* row_loss = nullptr,      // row_loss: per row (0 without target)
             const float* w = nullptr,       // per-row weight of the loss and gradient (null: 1)
             const float* lp_old = nullptr,  // with w: the weight is w * min(p / exp(lp_old), tis_cap) (TIS)
             float tis_cap = 2.f,
             float clip_low = 0.f,
             float clip_high = 0.f,      // PPO clip of the ratio (clip_high 0: off)
             const int* cand = nullptr,  // unlikelihood: [rows][nc] candidate tokens (-1 padded), or null
             int nc = 0,
             float ul = 0.f,            // its weight: ul * -sum_c log(1 - p_c) is added to each row's loss
             float* ul_sum = nullptr);  // += that term * inv_n (its mean, for the log)

// -- optimizer (kernels_basic.cu) ---------------------------------------------------------------------------------

// The learning rate of update `step` (0-based: updates done before it), as the Python pipeline's LinearWarmup:
// linear warmup to lr over `warmup` updates, then constant; with decay_steps > 0, from update decay_start a fall to
// min_ratio x lr over decay_steps updates, linear or (cosine) half a cosine, then min_ratio x lr.
struct Schedule {
    float lr{};
    int warmup{}, decay_start{}, decay_steps{};
    float min_ratio{};
    int cosine = 0;
    int cycles = 1;  // > 1: the decay repeats (warm restarts), each restart warming up from min_ratio over `warmup`
    int start = 0;   // the step the schedule counts from (a stage's first update)
    float warm_from = 0.f;  // the warmup's starting fraction (a later stage: the previous stage's final min_ratio)
};

__host__ __device__ inline float schedule_lr(const Schedule& s, int step) {
    step -= s.start;
    float frac = float(step + 1) / float(s.warmup > 1 ? s.warmup : 1);
    frac = s.warm_from + (1.f - s.warm_from) * (frac < 1.f ? frac : 1.f);
    if (s.decay_steps > 0) {
        int pos = step - s.decay_start;
        const int k = pos > 0 ? pos / s.decay_steps : 0;  // the cycle
        if (s.cycles > 1 && k >= 1 && k < s.cycles) {
            pos -= k * s.decay_steps;
            float const w = float(pos + 1) / float(s.warmup > 1 ? s.warmup : 1);
            frac = s.min_ratio + (1.f - s.min_ratio) * (w < 1.f ? w : 1.f);
        }
        float done = (s.cycles > 1 && k >= 1 && k < s.cycles) ? float(pos) / float(s.decay_steps)
                                                              : float(step - s.decay_start) / float(s.decay_steps);
        if (s.cycles > 1 && k >= s.cycles) { done = 1.f; }
        done = fminf(fmaxf(done, 0.f), 1.f);
        const float shape =
            (s.cosine != 0) ? 0.5f * (1.f + cosf(std::numbers::pi_v<float> * done)) : 1.f - done;  // 1 -> 0
        frac *= s.min_ratio + (1.f - s.min_ratio) * shape;
    }
    return s.lr * frac;
}

void sumsq(const float* x, size_t n, float* out, cudaStream_t s);
// AdamW with the update scaled by min(1, clip / ||g||) (||g||^2 read from gsq on the device: no host sync); the
// step count (already incremented: the bias corrections and the schedule) is read from the device as well, so the
// step can be a CUDA graph. The learning rate used is written to lr_out.
void adamw(float* p,
           const float* g,
           float* m,
           float* v,
           const uint8_t* decay,
           size_t n,
           const Schedule& sched,
           float beta1,
           float beta2,
           float eps,
           float wd,
           const int* step,
           const float* gsq,
           float clip,
           float* lr_out,
           cudaStream_t s);
void increment(int* x, cudaStream_t s);
// Muon (Model::muon_step), one block of rows x cols at row stride ld in the flat buffers: muon_prep updates the
// momentum (m = mu m + g, g clipped as adamw does) and writes the Nesterov update g + mu m into x (rows x cols, bf16)
// with ||x||^2 added to *norm; muon_normalize divides x by ||x||; muon_apply does p -= lr_scale lr (wd p + rms o), lr from the
// schedule at the device step. A non-finite gradient norm skips the update, as in adamw.
void muon_prep(const float* g,
               float* m,
               bf16* x,
               int rows,
               int cols,
               int ld,
               float mu,
               const float* gsq,
               float clip,
               float* norm,
               cudaStream_t s);
void muon_normalize(bf16* x, size_t n, const float* norm, cudaStream_t s);
void muon_apply(float* p,
                const bf16* o,
                int rows,
                int cols,
                int ld,
                const Schedule& sched,
                float wd,
                float rms,
                float lr_scale,
                const int* step,
                const float* gsq,
                cudaStream_t s);
// Wb = bf16(diag(gain) W) (gain may be null)
void prep_weight(const float* W, const float* gain, bf16* Wb, int rows, int cols, cudaStream_t s);
// From dWp = dL/d(diag(gain) W): dW += diag(gain) dWp, dgain += rowsum(W * dWp) (gain may be null: dW += dWp)
void unfold_grad(const float* dWp,
                 const float* W,
                 const float* gain,
                 float* dW,
                 float* dgain,
                 int rows,
                 int cols,
                 cudaStream_t s);

// -- LRU (kernels_lru_addr.cu) ------------------------------------------------------------------------------------

// h_t = lambda h_{t-1} + gamma (P_re + i P_im), restarting at resets; lambda = exp(-exp(nu) + i theta),
// gamma = sqrt(1 - |lambda|^2). P: columns [0, N) real, [N, 2N) imaginary, row stride ldp. s = [Re h, Im h].
// h0: [B][2N] each row's state before its first token (truncated BPTT: the previous batch's last state), or null
// (zero). The backward treats it as a constant: the decay gradients include its term, nothing flows into it.
void lru_fwd(const bf16* P,
             int ldp,
             const float* nu,
             const float* theta,
             const uint8_t* reset,
             float* s,
             bf16* sb,
             int B,
             int T,
             int N,
             cudaStream_t st,
             const float* h0 = nullptr);
// dP (bf16, row stride lddp) from ds; dnu, dtheta accumulated
void lru_bwd(const float* ds,
             const float* s,
             const bf16* P,
             int ldp,
             const float* nu,
             const float* theta,
             const uint8_t* reset,
             bf16* dP,
             int lddp,
             float* dnu,
             float* dtheta,
             int B,
             int T,
             int N,
             cudaStream_t st,
             const float* h0 = nullptr);
// truncated BPTT: h0[b] = s[b][T - 1] (fp32) and sb0[b] = bf16 of it (the first key's GEMM operand next step)
void lru_carry(const float* s, float* h0, bf16* sb0, int B, int T, int N, cudaStream_t st);

// -- addressing: keys, queries, routing, gates (kernels_lru_addr.cu) ----------------------------------------------

struct AddrFwd {
    const bf16* G{};  // [M][ldG]: [k_pre (H*DH) | q_pre (H*DH) | read gate (H) | write gate (H)], k_pre unshifted
    int ldG{};
    const float *bk{}, *brg{}, *bwg{};  // key bias [H*DH], gate biases [H]
    const float* R{};                   // router [H][DH][SLOTS]
    const uint8_t* reset{};
    bf16 *k{}, *q{};       // [M][H][DH], normalized
    float *kn{}, *qn{};    // [M][H] norms before normalizing
    uint8_t* ridx{};       // [M][H][2] the two slots
    float* rp1{};          // [M][H] weight of the first slot (the second gets 1 - p1)
    uint8_t *gw{}, *gr{};  // [M][H] binary write / read gates (no write without a previous state:
                           // at a reset or a row's first token the key would be normalize(b_k) alone)
    float *aw{}, *ar{};    // [M][H] gate logits
    float* stats{};        // [4]: sum sigmoid(aw), open writes, sum sigmoid(ar), open reads
    uint2* info{};         // [M][H] packed for the bank walks: x = m1 | m2 << 8 | flags << 16 (flags: 1 write
                           // gate, 2 read gate, 4 reset, 8 hop-2 read gate, set by q2_fwd), y = bits of p1
    int M{}, T{}, H{};
    int dh = 32;        // head size (32 or 64)
    int slots = SLOTS;  // the layer's bank size (2..SLOTS): routing picks among slots [0, slots); the
                        // others are never written, so the reads never see them
    // incremental decoding (one token per row, rows = active rollouts): the previous token's k_pre comes from
    // Gprev[map[r]] (row stride ldGprev) instead of row r - 1, and the token's position from tpos[r]
    const bf16* Gprev = nullptr;
    int ldGprev = 0;
    const int* map = nullptr;
    const int* tpos = nullptr;
    // truncated BPTT: G0 [B][ldG0] holds k_pre of each row's carried state (W_k s_{-1}, this step's weights), so a
    // row's first token has a previous state (its key, its write gate) unless it is a reset; null: it has none
    const bf16* G0 = nullptr;
    int ldG0 = 0;
};

void addr_fwd(const AddrFwd& a, cudaStream_t s);

struct AddrBwd {
    // from the bank adjoint, per (row, head) and routed slot s = 0, 1: dk2 / dv2 [M][H][2][DH] (written only when
    // the write gate is open), dr2 [M][H][2] the gradient of the slot's write weight
    const bf16 *dk2{}, *dv2{};
    const float* dr2{};
    const float* dq{};   // [M][H][DH] gradient of the normalized query
    const float* dgr{};  // [M][H] read gate (dy . o)
    const uint8_t* gw{};
    bf16* dv{};  // the values' gradient, row r head h at dv + r * lddv + h * DH
    int lddv{};
    const bf16 *k{}, *q{};
    const float *kn{}, *qn{}, *rp1{}, *aw{}, *ar{};
    const uint8_t *ridx{}, *reset{};
    const float* R{};
    float *dR{}, *dbk{}, *dbrg{}, *dbwg{};
    // budget loss budget * (mean sigmoid(a) - target)^2 per gate group, the means from the forward's stats
    // (stats[0] / count: write gates, stats[2] / count: read gates; count = M * H)
    const float* stats{};
    float budget{}, target{}, count{};
    bf16* dG{};  // [M][ldG] (every used column written; padding columns must be zero)
    int ldG{}, M{}, T{}, H{};
    // [M][H][2] each routed slot's mass just before the write (BankRead::zpre): a closed write gate's gradient is
    // the secant to the open state (see addr_bwd); null: the tangent at weight 0
    const float* zpre = nullptr;
    int dh = 32;
    // truncated BPTT (AddrFwd::G0 set): dG0 [B][H*DH], the gradient of each row's first k_pre (every entry written:
    // 0 at a reset); null: the first token has no previous state
    bf16* dG0 = nullptr;
};

void addr_bwd(const AddrBwd& a, cudaStream_t s);

// Hop-2 query: G2 [M][ldG2] = [q2_pre (H*DH) | read gate (H)]
struct Q2Fwd {
    const bf16* G2{};
    int ldG2{};
    const float* brg{};
    bf16* q{};
    float *qn{}, *ar{};
    uint8_t* gr{};
    float* stats{};  // [2]: sum sigmoid(ar), open reads
    uint2* info{};   // the layer's packed step words: the hop-2 read gate is OR-ed in (flag 8)
    int M{}, H{};
    int dh = 32;
};

void q2_fwd(const Q2Fwd& a, cudaStream_t s);

struct Q2Bwd {
    const float *dq{}, *dgr{};
    const bf16* q{};
    const float *qn{}, *ar{};
    float* dbrg{};
    const float* stats{};  // the hop-2 read gates' stats (stats[0] / count is the mean sigmoid)
    float budget{}, target{}, count{};
    bf16* dG2{};
    int ldG2{}, M{}, H{};
    int dh = 32;
};

void q2_bwd(const Q2Bwd& a, cudaStream_t s);

// -- banks (kernels_bank.cu) --------------------------------------------------------------------------------------

// What a layer's banks are built from (shared by both reads). Per (row, head) one slot table: each slot holds the
// decayed sums K (DH), V (DH) and the mass Z of its writes; the read is softmax_m(tau q.K_m / Z_m) over the slots
// with Z_m > BANK_EPS, then sum_m p_m V_m / Z_m.
struct BankWrites {
    const bf16* k{};  // [M][H][DH]
    const bf16* v{};  // row r, head h at v + r * ldv + h * DH
    int ldv{};
    const uint2* info{};  // [M][H] routing, gates, reset, p1 (AddrFwd::info)
    const float* loga{};  // [H][SLOTS]: log of each slot's decay per token
    int B{}, T{}, H{};
    // Chunks: the walks run one block per (row, head, chunk of `chunk` tokens). With more than one chunk per row the
    // forward and the re-walk start from per-chunk checkpoints of the slot states (bank_checkpoint), and the adjoint
    // runs in two passes around a sequential combine (exact: the adjoint is linear).
    int chunk{};
    const float* ckpt{};  // [B][H][chunks][ckpt_fields(dh)][SLOTS], or null with one chunk per row
    int dh = 32;          // head size (32 or 64)
    // truncated BPTT: [B][H][ckpt_fields(dh)][SLOTS] each row's slot states before its first token (the previous
    // batch's final states, times relative to this row's start), or null (empty slots)
    const float* init = nullptr;
};

inline int ckpt_fields(int dh) { return 2 * dh + 2; }  // Kb, Vb, Zs, tm (forward); LK, LV, LZ (adjoint)

__host__ __device__ inline int bank_chunks(const BankWrites& w) { return (w.T + w.chunk - 1) / w.chunk; }

// The slot states at every chunk start but the first: local chunk states in parallel (writes only), then a per-slot
// merge (local, rflags: scratch of the checkpoint's size and [B][H][chunks]).
void bank_checkpoint(const BankWrites& w, float* ckpt, float* local, int* rflags, cudaStream_t s);

struct BankRead {
    const bf16* q{};         // [M][H][DH]
    const uint8_t* gr{};     // [M][H]
    const float* log_tau{};  // [H]
    int gate_bit{};          // this read's gate in the step word's flags: 2 (hop 1) or 8 (hop 2)
    bf16* o{};               // forward: [M][H*DH], g_r * read (closed reads: 0, not computed)
    const float* dy{};       // backward: [M][H*DH] gradient of o
    float* dq{};             // [M][H][DH]
    float* dg{};             // [M][H] dy . read (open reads), 0 (closed)
    bf16* coef{};            // scratch [3][M*H*SLOTS]: the read's local gradients on each slot's K, V, Z
    float* dlog_tau{};       // [H] accumulated
    float* zpre = nullptr;   // forward (hop 1): [M][H][2] each routed slot's mass just before the write (for AddrBwd)
    // re-walk (truncated BPTT): [B][H][ckpt_fields(dh)][SLOTS] each row's slot states after its last token, times
    // shifted by -T (the next row's clock), written by the last chunk's blocks. May be BankWrites::init itself when a
    // row is one chunk (each block reads its record before writing it); with more chunks it must be another buffer.
    float* final_state = nullptr;
};

void bank_fwd(const BankWrites& w, const BankRead& r, cudaStream_t s);
void bank_rewalk(const BankWrites& w, const BankRead& r, cudaStream_t s);

struct BankAdjoint {
    const bf16* q[2];
    const float* dy[2];
    const bf16* coef[2];
    int hops;         // the reads' gates come from the step words (flags 2 and 8)
    bf16 *dk2, *dv2;  // see AddrBwd
    float* dr2;
    float* carry;  // with more than one chunk: [B][H][chunks][CKPT_FIELDS][SLOTS] scratch
    int* flags;    // [B][H][chunks] scratch
};

void bank_adjoint(const BankWrites& w, const BankAdjoint& a, cudaStream_t s);

// -- incremental decoding (kernels_lru_addr.cu, kernels_bank.cu, kernels_basic.cu) --------------------------------
// One token for each of R rows; row r's recurrent state lives in slot map[r] of the state buffers.
// LRU: state [slots][2N] fp32 (h_{t-1}, updated in place), sb [R][2N] bf16 (the W_s GEMM's operand)
void lru_step(const bf16* P,
              int ldp,
              const float* nu,
              const float* theta,
              const uint8_t* reset,
              const int* map,
              float* state,
              bf16* sb,
              int R,
              int N,
              cudaStream_t st);
// Gprev[map[r]][0, cols) = G[r][0, cols) (the keys' k_pre for the next token)
void store_rows(const bf16* G, int ldG, int cols, const int* map, bf16* Gprev, int ldGprev, int R, cudaStream_t st);
// the banks of one layer for one token: state [slots][H][ckpt_fields(dh)][SLOTS] (Kb, Vb, Zs, tm as bank_fwd);
// write: apply the token's write (and reset) first, then the read r.gate_bit (2 hop 1, 8 hop 2) into r.o. w.B = R,
// w.T = 1, the step words / k / v indexed by r; positions tpos[r].
void bank_step(const BankWrites& w,
               const BankRead& r,
               const int* map,
               const int* tpos,
               float* state,
               bool write,
               cudaStream_t s);
// per row: sample from softmax(logits / temperature) (Gumbel-max with a counter-based hash of key[r] and the token
// id; temperature <= 0: argmax), its log-probability under that distribution and the distribution's entropy
void sample_rows(const float* logits,
                 int R,
                 int V,
                 float temperature,
                 const uint64_t* key,
                 int* tok,
                 float* logp,
                 float* ent,
                 cudaStream_t s);
// copy rows of `bytes` (a multiple of 16) inside buf: dst row pairs[i].x <- src row pairs[i].y
void copy_rows(void* buf, size_t bytes, const int2* pairs, int n, cudaStream_t s);

}  // namespace lbc
