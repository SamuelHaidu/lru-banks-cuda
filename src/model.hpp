#pragma once
// The model: lru-banks with binary read / write gates, one FFN per layer, hand-written forward and backward.
//
//   h = E[ids] W_in                                   (embedding E: V x E0, stream width D; not factorized:
//                                                     E is V x D and h = E[ids])
//   per layer:
//     x = rms(h);  [P | v] = x W_a                    P: LRU input (2N), v: bank values (H x DH)
//     s_t = LRU(P), complex diagonal, no input gate   s = [Re h, Im h] (2N)
//     [k_pre | q_pre | a_r | a_w] = s W_s             key from s_{t-1} (+ b_k), query from s_t, gate logits from s_t
//     write gate g_w = [a_w > 0] (never at <bos> or a row's first token: no previous state, the key would be
//     normalize(b_k) alone), read gate g_r = [a_r > 0], one per head, straight-through
//     banks: top-2 routing of the key into SLOTS slots per head, mass-normalized, fixed per-slot decays
//     h += W_o (g_r * read(q))                        hop 1 (closed reads are not computed)
//     h += W_2 gelu(W_1 rms(h) + b_1) + b_2           the layer's one FFN, between the hops
//     q2 = normalize(rms(h) W_q2), a_r2 likewise;  h += W_o2 (g_r2 * read(q2))      hop 2, same banks (2-hop
//                                                     layers only)
//   logits = rms(h) W_out E^T (not factorized: (rms(h) * g) E^T; untied: E_out instead of E)
//   loss = mean cross-entropy over every position with a target
//
// RMSNorm gains are folded into the next GEMM's bf16 weight (diag(g) W); their gradients are unfolded after the
// weight-gradient GEMM. Optional budget loss on the gates: budget * (mean sigmoid(a) - target)^2 per gate group.

#include <cstdint>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "common.hpp"
#include "gemm.hpp"
#include "kernels.hpp"

namespace lbc {

struct Config {
    int V = 32768, E0 = 128, D = 512, L = 4, N = 256, H = 16, F = 2048;
    bool factorized = true;  // false: E is V x D (E0 = D), no W_in / W_out
    bool tied = true;        // false: the readout has its own table E_out (V x E0)
    std::vector<int> hops;   // per layer, 1 or 2 (empty: 2 everywhere)
    std::vector<int> slots;  // per layer bank size, 2..SLOTS (empty: SLOTS everywhere)
    int B = 64, T = 512;
    int bank_chunk = 512;  // tokens per bank-walk block (rows longer than this walk in parallel chunks)
    int ce_chunk = 256;  // loss rows per chunk: 256 x V fp32 logits + bf16 gradients (50 MB) stay in the L2
    float lr = 2e-3f, wd = 0.1f, beta1 = 0.9f, beta2 = 0.999f, eps = 1e-7f, clip = 1.f;
    int warmup = 0, decay_start = 0, decay_steps = 0;  // the schedule (kernels.hpp: schedule_lr)
    float min_ratio = 0.1f;
    float tau = 8.f, lru_hl_min = 1.f, lru_hl_max = 16.f, slot_hl_min = 1.f, slot_hl_max = 1024.f;
    float gate_bias_w = 2.f, gate_bias_r = 2.f;  // initial gate logit biases (how many gates start open)
    float budget = 0.f, budget_target = 0.5f;

    int M() const { return B * T; }
    int hops_of(int n) const { return hops.empty() ? 2 : hops[size_t(n)]; }
    int slots_of(int n) const { return slots.empty() ? SLOTS : slots[size_t(n)]; }
    int HD() const { return H * DH; }
    int S2() const { return 2 * N; }
    int ldPV() const { return 2 * N + H * DH; }
    // GEMM output strides rounded up to 8 columns (16-byte rows in bf16); the padding columns stay 0
    int ldG() const { return (2 * H * DH + 2 * H + 7) / 8 * 8; }
    int ldG2() const { return (H * DH + H + 7) / 8 * 8; }
};

struct Tensor {
    std::string name;
    size_t off;
    int rows, cols;
    bool decay;
};

// One step's numbers, decoded from the device metrics record (Model::metrics_device)
struct StepMetrics {
    float loss, accuracy, grad_norm, lr;
    float open_w, open_r1, open_r2;  // open fraction of the write / hop-1 / hop-2 read gates, mean over the layers
                                     // (open_r2: over the 2-hop layers; -1 without any)
};

// device pointers (fixed across steps, so a step can be a CUDA graph); inv_n = 1 / targets, from the loader
struct Batch {
    const int* ids;
    const int* y;
    const uint8_t* reset;
    const float* inv_n;
};

class Model {
  public:
    explicit Model(const Config& c, uint64_t seed = 1);
    ~Model();
    Model(const Model&) = delete;
    Model& operator=(const Model&) = delete;

    void forward_backward(const Batch& batch, cudaStream_t s);  // loss and every gradient (gradients zeroed first)
    void optimizer_step(cudaStream_t s);                        // the step count lives on the device
    void forward_loss(const Batch& batch, cudaStream_t s);      // loss and accuracy only (validation)

    const float* loss_device() const { return loss_; }
    // [loss, accuracy, gate stats (6 per layer), ||g||^2, lr]: one contiguous record, so a step's numbers reach the
    // host in one asynchronous copy. forward_backward / forward_loss write the first 2 + 6 L, optimizer_step the rest.
    const float* metrics_device() const { return metrics_; }
    int metrics_count() const { return 4 + 6 * c_.L; }
    StepMetrics decode_metrics(const float* record) const;
    std::vector<float> gate_stats(cudaStream_t s);  // per layer: open fraction of [write, read 1, read 2]
    const Config& config() const { return c_; }
    const std::vector<Tensor>& tensors() const { return tensors_; }
    size_t num_params() const { return n_; }
    float* params() { return P_; }
    float* grads() { return G_; }
    float* adam_m() { return M1_; }
    float* adam_v() { return M2_; }
    int* step_device() { return step_; }  // updates done (the Adam bias corrections and the schedule)
    size_t embedding_params() const;      // E (and E_out)
    void prepare_weights(cudaStream_t s);  // bf16 copies with folded gains (after any change of the parameters)

    // per-category GPU time of one step (events around every launch); off by default
    void set_profile(bool on) { profile_ = on; }
    std::vector<std::pair<std::string, float>> profile_ms();

    // discrete decisions of layer n (tests replay them): ridx [M][H][2], gw, gr, gr2 [M][H]
    void decisions(int n, std::vector<uint8_t>& ridx, std::vector<uint8_t>& gw, std::vector<uint8_t>& gr,
                   std::vector<uint8_t>& gr2, cudaStream_t s);

  private:
    struct LayerIdx {  // tensor indices (-1: absent, the hop-2 tensors of a 1-hop layer)
        int norm1, Wa, nu, theta, Ws, bk, brg, bwg, R, ltau, ltau2 = -1, Wo, Wo2 = -1, ffn_norm, W1, b1, W2, b2,
            q2_norm = -1, Wq2 = -1, brg2 = -1;
    };
    struct LayerW {  // bf16 GEMM weights (and the FFN's first bias: the GELU epilogue takes it in bf16)
        bf16 *Wa, *Ws, *Wo, *Wo2, *W1, *W2, *Wq2, *b1;
    };
    struct LayerA {  // what the backward needs
        bf16 *xh1, *PV, *sb, *k, *q, *o1, *xh2, *Z, *U, *xh3, *q2, *o2;
        float *rstd1, *s, *kn, *qn, *rp1, *aw, *ar, *rstd2, *rstd3, *q2n, *ar2;
        uint8_t *ridx, *gw, *gr, *gr2;
        uint2* info;
        float* zpre;  // [M][H][2] the routed slots' mass before each write (the closed write gates' gradient)
        float* ckpt;  // slot states at chunk starts (rows longer than bank_chunk), shared by forward and backward
    };

    int add(const std::string& name, int rows, int cols, bool decay);
    float* p(int i) { return P_ + tensors_[i].off; }
    float* g(int i) { return G_ + tensors_[i].off; }
    template <class T>
    T* alloc(size_t n);
    void init(uint64_t seed);
    void layer_fwd(int n, const Batch& b, cudaStream_t s);
    void layer_bwd(int n, const Batch& b, cudaStream_t s);
    void mm(cudaStream_t s, bool tA, bool tB, int M, int N, int K, const bf16* A, int lda, const bf16* B, int ldb,
            void* D, int ldd, Out out, float beta = 0.f, const Epilogue& epi = {});
    struct BankWrites writes(int n, const Batch& b);
    void tic(cudaStream_t s, const char* cat);
    void toc(cudaStream_t s);

    Config c_;
    Gemm gemm_;
    std::vector<Tensor> tensors_;
    size_t n_ = 0;
    float *P_ = nullptr, *G_ = nullptr, *M1_ = nullptr, *M2_ = nullptr;
    uint8_t* decay_ = nullptr;
    int E_, E_out_ = -1, W_in_ = -1, final_norm_, W_out_ = -1;
    std::vector<LayerIdx> li_;
    std::vector<LayerW> lw_;
    std::vector<LayerA> la_;
    bf16 *Eb_ = nullptr, *Eoutb_ = nullptr, *Winb_ = nullptr, *Woutb_ = nullptr;
    const bf16* readout_table() const { return c_.tied ? Eb_ : Eoutb_; }
    void embed_fwd(const Batch& b, cudaStream_t s);
    void readout_loss(const Batch& b, bool backward, cudaStream_t s);
    float* b2_colsum(int n);  // layer n's b2 gradient when it is a 1-hop layer (else null)
    Schedule schedule() const { return Schedule{c_.lr, c_.warmup, c_.decay_start, c_.decay_steps, c_.min_ratio}; }
    float* loga_;

    // forward buffers
    float* h_;
    bf16 *x0_ = nullptr, *xhF_, *hr_;
    float *rstdF_, *logits_;
    bf16 *Gt_, *G2t_;
    bf16* dlog_;
    // backward buffers
    bf16 *dk2_, *dv2_;
    float *dh_, *dxh_, *dO1_, *dO2_, *dS_, *dr2_, *dq1_, *dq2_, *dgr1_, *dgr2_, *dx0_ = nullptr, *dWp_;
    bf16 *db1b_, *coef1_, *coef2_, *dhrb_, *dhb_, *dZ_, *dG_, *dG2_, *dPV_;
    float *metrics_, *loss_, *acc_, *stats_, *gsq_, *lr_, *carry_;
    int* step_;
    int* cflags_;

    std::vector<void*> allocations_;
    bool profile_ = false;
    struct Mark {
        std::string cat;
        cudaEvent_t a, b;
    };
    std::vector<Mark> marks_;
};

}  // namespace lbc
