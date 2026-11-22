#pragma once
// Incremental decoding: the model one token at a time for many rows, each with its own recurrent state (per layer
// the LRU state, the previous token's key projection and the bank slots), in the training forward's arithmetic (the
// same kernels' math, the same bf16 roundings), so a rollout's log-probabilities match what training computes.
//
// Rows are addressed through slots: row r of a step reads and writes state slot map[r], so finished rows simply stop
// being listed (no state moves), and a group's rows start as copies of one prefilled slot (copy_slots).

#include <cstdint>
#include <vector>

#include <cuda_runtime.h>

#include "model.hpp"

namespace lbc {

class Decoder {
  public:
    Decoder(Model& model, int slots, int max_rows);  // state for `slots` rollouts, at most max_rows per step
    ~Decoder();
    Decoder(const Decoder&) = delete;
    Decoder& operator=(const Decoder&) = delete;
    Decoder(Decoder&&) = delete;
    Decoder& operator=(Decoder&&) = delete;

    // One token for R rows (host arrays): ids, slot, position (0 at the row's first token), reset (the token is the
    // document marker: the state restarts). Leaves the next-token logits on the device (logits(), R x V, when
    // R <= logit_rows()) and, when `key` is given, samples: tok / logp / ent (host arrays of R).
    void step(const int* ids,
              const int* slot,
              const int* tpos,
              const uint8_t* reset,
              int R,
              float temperature,
              const uint64_t* key,
              int* tok,
              float* logp,
              float* ent,
              cudaStream_t s);
    void copy_slots(const std::vector<int2>& pairs, cudaStream_t s);  // slot pairs (dst, src)

    const float* logits() const { return logits_; }

    int logit_rows() const { return logit_rows_; }

    int slots() const { return slots_; }

    size_t state_bytes() const;

  private:
    template <class T>
    T* alloc(size_t n);
    Model& m_;
    int slots_, max_rows_, logit_rows_;
    std::vector<void*> allocs_;
    // state per layer (slots_ + 1 rows: the last is the padding rows' scratch)
    std::vector<float*> lru_, bank_;
    std::vector<bf16*> gprev_;
    // per-step buffers (max_rows_ rows)
    int *d_ids_, *d_map_, *d_tpos_, *d_tok_;
    uint8_t* d_reset_;
    uint64_t* d_key_;
    float *d_logp_, *d_ent_;
    int2* d_pairs_;
    int pairs_cap_ = 0;
    float *h_, *rstd_, *kn_, *qn_, *rp1_, *aw_, *ar_, *q2n_, *ar2_, *stats_, *logits_;
    bf16 *x0_, *xh_, *PV_, *sb_, *G_, *k_, *q_, *o_, *U_, *Z_, *G2_, *q2_, *hr_;
    uint8_t *ridx_, *gw_, *gr_, *gr2_;
    uint2* info_;
    // pinned host staging
    int* h_in_ = nullptr;
    int* h_out_ = nullptr;
    float* h_outf_ = nullptr;
};

}  // namespace lbc
