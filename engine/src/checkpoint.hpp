#pragma once
// Checkpoints: the weights and the optimizer state (Adam moments and step count), with the run's config, as one file
//   "LBCKPT01" | uint64 header length | header (JSON) | zero padding to 256 bytes | params | Adam m | Adam v
// (float32, n_params each, in the model's flat layout; the header lists the tensors: name, offset, rows, cols).
// Written to PATH.tmp and renamed, so a crash never leaves a partial checkpoint behind.

#include <string>

#include <cuda_runtime.h>

#include "json.hpp"
#include "model.hpp"

namespace lbc {

struct CheckpointInfo {
    int epoch = 0;              // epochs completed
    long long step = 0;         // updates done
    double best_val_loss = -1;  // < 0: none yet
    json::Value config;         // the run config it was trained with
};

void save_checkpoint(const std::string& path,
                     Model& model,
                     const json::Value& config,
                     int epoch,
                     double best_val_loss,
                     cudaStream_t s);
CheckpointInfo read_checkpoint_info(const std::string& path);
// parameters, moments and step count into `model` (the tensor list must match), then the bf16 weight copies
void load_checkpoint(const std::string& path, Model& model, cudaStream_t s);
// the parameters only (the tensor list must match): a new run from trained weights, fresh optimizer, step 0
void load_weights(const std::string& path, Model& model, cudaStream_t s);

}  // namespace lbc
