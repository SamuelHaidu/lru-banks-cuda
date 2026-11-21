#pragma once
// A run's configuration: RUN_DIR/config.json, the only home of its hyperparameters (as the Python pipeline's
// runs/<name>/config.json). Sections:
//   arch_params  the model (key names follow the Python lru-banks config where the concept is the same)
//   data         dataset directory (metadata.json + the split files), batch size, epoch_passes, seed
//   training     optimizer, schedule (warmup-stable-decay), epochs, validation, checkpoints
//   runtime      kernel settings that do not change the math
// Unknown keys are rejected; every value is checked (shapes the kernels support).

#include <string>
#include <vector>

#include "json.hpp"
#include "model.hpp"

namespace lbc {

struct ArchCfg {
    int vocabulary_size = 32768;
    int dim = 512;               // the residual stream
    int embed_dim = 128;         // the factorized embedding's width (ignored when not factorized)
    bool factorized = true;      // ids -> E (V x embed_dim) -> W_in -> dim; readout rms(h) W_out E^T. Not
                                 // factorized: E is V x dim, no W_in / W_out
    bool tied_embeddings = true; // the readout uses E, or its own table E_out
    int num_layers = 4;
    std::vector<int> hops = {2, 2, 2, 2};  // per layer: 1 (read, FFN) or 2 (read, FFN, second read)
    int lru_dim = 256;           // complex channels (state 2 x lru_dim)
    int num_heads = 8;           // bank heads of head_dim (keys and values): 8 x 64 = 512 wide
    double ffn_mult = 4.0;       // FFN width = ffn_mult x dim
    std::vector<double> lru_half_life = {1.0, 16.0};     // LRU decays at init, log-spaced
    std::vector<double> slot_half_life = {1.0, 1024.0};  // fixed slot decays, log-spaced
    double tau = 8.0;            // read temperature at init
    // initial gate logit biases. Measured (RESEARCH_LOG): trained write gates (2) learn as well as open ones; trained
    // read gates (2) collapse closed without a budget (a closed read gets no gradient), and with gate_budget 10 /
    // target 0.5 they trade loss for speed early on. Default: reads held open (20); read gating is gate_bias_read 2
    // with a gate_budget.
    double gate_bias_write = 2.0, gate_bias_read = 20.0;
    double gate_budget = 0.0, gate_budget_target = 0.5;  // budget * (mean sigmoid(a) - target)^2 per gate group
    int window_size = 512;
    std::vector<int> bank_slots;  // per layer bank size, 2..128 (the kernels' maximum); [] = 128 everywhere. Fewer
                                  // slots change the model, not the step time (the walks still run 128 threads)
    int head_dim = 64;            // 32 or 64 (the kernels are compiled for both)
    int topk = 2;                 // fixed by the kernels (any other value is rejected)
};

struct DataCfg {
    std::string dataset;         // directory with metadata.json and the split files
    int batch_size = 64;
    double epoch_passes = 0.02;  // one epoch = this fraction of a pass over the train split (random windows)
    int seed = 42;               // data order, validation rows and the initialization
};

struct TrainCfg {
    double learning_rate = 2e-3;
    int warmup_steps = 100;      // linear warmup over the first updates
    int lr_decay_start = 0, lr_decay_steps = 0;  // warmup-stable-decay: linear to lr_min_ratio x (0 steps: none)
    double lr_min_ratio = 0.1;
    double weight_decay = 0.1, beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-7;
    double global_clipnorm = 1.0;  // 0 or null: off
    int epochs = 100;
    int validation_steps = 50;   // batches of validation rows at every epoch end (0: no validation)
    int checkpoint_every = 1;    // numbered checkpoints every N epochs (0: none); latest and best always
};

struct RuntimeCfg {
    int bank_chunk = 512;        // tokens per bank-walk block (longer rows walk in parallel chunks)
    int ce_chunk = 256;          // loss rows per chunk (256 keeps a chunk's logits in the 4090's L2)
    bool cuda_graph = true;      // the training step as one CUDA graph
};

struct RunConfig {
    ArchCfg arch;
    DataCfg data;
    TrainCfg training;
    RuntimeCfg runtime;

    json::Value to_json() const;
    static RunConfig from_json(const json::Value& v);  // unknown keys and invalid values throw std::runtime_error
    void validate() const;
    Config model_config() const;
    int ffn_dim() const;
};

RunConfig load_run_config(const std::string& run_dir);
void save_run_config(const std::string& run_dir, const RunConfig& rc);
// `dotted` = `raw` (parsed as JSON; a plain string when it is not JSON). The key must exist; the result is validated.
RunConfig with_key(const RunConfig& rc, const std::string& dotted, const std::string& raw);

// datasets/<name>/metadata.json, as written by the research repository's dataset builders
struct DatasetMeta {
    std::string dir, name;
    int vocabulary_size = 0, doc_token_id = 0;
    std::string train_file, validation_file;     // full paths
    double train_chars_per_token = 0.0, validation_chars_per_token = 0.0;  // 0: unknown (no bpc)
    static DatasetMeta load(const std::string& dir);
};

}  // namespace lbc
