// lbc: create, configure and train a run, the way the research repository's `model` CLI does.
//
//   lbc create RUN_DIR --dataset DATASET_DIR [section.key=value ...]   writes RUN_DIR/config.json
//   lbc config RUN_DIR [section.key value]                             prints it, or sets one key (JSON value)
//   lbc train  RUN_DIR [--epochs N]                                    trains, or resumes from RUN_DIR/model.lbc
//   lbc eval   RUN_DIR [--checkpoint best|latest|PATH] [--split FILE] [--steps N]
//                                                                      loss / accuracy on fixed rows of a token file
//
// One epoch is `data.epoch_passes` of a pass over the train split in shuffled random windows (steps_per_epoch
// updates). After every epoch: the validation loss on fixed rows, then
//   RUN_DIR/model.lbc                            the latest checkpoint (the resume point; weights + Adam state)
//   RUN_DIR/checkpoints/checkpoint_NNNNN.lbc     every training.checkpoint_every epochs
//   RUN_DIR/checkpoints/best.lbc                 the lowest val_loss so far
//   RUN_DIR/metrics.csv                          one row per epoch (the Python pipeline's columns first)
// The progress bar and the epoch lines follow Keras's (stdout; scripts/train.sh appends them to RUN_DIR/train.log).

#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "../src/checkpoint.hpp"
#include "../src/data.hpp"
#include "../src/json.hpp"
#include "../src/model.hpp"
#include "../src/run_config.hpp"
#include "lbc/check.hpp"

using namespace lbc;
namespace fs = std::filesystem;

namespace {

using Clock = std::chrono::steady_clock;
double seconds_since(Clock::time_point t) { return std::chrono::duration<double>(Clock::now() - t).count(); }

[[noreturn]] void die(const std::string& msg) {
    std::fprintf(stderr, "lbc: %s\n", msg.c_str());
    std::exit(1);
}

void usage() {
    std::fprintf(stderr,
                 "usage:\n"
                 "  lbc create RUN_DIR --dataset DATASET_DIR [section.key=value ...]\n"
                 "  lbc config RUN_DIR [section.key value]\n"
                 "  lbc train  RUN_DIR [--epochs N]\n"
                 "  lbc eval   RUN_DIR [--checkpoint best|latest|PATH] [--split FILE] [--steps N] [--rows N]\n"
                 "             [--silence L:H,...]\n"
                 "  lbc dump   RUN_DIR [--checkpoint best|latest|PATH] [--split FILE] [--rows N] [--out DIR]\n"
                 "             [--silence L:H,...]\n");
    std::exit(2);
}

std::string with_commas(unsigned long long n) {
    std::string s = std::to_string(n);
    for (int i = int(s.size()) - 3; i > 0; i -= 3) s.insert(size_t(i), ",");
    return s;
}

// "--silence 0:2,3:1,1:ffn,2:all" (layer:1|2|ffn|all) -> per-layer bit masks (bit 1: hop 1, 2: hop 2, 4: the FFN)
std::vector<int> parse_silence(const std::string& spec, int L) {
    std::vector<int> mask(size_t(L), 0);
    std::stringstream ss(spec);
    std::string item;
    while (std::getline(ss, item, ',')) {
        const size_t c = item.find(':');
        if (c == std::string::npos) die("--silence takes layer:hop pairs, e.g. 0:2,3:1");
        const int l = std::atoi(item.substr(0, c).c_str());
        const std::string w = item.substr(c + 1);
        const int bits = w == "1" ? 1 : w == "2" ? 2 : w == "ffn" ? 4 : w == "all" ? 7 : 0;
        if (l < 0 || l >= L || !bits) die("--silence: no part " + item + " (layer:1|2|ffn|all)");
        mask[size_t(l)] |= bits;
    }
    return mask;
}

// -- create / config ----------------------------------------------------------------------------------------------

int cmd_create(int argc, char** argv) {
    if (argc < 1) usage();
    const std::string run = argv[0];
    std::string dataset;
    std::vector<std::pair<std::string, std::string>> sets;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--dataset" && i + 1 < argc) {
            dataset = argv[++i];
        } else if (const size_t eq = a.find('='); eq != std::string::npos) {
            sets.emplace_back(a.substr(0, eq), a.substr(eq + 1));
        } else {
            usage();
        }
    }
    if (dataset.empty()) die("create needs --dataset DATASET_DIR (with metadata.json)");
    if (fs::exists(run + "/config.json")) die(run + "/config.json exists: edit it with `lbc config`");
    const DatasetMeta meta = DatasetMeta::load(dataset);
    RunConfig rc;
    rc.data.dataset = fs::absolute(dataset).lexically_normal().string();
    rc.arch.vocabulary_size = meta.vocabulary_size;
    rc = with_keys(rc, sets);  // validated together (e.g. num_layers with its hops)
    save_run_config(run, rc);
    std::printf("%s\ncreated %s/config.json\n", json::dump(rc.to_json()).c_str(), run.c_str());
    return 0;
}

int cmd_config(int argc, char** argv) {
    if (argc != 1 && argc != 3) usage();
    const std::string run = argv[0];
    RunConfig rc = load_run_config(run);
    if (argc == 3) {
        rc = with_key(rc, argv[1], argv[2]);
        save_run_config(run, rc);
    }
    std::printf("%s\n", json::dump(rc.to_json()).c_str());
    return 0;
}

// -- the progress bar (Keras's layout) ----------------------------------------------------------------------------

std::string fmt_value(double v) {
    char b[32];
    std::snprintf(b, sizeof b, std::fabs(v) >= 1e-3 || v == 0 ? "%.4f" : "%.4e", v);
    return b;
}

std::string fmt_eta(double s) {
    char b[32];
    const long t = long(s + 0.5);
    if (t >= 3600)
        std::snprintf(b, sizeof b, "%ld:%02ld:%02ld", t / 3600, (t % 3600) / 60, t % 60);
    else if (t >= 60)
        std::snprintf(b, sizeof b, "%ld:%02ld", t / 60, t % 60);
    else
        std::snprintf(b, sizeof b, "%lds", t);
    return b;
}

std::string fmt_per_step(double sec) {
    char b[32];
    if (sec >= 1)
        std::snprintf(b, sizeof b, "%.0fs/step", sec);
    else if (sec >= 1e-3)
        std::snprintf(b, sizeof b, "%.0fms/step", sec * 1e3);
    else
        std::snprintf(b, sizeof b, "%.0fus/step", sec * 1e6);
    return b;
}

class Progress {
  public:
    explicit Progress(int total) : total_(total) {}
    // `done` steps of `total`; `time` is the ETA (in progress) or the epoch's time (final)
    void draw(int done, const std::string& time, double per_step, const std::string& values, bool final) {
        const int width = int(std::to_string(total_).size());
        const int fill = done >= total_ ? 20 : 20 * done / total_;
        std::string bar;
        for (int k = 0; k < fill; ++k) bar += "━";
        std::string rest;
        for (int k = fill; k < 20; ++k) rest += "━";
        char head[64];
        std::snprintf(head, sizeof head, "%*d/%d", width, done, total_);
        std::string line = std::string("\x1b[1m") + head + "\x1b[0m \x1b[32m" + bar + "\x1b[0m\x1b[37m" + rest +
                           "\x1b[0m \x1b[1m" + time + "\x1b[0m " + fmt_per_step(per_step) + values;
        const size_t len = line.size();
        if (len < last_len_) line += std::string(last_len_ - len, ' ');
        last_len_ = len;
        std::fputs(("\r" + line + (final ? "\n" : "")).c_str(), stdout);
        std::fflush(stdout);
    }

  private:
    int total_;
    size_t last_len_ = 0;
};

// -- train ----------------------------------------------------------------------------------------------------------

struct EpochSums {
    int n = 0;
    double loss = 0, acc = 0, grad_norm = 0, open_w = 0, open_r1 = 0, open_r2 = 0;
    StepMetrics last{};
    void add(const StepMetrics& m) {
        ++n;
        loss += m.loss;
        acc += m.accuracy;
        grad_norm += m.grad_norm;
        open_w += m.open_w;
        open_r1 += m.open_r1;
        open_r2 += m.open_r2;
        last = m;
    }
};

std::string open_str(double w, double r1, double r2) {
    char b[64];
    if (r2 < 0)
        std::snprintf(b, sizeof b, "%.2f/%.2f/-", w, r1);
    else
        std::snprintf(b, sizeof b, "%.2f/%.2f/%.2f", w, r1, r2);
    return b;
}

// epochs completed according to metrics.csv (its last row's 0-based epoch + 1; 0 when missing or empty)
int csv_epochs(const std::string& path) {
    std::ifstream f(path);
    std::string line, last;
    int rows = 0;
    while (std::getline(f, line))
        if (!line.empty()) {
            last = line;
            ++rows;
        }
    if (rows < 2) return 0;
    return std::atoi(last.substr(0, last.find(',')).c_str()) + 1;
}

void copy_atomic(const std::string& from, const std::string& to) {
    fs::create_directories(fs::path(to).parent_path());
    fs::copy_file(from, to + ".tmp", fs::copy_options::overwrite_existing);
    fs::rename(to + ".tmp", to);
}

json::Value arch_without_window(const json::Value& config) {
    json::Value a = config.find("arch_params") ? *config.find("arch_params") : json::Value::object();
    if (json::Value* w = a.find("window_size")) *w = json::Value();  // a resumed run may change its window
    return a;
}

int cmd_train(int argc, char** argv) {
    if (argc < 1) usage();
    const std::string run = argv[0];
    int epochs_override = 0;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--epochs" && i + 1 < argc)
            epochs_override = std::atoi(argv[++i]);
        else
            usage();
    }
    const RunConfig rc = load_run_config(run);
    if (rc.data.dataset.empty()) die("data.dataset is empty (lbc config " + run + " data.dataset DIR)");
    const DatasetMeta meta = DatasetMeta::load(rc.data.dataset);
    if (meta.vocabulary_size != rc.arch.vocabulary_size)
        die("arch_params.vocabulary_size " + std::to_string(rc.arch.vocabulary_size) + " but the dataset has " +
            std::to_string(meta.vocabulary_size));
    if (meta.row_length > 0 && meta.row_length != rc.arch.window_size)
        die("the dataset is packed in rows of " + std::to_string(meta.row_length) +
            " tokens: set arch_params.window_size to it");
    const int total_epochs = epochs_override > 0 ? epochs_override : rc.training.epochs;
    const std::string latest = run + "/model.lbc", csv_path = run + "/metrics.csv";
    const std::string run_name = fs::path(run).lexically_normal().filename().string();
    const json::Value config_json = rc.to_json();

    int initial_epoch = 0;
    double best_val = -1;
    const bool resume = fs::exists(latest);
    if (resume) {
        const CheckpointInfo info = read_checkpoint_info(latest);
        if (arch_without_window(info.config) != arch_without_window(config_json))
            die(latest + " was trained with other arch_params: start a new run, or restore them in config.json");
        initial_epoch = info.epoch;
        best_val = info.best_val_loss;
    }
    const int csv_done = csv_epochs(csv_path);
    if (csv_done != initial_epoch)
        std::printf("warning: metrics.csv records %d epochs, the checkpoint %d: the run continues from %d\n", csv_done,
                    initial_epoch, initial_epoch);
    if (initial_epoch >= total_epochs) {
        std::printf("Run '%s' already completed %d/%d epochs. Nothing to do.\n", run_name.c_str(), initial_epoch,
                    total_epochs);
        return 0;
    }

    Config c = rc.model_config();
    cudaStream_t s;
    LBC_CUDA(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
    TokenFile train_file(meta.train_file);
    const size_t spp = Loader::steps_per_pass(train_file.size(), c.B, c.T);
    if (spp == 0) die("the train split holds less than one batch");
    const int spe = std::max(1, int(double(spp) * rc.data.epoch_passes));
    const uint64_t seed = uint64_t(rc.data.seed);
    Model model(c, seed);
    if (resume) {
        load_checkpoint(latest, model, s);
        std::printf("Resuming '%s' from epoch %d (optimizer state restored).\n", run_name.c_str(), initial_epoch);
    } else {
        std::printf("Initialized a fresh 'lru-banks-cuda' model for run '%s'.\n", run_name.c_str());
    }
    std::unique_ptr<TokenFile> val_file;
    std::unique_ptr<ValidationSet> val;
    if (rc.training.validation_steps > 0) {
        val_file = std::make_unique<TokenFile>(meta.validation_file);
        val = std::make_unique<ValidationSet>(*val_file, c.B, c.T, rc.training.validation_steps, seed,
                                              meta.doc_token_id, meta.score_after_token, meta.score_prev_token,
                                              meta.row_length > 0);
    }
    Loader loader(train_file, c.B, c.T, seed, meta.doc_token_id, uint64_t(initial_epoch) * uint64_t(spe),
                  meta.score_after_token, meta.score_prev_token, meta.row_length > 0);

    const size_t ntok = size_t(c.B) * c.T;
    int *d_ids, *d_y;
    uint8_t* d_reset;
    float *d_inv_n, *d_val;
    LBC_CUDA(cudaMalloc(&d_ids, ntok * sizeof(int)));
    LBC_CUDA(cudaMalloc(&d_y, ntok * sizeof(int)));
    LBC_CUDA(cudaMalloc(&d_reset, ntok));
    LBC_CUDA(cudaMalloc(&d_inv_n, sizeof(float)));
    LBC_CUDA(cudaMalloc(&d_val, 2 * sizeof(float) * size_t(std::max(1, rc.training.validation_steps))));
    const Batch batch{d_ids, d_y, d_reset, d_inv_n};
    auto copy_batch = [&](const HostBatch& hb) {
        LBC_CUDA(cudaMemcpyAsync(d_ids, hb.ids, ntok * sizeof(int), cudaMemcpyHostToDevice, s));
        LBC_CUDA(cudaMemcpyAsync(d_y, hb.y, ntok * sizeof(int), cudaMemcpyHostToDevice, s));
        LBC_CUDA(cudaMemcpyAsync(d_reset, hb.reset, ntok, cudaMemcpyHostToDevice, s));
        LBC_CUDA(cudaMemcpyAsync(d_inv_n, hb.inv_n, sizeof(float), cudaMemcpyHostToDevice, s));
    };

    cudaDeviceProp prop{};
    int dev = 0;
    LBC_CUDA(cudaGetDevice(&dev));
    LBC_CUDA(cudaGetDeviceProperties(&prop, dev));
    size_t free_b = 0, total_b = 0;
    LBC_CUDA(cudaMemGetInfo(&free_b, &total_b));
    const size_t emb = model.embedding_params();
    size_t real = 0;  // without the alignment padding of the flat buffer
    for (const auto& t : model.tensors()) real += size_t(t.rows) * t.cols;
    std::printf("device=cuda:%d (%s)  arch=lru-banks-cuda  params=%s (embeddings %s / mechanism %s)  memory=%.2f GiB\n",
                dev, prop.name, with_commas(real).c_str(), with_commas(emb).c_str(), with_commas(real - emb).c_str(),
                (total_b - free_b) / 1073741824.0);
    std::printf("dataset=%s  context=%d  batch=%d  steps/epoch=%d (%.3g pass, random windows)  dense supervision  "
                "epochs %d -> %d\n",
                meta.name.c_str(), c.T, c.B, spe, double(spe) / double(spp), initial_epoch, total_epochs);
    std::fflush(stdout);

    // per-step numbers: one asynchronous copy of the model's metrics record into a pinned ring after every step,
    // read once its event has completed (the GPU never waits for the log)
    constexpr int R = 16;
    const int rec = model.metrics_count();
    float* ring = nullptr;
    LBC_CUDA(cudaHostAlloc(&ring, sizeof(float) * size_t(R) * size_t(rec), cudaHostAllocDefault));
    cudaEvent_t ev[R];
    for (auto& e : ev) LBC_CUDA(cudaEventCreateWithFlags(&e, cudaEventDisableTiming));
    std::vector<float> val_host(2 * size_t(std::max(1, rc.training.validation_steps)));

    const bool use_graph = rc.runtime.cuda_graph;
    cudaGraphExec_t step_graph = nullptr, val_graph = nullptr;
    bool step_eager_done = false, val_eager_done = false;
    auto capture = [&](auto&& body) {
        cudaGraph_t g;
        cudaGraphExec_t exec;
        LBC_CUDA(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal));
        body();
        LBC_CUDA(cudaStreamEndCapture(s, &g));
        LBC_CUDA(cudaGraphInstantiate(&exec, g, 0));
        LBC_CUDA(cudaGraphDestroy(g));
        return exec;
    };
    const double ln2 = std::log(2.0);
    const double tcpt = meta.train_chars_per_token, vcpt = meta.validation_chars_per_token;

    for (int epoch = initial_epoch; epoch < total_epochs; ++epoch) {
        std::printf("Epoch %d/%d\n", epoch + 1, total_epochs);
        std::fflush(stdout);
        Progress bar(spe);
        EpochSums sums;
        int consumed = 0;
        const auto t0 = Clock::now();
        auto last_draw = t0;
        auto consume = [&](int i) {
            sums.add(model.decode_metrics(ring + size_t(i % R) * rec));
        };
        auto values = [&](const StepMetrics& m, double per_step) {
            std::string v = " - loss: " + fmt_value(m.loss) +
                            " - sparse_categorical_accuracy: " + fmt_value(sums.acc / std::max(1, sums.n));
            if (tcpt > 0) v += " - bpc: " + fmt_value(m.loss / ln2 / tcpt);
            char b[160];
            std::snprintf(b, sizeof b, " - lr: %.2e - grad_norm: %s - open: %s - tok/s: %.0fk", m.lr,
                          fmt_value(m.grad_norm).c_str(), open_str(m.open_w, m.open_r1, m.open_r2).c_str(),
                          ntok / per_step / 1e3);
            return v + b;
        };
        for (int i = 0; i < spe; ++i) {
            HostBatch* hb = loader.next();
            copy_batch(*hb);
            loader.release(hb, s);
            if (!use_graph || !step_eager_done) {  // the first step eagerly: it builds every GEMM plan
                model.forward_backward(batch, s);
                model.optimizer_step(s);
                step_eager_done = true;
            } else {
                if (!step_graph) step_graph = capture([&] {
                    model.forward_backward(batch, s);
                    model.optimizer_step(s);
                });
                LBC_CUDA(cudaGraphLaunch(step_graph, s));
            }
            if (i >= R && consumed <= i - R) {  // the slot is still unread: wait for it
                LBC_CUDA(cudaEventSynchronize(ev[(i - R) % R]));
                while (consumed <= i - R) consume(consumed++);
            }
            LBC_CUDA(cudaMemcpyAsync(ring + size_t(i % R) * rec, model.metrics_device(), sizeof(float) * rec,
                                     cudaMemcpyDeviceToHost, s));
            LBC_CUDA(cudaEventRecord(ev[i % R], s));
            while (consumed <= i) {
                const cudaError_t e = cudaEventQuery(ev[consumed % R]);
                if (e == cudaErrorNotReady) {
                    (void)cudaGetLastError();
                    break;
                }
                LBC_CUDA(e);
                consume(consumed++);
            }
            if (consumed > 0 && seconds_since(last_draw) >= 0.5 && consumed < spe) {
                last_draw = Clock::now();
                const double per_step = seconds_since(t0) / consumed;
                bar.draw(consumed, fmt_eta(per_step * (spe - consumed)), per_step, values(sums.last, per_step),
                         false);
            }
        }
        LBC_CUDA(cudaStreamSynchronize(s));
        while (consumed < spe) consume(consumed++);
        const double train_s = seconds_since(t0);
        const double per_step = train_s / spe;

        // validation: the same rows every epoch, forward and loss only
        double val_loss = -1, val_acc = -1;
        if (val) {
            for (int k = 0; k < val->steps(); ++k) {
                copy_batch(val->batch(k));
                if (!use_graph || !val_eager_done) {
                    model.forward_loss(batch, s);
                    val_eager_done = true;
                } else {
                    if (!val_graph) val_graph = capture([&] { model.forward_loss(batch, s); });
                    LBC_CUDA(cudaGraphLaunch(val_graph, s));
                }
                LBC_CUDA(cudaMemcpyAsync(d_val + 2 * k, model.metrics_device(), 2 * sizeof(float),
                                         cudaMemcpyDeviceToDevice, s));
            }
            LBC_CUDA(cudaMemcpyAsync(val_host.data(), d_val, 2 * sizeof(float) * size_t(val->steps()),
                                     cudaMemcpyDeviceToHost, s));
            LBC_CUDA(cudaStreamSynchronize(s));
            val_loss = val_acc = 0;
            for (int k = 0; k < val->steps(); ++k) {
                val_loss += val_host[2 * size_t(k)] / val->steps();
                val_acc += val_host[2 * size_t(k) + 1] / val->steps();
            }
        }
        const double epoch_s = seconds_since(t0);

        std::string v = values(sums.last, per_step);
        if (val) {
            v += " - val_loss: " + fmt_value(val_loss) + " - val_sparse_categorical_accuracy: " + fmt_value(val_acc);
            if (vcpt > 0) v += " - val_bpc: " + fmt_value(val_loss / ln2 / vcpt);
        }
        bar.draw(spe, std::to_string(long(epoch_s + 0.5)) + "s", per_step, v, true);

        // checkpoints: the latest (the resume point), numbered every checkpoint_every epochs, the best val_loss
        const bool improved = val && std::isfinite(val_loss) && (best_val < 0 || val_loss < best_val);
        if (improved) best_val = val_loss;
        save_checkpoint(latest, model, config_json, epoch + 1, best_val, s);
        const int every = rc.training.checkpoint_every;
        if (every > 0 && (epoch + 1) % every == 0) {
            char name[64];
            std::snprintf(name, sizeof name, "/checkpoints/checkpoint_%05d.lbc", epoch + 1);
            copy_atomic(latest, run + name);
        }
        if (improved) copy_atomic(latest, run + "/checkpoints/best.lbc");

        // metrics.csv: the Python pipeline's columns, then this trainer's
        const bool header = !fs::exists(csv_path) || fs::file_size(csv_path) == 0;
        std::ofstream csv(csv_path, std::ios::app);
        if (header)
            csv << "epoch,loss,sparse_categorical_accuracy,val_loss,val_sparse_categorical_accuracy,bpc,val_bpc,lr,"
                   "grad_norm,open_w,open_r1,open_r2,ms_per_step,tokens_per_s,seconds\n";
        const int n = std::max(1, sums.n);
        const double mean_loss = sums.loss / n;
        auto num = [](double x) {
            char b[32];
            std::snprintf(b, sizeof b, "%.8g", x);
            return std::string(b);
        };
        auto opt = [&](bool have, double x) { return have ? num(x) : std::string(); };
        csv << epoch << "," << num(mean_loss) << "," << num(sums.acc / n) << "," << opt(bool(val), val_loss) << ","
            << opt(bool(val), val_acc) << "," << opt(tcpt > 0, mean_loss / ln2 / tcpt) << ","
            << opt(val && vcpt > 0, val_loss / ln2 / vcpt) << "," << num(sums.last.lr) << ","
            << num(sums.grad_norm / n) << "," << num(sums.open_w / n) << "," << num(sums.open_r1 / n) << ","
            << num(sums.open_r2 / n) << "," << num(per_step * 1e3) << "," << num(ntok / per_step) << ","
            << num(epoch_s) << "\n";
    }
    return 0;
}

// One forward pass of a checkpoint on `rows` fixed rows of a token file, dumped for offline analysis
// (scripts/analyze_copy.py): config.txt, manifest.txt, params.bin, ids / y / reset, each layer's decisions (routed slots,
// gates) and the loss.
int cmd_dump(int argc, char** argv) {
    if (argc < 1) usage();
    const std::string run = argv[0];
    std::string which = "latest", split, out = "/tmp/lbc_inspect", silence;
    int rows = 2;
    double tau_scale = 1.0, write_bias_add = 0.0;  // inference-time interventions (analysis only)
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--checkpoint" && i + 1 < argc) which = argv[++i];
        else if (a == "--split" && i + 1 < argc) split = argv[++i];
        else if (a == "--rows" && i + 1 < argc) rows = std::atoi(argv[++i]);
        else if (a == "--out" && i + 1 < argc) out = argv[++i];
        else if (a == "--silence" && i + 1 < argc) silence = argv[++i];
        else if (a == "--tau-scale" && i + 1 < argc) tau_scale = std::atof(argv[++i]);
        else if (a == "--write-bias-add" && i + 1 < argc) write_bias_add = std::atof(argv[++i]);
        else usage();
    }
    const std::string path = which == "best" ? run + "/checkpoints/best.lbc" : which == "latest" ? run + "/model.lbc" : which;
    const CheckpointInfo info = read_checkpoint_info(path);
    const RunConfig rc = RunConfig::from_json(info.config);
    const DatasetMeta meta = DatasetMeta::load(rc.data.dataset);
    if (split.empty()) split = meta.validation_file;
    Config c = rc.model_config();
    c.B = rows;
    cudaStream_t s;
    LBC_CUDA(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
    Model model(c, uint64_t(rc.data.seed));
    load_checkpoint(path, model, s);
    if (!silence.empty()) model.set_silence(parse_silence(silence, c.L));
    if (tau_scale != 1.0 || write_bias_add != 0.0) {  // edit the loaded parameters: read temperatures, write-gate biases
        std::vector<float> P(model.num_params());
        LBC_CUDA(cudaMemcpy(P.data(), model.params(), P.size() * 4, cudaMemcpyDeviceToHost));
        for (const auto& t : model.tensors()) {
            const bool tau = t.name.find("log_tau") != std::string::npos, bias = t.name.size() > 4 &&
                             t.name.compare(t.name.size() - 4, 4, "_bwg") == 0;
            for (size_t i = 0; i < size_t(t.rows) * t.cols; ++i) {
                if (tau) P[t.off + i] += float(std::log(tau_scale));
                if (bias) P[t.off + i] += float(write_bias_add);
            }
        }
        LBC_CUDA(cudaMemcpy(model.params(), P.data(), P.size() * 4, cudaMemcpyHostToDevice));
        model.prepare_weights(s);
        std::printf("intervention: read temperatures x%.2f, write-gate biases %+.1f\n", tau_scale, write_bias_add);
    }
    model.keep_row_loss();
    TokenFile file(split);
    ValidationSet vs(file, c.B, c.T, 1, uint64_t(rc.data.seed), meta.doc_token_id, meta.score_after_token,
                     meta.score_prev_token, meta.row_length > 0);
    const HostBatch& hb = vs.batch(0);
    const size_t ntok = size_t(c.B) * c.T;
    int *d_ids, *d_y;
    uint8_t* d_reset;
    float* d_inv_n;
    LBC_CUDA(cudaMalloc(&d_ids, ntok * sizeof(int)));
    LBC_CUDA(cudaMalloc(&d_y, ntok * sizeof(int)));
    LBC_CUDA(cudaMalloc(&d_reset, ntok));
    LBC_CUDA(cudaMalloc(&d_inv_n, sizeof(float)));
    LBC_CUDA(cudaMemcpy(d_ids, hb.ids, ntok * sizeof(int), cudaMemcpyHostToDevice));
    LBC_CUDA(cudaMemcpy(d_y, hb.y, ntok * sizeof(int), cudaMemcpyHostToDevice));
    LBC_CUDA(cudaMemcpy(d_reset, hb.reset, ntok, cudaMemcpyHostToDevice));
    LBC_CUDA(cudaMemcpy(d_inv_n, hb.inv_n, sizeof(float), cudaMemcpyHostToDevice));
    model.forward_loss(Batch{d_ids, d_y, d_reset, d_inv_n}, s);
    float r[2];
    LBC_CUDA(cudaMemcpyAsync(r, model.metrics_device(), sizeof(r), cudaMemcpyDeviceToHost, s));
    LBC_CUDA(cudaStreamSynchronize(s));
    fs::create_directories(out);
    auto bin = [&](const std::string& name, const void* p, size_t bytes) {
        std::ofstream f(out + "/" + name, std::ios::binary);
        f.write(static_cast<const char*>(p), std::streamsize(bytes));
    };
    {
        std::ofstream f(out + "/config.txt");
        f << "V " << c.V << "\nE0 " << c.E0 << "\nD " << c.D << "\nL " << c.L << "\nN " << c.N << "\nH " << c.H
          << "\nF " << c.F << "\nB " << c.B << "\nT " << c.T << "\nDH " << c.dh << "\nSLOTS " << SLOTS
          << "\ntau " << c.tau << "\nslot_hl_min " << c.slot_hl_min << "\nslot_hl_max " << c.slot_hl_max
          << "\nfactorized " << int(c.factorized) << "\ntied " << int(c.tied) << "\nhops ";
        for (int l = 0; l < c.L; ++l) f << (l ? "," : "") << c.hops_of(l);
        f << "\nbank_slots ";
        for (int l = 0; l < c.L; ++l) f << (l ? "," : "") << c.slots_of(l);
        f << "\nloss " << r[0] << "\naccuracy " << r[1] << "\nn_targets " << hb.n_targets << "\nepoch " << info.epoch
          << "\n";
    }
    {
        std::ofstream f(out + "/manifest.txt");
        for (const auto& t : model.tensors()) f << t.name << " " << t.off << " " << t.rows << " " << t.cols << "\n";
    }
    std::vector<float> P(model.num_params());
    LBC_CUDA(cudaMemcpy(P.data(), model.params(), P.size() * 4, cudaMemcpyDeviceToHost));
    bin("params.bin", P.data(), P.size() * 4);
    bin("ids.bin", hb.ids, ntok * sizeof(int));
    bin("y.bin", hb.y, ntok * sizeof(int));
    bin("reset.bin", hb.reset, ntok);
    const std::vector<float> rl = model.row_loss(s);
    bin("row_loss.bin", rl.data(), rl.size() * 4);
    for (int l = 0; l < c.L; ++l) {
        std::vector<uint8_t> ridx, gw, gr, gr2;
        model.decisions(l, ridx, gw, gr, gr2, s);
        const std::string pfx = "l" + std::to_string(l) + "_";
        bin(pfx + "ridx.bin", ridx.data(), ridx.size());
        bin(pfx + "gw.bin", gw.data(), gw.size());
        bin(pfx + "gr.bin", gr.data(), gr.size());
        bin(pfx + "gr2.bin", gr2.data(), gr2.size());
        const char* names[5] = {"q1", "q2", "o1", "o2", "k"};
        for (int w = 0; w < 5; ++w) {
            const std::vector<bf16> x = model.download_read(l, w, s);
            if (!x.empty()) bin(pfx + names[w] + ".bf16", x.data(), x.size() * sizeof(bf16));
        }
    }
    std::printf("dumped %d rows of %s (epoch %d) to %s: loss %.4f accuracy %.4f over %d targets\n", rows, path.c_str(),
                info.epoch, out.c_str(), r[0], r[1], hb.n_targets);
    return 0;
}

// loss and accuracy of a checkpoint on `steps` fixed batches of a token file (the validation split by default; same
// rows and scoring as the trainer's validation)
int cmd_eval(int argc, char** argv) {
    if (argc < 1) usage();
    const std::string run = argv[0];
    std::string which = "best", split, silence;
    int steps = 0, rows = 0;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--checkpoint" && i + 1 < argc) which = argv[++i];
        else if (a == "--split" && i + 1 < argc) split = argv[++i];
        else if (a == "--steps" && i + 1 < argc) steps = std::atoi(argv[++i]);
        else if (a == "--rows" && i + 1 < argc) rows = std::atoi(argv[++i]);
        else if (a == "--silence" && i + 1 < argc) silence = argv[++i];
        else usage();
    }
    std::string path = which == "best" ? run + "/checkpoints/best.lbc" : which == "latest" ? run + "/model.lbc" : which;
    if (which == "best" && !fs::exists(path)) path = run + "/model.lbc";
    const CheckpointInfo info = read_checkpoint_info(path);
    const RunConfig rc = RunConfig::from_json(info.config);
    const DatasetMeta meta = DatasetMeta::load(rc.data.dataset);
    if (split.empty()) split = meta.validation_file;
    if (steps <= 0) steps = std::max(1, rc.training.validation_steps);
    Config c = rc.model_config();
    if (rows > 0) c.B = rows;  // fewer rows per batch (e.g. beside a training run)
    cudaStream_t s;
    LBC_CUDA(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
    Model model(c, uint64_t(rc.data.seed));
    load_checkpoint(path, model, s);
    if (!silence.empty()) model.set_silence(parse_silence(silence, c.L));
    TokenFile file(split);
    ValidationSet vs(file, c.B, c.T, steps, uint64_t(rc.data.seed), meta.doc_token_id, meta.score_after_token,
                     meta.score_prev_token, meta.row_length > 0);
    const size_t ntok = size_t(c.B) * c.T;
    int *d_ids, *d_y;
    uint8_t* d_reset;
    float* d_inv_n;
    LBC_CUDA(cudaMalloc(&d_ids, ntok * sizeof(int)));
    LBC_CUDA(cudaMalloc(&d_y, ntok * sizeof(int)));
    LBC_CUDA(cudaMalloc(&d_reset, ntok));
    LBC_CUDA(cudaMalloc(&d_inv_n, sizeof(float)));
    const Batch batch{d_ids, d_y, d_reset, d_inv_n};
    double loss = 0, acc = 0;
    long long targets = 0;
    for (int k = 0; k < vs.steps(); ++k) {
        const HostBatch& hb = vs.batch(k);
        LBC_CUDA(cudaMemcpyAsync(d_ids, hb.ids, ntok * sizeof(int), cudaMemcpyHostToDevice, s));
        LBC_CUDA(cudaMemcpyAsync(d_y, hb.y, ntok * sizeof(int), cudaMemcpyHostToDevice, s));
        LBC_CUDA(cudaMemcpyAsync(d_reset, hb.reset, ntok, cudaMemcpyHostToDevice, s));
        LBC_CUDA(cudaMemcpyAsync(d_inv_n, hb.inv_n, sizeof(float), cudaMemcpyHostToDevice, s));
        model.forward_loss(batch, s);
        float r[2];
        LBC_CUDA(cudaMemcpyAsync(r, model.metrics_device(), sizeof(r), cudaMemcpyDeviceToHost, s));
        LBC_CUDA(cudaStreamSynchronize(s));
        loss += double(r[0]) * hb.n_targets;  // target-weighted (rows differ in scored targets)
        acc += double(r[1]) * hb.n_targets;
        targets += hb.n_targets;
    }
    std::printf("checkpoint=%s (epoch %d)  split=%s  steps=%d  targets=%lld  silence=%s  loss=%.4f  accuracy=%.4f\n",
                path.c_str(), info.epoch, split.c_str(), vs.steps(), targets, silence.empty() ? "-" : silence.c_str(),
                loss / std::max(1LL, targets), acc / std::max(1LL, targets));
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) usage();
    const std::string cmd = argv[1];
    try {
        if (cmd == "create") return cmd_create(argc - 2, argv + 2);
        if (cmd == "config") return cmd_config(argc - 2, argv + 2);
        if (cmd == "train") return cmd_train(argc - 2, argv + 2);
        if (cmd == "eval") return cmd_eval(argc - 2, argv + 2);
        if (cmd == "dump") return cmd_dump(argc - 2, argv + 2);
    } catch (const std::exception& e) {
        die(e.what());
    }
    usage();
}
