// Trains the model on real data and reports the throughput of the whole pipeline: disk reads (memory-mapped token
// file, a loader thread), host-to-device copies, forward, loss, backward and the optimizer, every step.
//
//   train_bench --data /root/data/pretrain-mix-bpe32768/train.npy [--steps 30] [--warmup 5] [--batch 64]
//               [--seq 512] [--layers 4] [--gate-w 2] [--gate-r 2] [--lr 2e-3] [--budget 0] [--tiny]
//               [--ce-chunk 256] [--bank-chunk 512]  (tokens per bank-walk block; longer rows walk in chunks)
//                 (loss rows per chunk: 256 keeps a chunk's logits and gradients in the 4090's L2)
//               [--dump DIR]   (one forward / backward, then parameters, batch, decisions, loss and gradients
//                               written to DIR for tests/check_grads.py)  [--dump-after N]  (N updates first)
//               [--grad-report]  (the profiled step's gradient norm per tensor, largest first)
//               [--set section.key=value ...]   (any config.json key, e.g. arch_params.tied_embeddings=false,
//                               arch_params.hops=[2,1]; applied after the flags above)

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "../src/data.hpp"
#include "../src/model.hpp"
#include "../src/run_config.hpp"
#include "lbc/check.hpp"

using namespace lbc;

namespace {

struct Args {
    std::string data, dump;
    int dump_after = 0;
    int steps = 30, warmup = 5, batch = 64, seq = 512, layers = 4, ce_chunk = 256, bank_chunk = 512;
    float gate_w = 2.f, gate_r = 2.f, lr = 2e-3f, budget = 0.f;
    bool tiny = false, gate_report = false, graph = true, grad_report = false;
    float budget_target = 0.5f;
    uint64_t seed = 1;
    std::vector<std::pair<std::string, std::string>> sets;
};

Args parse(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        const std::string k = argv[i];
        auto val = [&]() -> const char* {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "%s needs a value\n", k.c_str());
                std::exit(2);
            }
            return argv[++i];
        };
        if (k == "--data") a.data = val();
        else if (k == "--dump") a.dump = val();
        else if (k == "--dump-after") a.dump_after = std::atoi(val());
        else if (k == "--steps") a.steps = std::atoi(val());
        else if (k == "--warmup") a.warmup = std::atoi(val());
        else if (k == "--batch") a.batch = std::atoi(val());
        else if (k == "--seq") a.seq = std::atoi(val());
        else if (k == "--layers") a.layers = std::atoi(val());
        else if (k == "--gate-w") a.gate_w = std::atof(val());
        else if (k == "--gate-r") a.gate_r = std::atof(val());
        else if (k == "--lr") a.lr = std::atof(val());
        else if (k == "--budget") a.budget = std::atof(val());
        else if (k == "--seed") a.seed = std::strtoull(val(), nullptr, 10);
        else if (k == "--ce-chunk") a.ce_chunk = std::atoi(val());
        else if (k == "--bank-chunk") a.bank_chunk = std::atoi(val());
        else if (k == "--tiny") a.tiny = true;
        else if (k == "--gate-report") a.gate_report = true;
        else if (k == "--grad-report") a.grad_report = true;
        else if (k == "--no-graph") a.graph = false;
        else if (k == "--budget-target") a.budget_target = std::atof(val());
        else if (k == "--set") {
            const std::string kv = val();
            const size_t eq = kv.find('=');
            if (eq == std::string::npos) {
                std::fprintf(stderr, "--set needs section.key=value\n");
                std::exit(2);
            }
            a.sets.emplace_back(kv.substr(0, eq), kv.substr(eq + 1));
        }
        else {
            std::fprintf(stderr, "unknown argument %s\n", k.c_str());
            std::exit(2);
        }
    }
    if (a.data.empty()) {
        std::fprintf(stderr, "--data PATH (a uint16 .npy token file) is required\n");
        std::exit(2);
    }
    return a;
}

template <class T>
void write_bin(const std::string& path, const T* p, size_t n) {
    std::ofstream f(path, std::ios::binary);
    f.write(reinterpret_cast<const char*>(p), std::streamsize(n * sizeof(T)));
}

template <class T>
std::vector<T> download(const T* d, size_t n) {
    std::vector<T> h(n);
    LBC_CUDA(cudaMemcpy(h.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost));
    return h;
}

void dump(const std::string& dir, Model& model, const HostBatch& hb, cudaStream_t s) {
    const Config& c = model.config();
    std::filesystem::create_directories(dir);
    float loss = 0.f;
    LBC_CUDA(cudaMemcpy(&loss, model.loss_device(), 4, cudaMemcpyDeviceToHost));
    {
        std::ofstream f(dir + "/config.txt");
        f << "V " << c.V << "\nE0 " << c.E0 << "\nD " << c.D << "\nL " << c.L << "\nN " << c.N << "\nH " << c.H
          << "\nF " << c.F << "\nB " << c.B << "\nT " << c.T << "\nDH " << c.dh << "\nSLOTS " << SLOTS << "\nldPV "
          << c.ldPV() << "\nldG " << c.ldG() << "\nldG2 " << c.ldG2() << "\ntau " << c.tau << "\nslot_hl_min "
          << c.slot_hl_min << "\nslot_hl_max " << c.slot_hl_max << "\nbudget " << c.budget << "\nbudget_target "
          << c.budget_target << "\nfactorized " << int(c.factorized) << "\ntied " << int(c.tied) << "\nhops ";
        for (int l = 0; l < c.L; ++l) f << (l ? "," : "") << c.hops_of(l);
        f << "\nbank_slots ";
        for (int l = 0; l < c.L; ++l) f << (l ? "," : "") << c.slots_of(l);
        f << "\nloss " << loss << "\nn_targets " << hb.n_targets << "\n";
    }
    {
        std::ofstream f(dir + "/manifest.txt");
        for (const auto& t : model.tensors()) f << t.name << " " << t.off << " " << t.rows << " " << t.cols << "\n";
    }
    write_bin(dir + "/params.bin", download(model.params(), model.num_params()).data(), model.num_params());
    write_bin(dir + "/grads.bin", download(model.grads(), model.num_params()).data(), model.num_params());
    const size_t n = size_t(c.B) * c.T;
    write_bin(dir + "/ids.bin", hb.ids, n);
    write_bin(dir + "/y.bin", hb.y, n);
    write_bin(dir + "/reset.bin", hb.reset, n);
    for (int l = 0; l < c.L; ++l) {
        std::vector<uint8_t> ridx, gw, gr, gr2;
        model.decisions(l, ridx, gw, gr, gr2, s);
        const std::string p = dir + "/l" + std::to_string(l) + "_";
        write_bin(p + "ridx.bin", ridx.data(), ridx.size());
        write_bin(p + "gw.bin", gw.data(), gw.size());
        write_bin(p + "gr.bin", gr.data(), gr.size());
        write_bin(p + "gr2.bin", gr2.data(), gr2.size());
    }
    std::printf("dumped to %s (loss %.6f)\n", dir.c_str(), loss);
}

}  // namespace

int main(int argc, char** argv) {
    const Args a = parse(argc, argv);
    // the flags, then --set, through the run config (so every value is checked); constant learning rate
    RunConfig rc;
    rc.data.batch_size = a.batch;
    rc.arch.window_size = a.seq;
    rc.arch.num_layers = a.layers;
    rc.arch.hops.assign(size_t(a.layers), 2);
    rc.arch.gate_bias_write = a.gate_w;
    rc.arch.gate_bias_read = a.gate_r;
    rc.arch.gate_budget = a.budget;
    rc.arch.gate_budget_target = a.budget_target;
    rc.training.learning_rate = a.lr;
    rc.training.warmup_steps = 0;
    rc.runtime.ce_chunk = a.ce_chunk;
    rc.runtime.bank_chunk = a.bank_chunk;
    // the benchmark protocol's model: 16 heads x 32 (the config default is 8 x 64)
    rc.arch.num_heads = 16;
    rc.arch.head_dim = 32;
    if (a.tiny) {  // small widths for the gradient check (SLOTS is fixed by the kernels)
        rc.arch.embed_dim = 32;
        rc.arch.dim = 64;
        rc.arch.lru_dim = 32;
        rc.arch.num_heads = 2;
        rc.arch.ffn_mult = 2;
        rc.arch.num_layers = 2;
        rc.arch.hops = {2, 2};
        rc.data.batch_size = 2;
        rc.arch.window_size = 64;
        rc.runtime.ce_chunk = 64;
    }
    try {
        rc.validate();
        for (const auto& [k, v] : a.sets) rc = with_key(rc, k, v);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "%s\n", e.what());
        return 2;
    }
    const Config c = rc.model_config();

    TokenFile file(a.data);
    Loader loader(file, c.B, c.T, a.seed);
    Model model(c, a.seed);
    cudaStream_t s;
    LBC_CUDA(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
    const size_t ntok = size_t(c.B) * c.T;
    int *d_ids, *d_y;
    uint8_t* d_reset;
    float* d_inv_n;
    LBC_CUDA(cudaMalloc(&d_inv_n, sizeof(float)));
    LBC_CUDA(cudaMalloc(&d_ids, ntok * sizeof(int)));
    LBC_CUDA(cudaMalloc(&d_y, ntok * sizeof(int)));
    LBC_CUDA(cudaMalloc(&d_reset, ntok));

    size_t free_b = 0, total_b = 0;
    LBC_CUDA(cudaMemGetInfo(&free_b, &total_b));
    std::printf("model: %zu params, %d layers (hops", model.num_params(), c.L);
    for (int l = 0; l < c.L; ++l) std::printf("%s%d", l ? "," : " ", c.hops_of(l));
    std::printf("), D %d, LRU %d complex, %d heads x %d, %d slots, FFN %d, vocab %d x %d (%s, %s)\n", c.D, c.N, c.H,
                c.dh, SLOTS, c.F, c.V, c.E0, c.factorized ? "factorized" : "not factorized",
                c.tied ? "tied" : "untied");
    std::printf("batch %d x %d = %zu tokens per step; gate biases write %.2f read %.2f; %zu tokens on disk\n", c.B,
                c.T, ntok, c.gate_bias_w, c.gate_bias_r, file.size());
    std::printf("GPU memory in use after allocation: %.2f GiB of %.2f\n", (total_b - free_b) / 1073741824.0,
                total_b / 1073741824.0);

    auto copy_batch = [&](HostBatch* hb) {
        LBC_CUDA(cudaMemcpyAsync(d_ids, hb->ids, ntok * sizeof(int), cudaMemcpyHostToDevice, s));
        LBC_CUDA(cudaMemcpyAsync(d_y, hb->y, ntok * sizeof(int), cudaMemcpyHostToDevice, s));
        LBC_CUDA(cudaMemcpyAsync(d_reset, hb->reset, ntok, cudaMemcpyHostToDevice, s));
        LBC_CUDA(cudaMemcpyAsync(d_inv_n, hb->inv_n, sizeof(float), cudaMemcpyHostToDevice, s));
        const Batch b{d_ids, d_y, d_reset, d_inv_n};
        return b;
    };

    if (!a.dump.empty()) {
        for (int i = 0; i < a.dump_after; ++i) {  // training steps first (eager)
            HostBatch* hb = loader.next();
            const Batch b = copy_batch(hb);
            loader.release(hb, s);
            model.forward_backward(b, s);
            model.optimizer_step(s);
        }
        HostBatch* hb = loader.next();
        HostBatch keep = *hb;  // the pinned buffers stay valid until the slot is refilled (after this run)
        const Batch b = copy_batch(hb);
        model.forward_backward(b, s);
        LBC_CUDA(cudaStreamSynchronize(s));
        dump(a.dump, model, keep, s);
        loader.release(hb, s);
        return 0;
    }

    const int total = a.warmup + a.steps;
    float* d_hist;
    LBC_CUDA(cudaMalloc(&d_hist, total * sizeof(float)));
    cudaEvent_t t0, t1;
    LBC_CUDA(cudaEventCreate(&t0));
    LBC_CUDA(cudaEventCreate(&t1));
    double wait_before = 0.0;
    // the step (forward, loss, backward, optimizer) as one CUDA graph, captured after an eager first step (which
    // builds every GEMM plan); the batch copies stay outside it
    cudaGraphExec_t step_graph = nullptr;
    for (int i = 0; i < total; ++i) {
        if (i == a.warmup) {
            LBC_CUDA(cudaEventRecord(t0, s));
            wait_before = loader.wait_ms();
        }
        HostBatch* hb = loader.next();
        const Batch b = copy_batch(hb);
        loader.release(hb, s);
        if (!a.graph || i == 0) {
            model.forward_backward(b, s);
            model.optimizer_step(s);
        } else {
            if (!step_graph) {
                cudaGraph_t g;
                LBC_CUDA(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal));
                model.forward_backward(b, s);
                model.optimizer_step(s);
                LBC_CUDA(cudaStreamEndCapture(s, &g));
                LBC_CUDA(cudaGraphInstantiate(&step_graph, g, 0));
                LBC_CUDA(cudaGraphDestroy(g));
            }
            LBC_CUDA(cudaGraphLaunch(step_graph, s));
        }
        LBC_CUDA(cudaMemcpyAsync(d_hist + i, model.loss_device(), sizeof(float), cudaMemcpyDeviceToDevice, s));
    }
    LBC_CUDA(cudaEventRecord(t1, s));
    LBC_CUDA(cudaEventSynchronize(t1));
    float ms = 0.f;
    LBC_CUDA(cudaEventElapsedTime(&ms, t0, t1));
    const double step_ms = ms / a.steps;
    std::printf("\n%d timed steps (after %d warm-up)%s: %.1f ms per step, %.0f tokens/s, %.2f updates/s\n", a.steps,
                a.warmup, a.graph ? ", CUDA graph" : "", step_ms, ntok / (step_ms * 1e-3), 1e3 / step_ms);
    std::printf("loader: waited %.1f ms in total over the timed steps; filling took %.2f ms per batch (thread)\n",
                loader.wait_ms() - wait_before, loader.fill_ms() / total);

    std::vector<float> hist(total);
    LBC_CUDA(cudaMemcpy(hist.data(), d_hist, total * sizeof(float), cudaMemcpyDeviceToHost));
    std::printf("loss by step:");
    for (int i = 0; i < total; i += std::max(1, total / 12)) std::printf(" [%d] %.3f", i + 1, hist[i]);
    std::printf(" [%d] %.3f\n", total, hist[total - 1]);

    const std::vector<float> g = model.gate_stats(s);
    std::printf("open gates (last step), per layer write / read 1 / read 2:");
    for (int n = 0; n < c.L; ++n) std::printf("  L%d %.2f %.2f %.2f", n, g[3 * n], g[3 * n + 1], g[3 * n + 2]);
    std::printf("\n");

    if (a.gate_report) {  // the last step's read gates: how often all heads of a token are closed (rows a GEMM could skip)
        std::printf("gate report (last step): per layer, hop: open reads, tokens with every head closed\n");
        for (int n = 0; n < c.L; ++n) {
            std::vector<uint8_t> ridx, gw, gr, gr2;
            model.decisions(n, ridx, gw, gr, gr2, s);
            for (int hop = 0; hop < 2; ++hop) {
                const std::vector<uint8_t>& g = hop == 0 ? gr : gr2;
                size_t open = 0, all_closed = 0;
                for (size_t r = 0; r < ntok; ++r) {
                    int o = 0;
                    for (int h = 0; h < c.H; ++h) o += g[r * c.H + h];
                    open += o;
                    all_closed += o == 0;
                }
                std::printf("  L%d hop %d: open %.3f, all heads closed %.4f\n", n, hop + 1,
                            double(open) / (double(ntok) * c.H), double(all_closed) / double(ntok));
            }
        }
    }

    // one more step with an event pair around every launch: GPU time per category
    model.set_profile(true);
    {
        HostBatch* hb = loader.next();
        const Batch b = copy_batch(hb);
        loader.release(hb, s);
        model.forward_backward(b, s);
        model.optimizer_step(s);
    }
    model.set_profile(false);
    if (a.grad_report) {
        LBC_CUDA(cudaStreamSynchronize(s));  // the stream is non-blocking: cudaMemcpy would not wait for it
        const std::vector<float> gh = download(model.grads(), model.num_params());
        std::vector<std::pair<double, std::string>> norms;
        double total = 0;
        for (const auto& t : model.tensors()) {
            double ss = 0;
            for (size_t i = 0; i < size_t(t.rows) * t.cols; ++i) ss += double(gh[t.off + i]) * gh[t.off + i];
            total += ss;
            norms.emplace_back(std::sqrt(ss), t.name);
        }
        std::sort(norms.rbegin(), norms.rend());
        std::printf("gradient norm (profiled step) %.4g; largest tensors:", std::sqrt(total));
        for (size_t i = 0; i < std::min<size_t>(14, norms.size()); ++i)
            std::printf(" %s %.3g", norms[i].second.c_str(), norms[i].first);
        std::printf("\n");
        // W_s's columns: [key (H*DH) | query (H*DH) | read gate (H) | write gate (H)]
        const int HD = c.H * c.dh;
        const int groups[5] = {0, HD, 2 * HD, 2 * HD + c.H, 2 * HD + 2 * c.H};
        const char* gname[4] = {"key", "query", "read gate", "write gate"};
        for (const auto& t : model.tensors()) {
            if (t.name.size() < 3 || t.name.compare(t.name.size() - 3, 3, "_Ws") != 0) continue;
            std::printf("  %s by columns:", t.name.c_str());
            for (int k = 0; k < 4; ++k) {
                double ss = 0;
                for (int r = 0; r < t.rows; ++r)
                    for (int j = groups[k]; j < groups[k + 1]; ++j) ss += double(gh[t.off + size_t(r) * t.cols + j]) * gh[t.off + size_t(r) * t.cols + j];
                std::printf(" %s %.3g", gname[k], std::sqrt(ss));
            }
            std::printf("\n");
        }
    }
    float sum = 0.f;
    const auto prof = model.profile_ms();
    for (const auto& [cat, t] : prof) sum += t;
    std::printf("GPU time by category (one profiled step, %.1f ms in kernels):\n", sum);
    for (const auto& [cat, t] : prof) std::printf("  %-18s %8.2f ms  %5.1f%%\n", cat.c_str(), t, 100.f * t / sum);
    LBC_CUDA(cudaMemGetInfo(&free_b, &total_b));
    std::printf("GPU memory in use: %.2f GiB\n", (total_b - free_b) / 1073741824.0);
    return 0;
}
