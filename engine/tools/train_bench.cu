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
//               [--carry-check]  (truncated BPTT, with --set data.stream_chunks=2 or more: two consecutive rows of T
//                               with the state carried from the first against one row of 2T, the loss of the second
//                               half; equal when the carried state is the true continuation)
// With data.stream_chunks > 1 the loader streams rows (truncated BPTT) and --dump also writes each layer's carried
// state (the dumped step's start, so use --dump-after 1 or more): tests/check_grads.py starts from it.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <ranges>
#include <string>
#include <vector>

#include "../src/data.hpp"
#include "../src/model.hpp"
#include "../src/run_config.hpp"
#include "lbc/check.hpp"
#include "lbc/text.hpp"

using namespace lbc;

namespace {

struct Args {
    std::string data, dump;
    int dump_after = 0;
    int steps = 30, warmup = 5, batch = 64, seq = 512, layers = 4, ce_chunk = 256, bank_chunk = 512;
    float gate_w = 2.f, gate_r = 2.f, lr = 2e-3f, budget = 0.f;
    bool tiny = false, gate_report = false, graph = true, grad_report = false, carry_check = false;
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
                std::exit(2);  // NOLINT(concurrency-mt-unsafe): argument parsing, no threads yet
            }
            return argv[++i];
        };
        if (k == "--data") {
            a.data = val();
        } else if (k == "--dump") {
            a.dump = val();
        } else if (k == "--dump-after") {
            a.dump_after = parse_int(val(), "--dump-after");
        } else if (k == "--steps") {
            a.steps = parse_int(val(), "--steps");
        } else if (k == "--warmup") {
            a.warmup = parse_int(val(), "--warmup");
        } else if (k == "--batch") {
            a.batch = parse_int(val(), "--batch");
        } else if (k == "--seq") {
            a.seq = parse_int(val(), "--seq");
        } else if (k == "--layers") {
            a.layers = parse_int(val(), "--layers");
        } else if (k == "--gate-w") {
            a.gate_w = parse_double(val(), "--gate-w");
        } else if (k == "--gate-r") {
            a.gate_r = parse_double(val(), "--gate-r");
        } else if (k == "--lr") {
            a.lr = parse_double(val(), "--lr");
        } else if (k == "--budget") {
            a.budget = parse_double(val(), "--budget");
        } else if (k == "--seed") {
            a.seed = parse_number<uint64_t>(val(), "--seed");
        } else if (k == "--ce-chunk") {
            a.ce_chunk = parse_int(val(), "--ce-chunk");
        } else if (k == "--bank-chunk") {
            a.bank_chunk = parse_int(val(), "--bank-chunk");
        } else if (k == "--tiny") {
            a.tiny = true;
        } else if (k == "--gate-report") {
            a.gate_report = true;
        } else if (k == "--grad-report") {
            a.grad_report = true;
        } else if (k == "--carry-check") {
            a.carry_check = true;
        } else if (k == "--no-graph") {
            a.graph = false;
        } else if (k == "--budget-target") {
            a.budget_target = parse_double(val(), "--budget-target");
        } else if (k == "--set") {
            const std::string kv = val();
            const size_t eq = kv.find('=');
            if (eq == std::string::npos) {
                std::fprintf(stderr, "--set needs section.key=value\n");
                std::exit(2);  // NOLINT(concurrency-mt-unsafe): argument parsing, no threads yet
            }
            a.sets.emplace_back(kv.substr(0, eq), kv.substr(eq + 1));
        } else {
            std::fprintf(stderr, "unknown argument %s\n", k.c_str());
            std::exit(2);  // NOLINT(concurrency-mt-unsafe): argument parsing, no threads yet
        }
    }
    if (a.data.empty()) {
        std::fprintf(stderr, "--data PATH (a uint16 .npy token file) is required\n");
        std::exit(2);  // NOLINT(concurrency-mt-unsafe): argument parsing, no threads yet
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

// carried: per layer the LRU and slot states the dumped step started from (truncated BPTT; empty without)
void dump(const std::string& dir,
          Model& model,
          const HostBatch& hb,
          cudaStream_t s,
          const std::vector<std::vector<float>>& carried,
          int nc = 0,
          float ul = 0.f) {
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
        for (int l = 0; l < c.L; ++l) { f << ((l != 0) ? "," : "") << c.hops_of(l); }
        f << "\nbank_slots ";
        for (int l = 0; l < c.L; ++l) { f << ((l != 0) ? "," : "") << c.slots_of(l); }
        f << "\nslot_hl_max_layers ";
        for (int l = 0; l < c.L; ++l) { f << ((l != 0) ? "," : "") << c.slot_hl_max_of(l); }
        f << "\nloss " << loss << "\nn_targets " << hb.n_targets << "\ncarry " << int(c.carry) << "\nfields "
          << ckpt_fields(c.dh) << "\nul_nc " << nc << "\nul_weight " << ul << "\n";
    }
    if (nc > 0) { write_bin(dir + "/cand.bin", hb.cand, size_t(c.B) * c.T * size_t(nc)); }
    for (size_t i = 0; i < carried.size(); ++i) {  // layer i / 2: 0 the LRU's h0, 1 the slot states
        write_bin(dir + "/l" + std::to_string(i / 2) + (i % 2 == 0 ? "_h0.bin" : "_binit.bin"), carried[i].data(),
                  carried[i].size());
    }
    {
        std::ofstream f(dir + "/manifest.txt");
        for (const auto& t : model.tensors()) { f << t.name << " " << t.off << " " << t.rows << " " << t.cols << "\n"; }
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

// --carry-check: batches 0 and 1 of a stream through a carrying model (forward_backward: the carry is updated in the
// backward), then the same rows joined into one row of 2T through a model without carry and with the same weights,
// scoring only the second half. The losses match when the carried LRU and slot states and the first key are the true
// continuation; they differ only by rounding (GEMMs of another height, which can flip a gate near its threshold).
int carry_check(const Config& c1, const TokenFile& file, uint64_t seed, int stream) {
    if (!c1.carry) {
        std::fprintf(stderr, "--carry-check needs --set data.stream_chunks=2 (or more)\n");
        return 2;
    }
    cudaStream_t s = nullptr;
    LBC_CUDA(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
    const int B = c1.B, T = c1.T;
    const size_t n1 = size_t(B) * T;
    int *d_ids = nullptr, *d_y = nullptr;
    uint8_t* d_reset = nullptr;
    float* d_inv = nullptr;
    LBC_CUDA(cudaMalloc(&d_ids, 2 * n1 * sizeof(int)));
    LBC_CUDA(cudaMalloc(&d_y, 2 * n1 * sizeof(int)));
    LBC_CUDA(cudaMalloc(&d_reset, 2 * n1));
    LBC_CUDA(cudaMalloc(&d_inv, sizeof(float)));
    const Batch batch{d_ids, d_y, d_reset, d_inv};
    std::vector<int> ids[2], ys[2];
    std::vector<uint8_t> rs[2];
    float inv1 = 0.f;
    int targets1 = 0;
    Loader ld(file, B, T, seed, 0, 0, -1, -1, false, 3, {}, stream);
    Model m1(c1, seed);
    for (int k = 0; k < 2; ++k) {
        HostBatch* hb = ld.next();
        ids[k].assign(hb->ids, hb->ids + n1);
        ys[k].assign(hb->y, hb->y + n1);
        rs[k].assign(hb->reset, hb->reset + n1);
        inv1 = *hb->inv_n;
        targets1 = hb->n_targets;
        LBC_CUDA(cudaMemcpy(d_ids, ids[k].data(), n1 * sizeof(int), cudaMemcpyHostToDevice));
        LBC_CUDA(cudaMemcpy(d_y, ys[k].data(), n1 * sizeof(int), cudaMemcpyHostToDevice));
        LBC_CUDA(cudaMemcpy(d_reset, rs[k].data(), n1, cudaMemcpyHostToDevice));
        LBC_CUDA(cudaMemcpy(d_inv, &inv1, sizeof(float), cudaMemcpyHostToDevice));
        ld.release(hb, s);
        m1.forward_backward(batch, s);
    }
    LBC_CUDA(cudaStreamSynchronize(s));
    float r1[2] = {0.f, 0.f};  // loss, accuracy
    LBC_CUDA(cudaMemcpy(r1, m1.metrics_device(), sizeof(r1), cudaMemcpyDeviceToHost));

    Config c2 = c1;
    c2.carry = false;
    c2.T = 2 * T;
    Model m2(c2, seed);
    LBC_CUDA(cudaMemcpy(m2.params(), m1.params(), m1.num_params() * sizeof(float), cudaMemcpyDeviceToDevice));
    m2.prepare_weights(s);
    std::vector<int> ids2(2 * n1), y2(2 * n1);
    std::vector<uint8_t> rs2(2 * n1);
    for (int b = 0; b < B; ++b) {
        for (int k = 0; k < 2; ++k) {
            const size_t src = size_t(b) * T, dst = size_t(b) * 2 * T + size_t(k) * T;
            std::copy_n(ids[k].begin() + long(src), T, ids2.begin() + long(dst));
            std::copy_n(rs[k].begin() + long(src), T, rs2.begin() + long(dst));
            if (k == 0) {
                std::fill_n(y2.begin() + long(dst), T, -1);  // the first half is context only
            } else {
                std::copy_n(ys[1].begin() + long(src), T, y2.begin() + long(dst));
            }
        }
    }
    LBC_CUDA(cudaMemcpy(d_ids, ids2.data(), 2 * n1 * sizeof(int), cudaMemcpyHostToDevice));
    LBC_CUDA(cudaMemcpy(d_y, y2.data(), 2 * n1 * sizeof(int), cudaMemcpyHostToDevice));
    LBC_CUDA(cudaMemcpy(d_reset, rs2.data(), 2 * n1, cudaMemcpyHostToDevice));
    m2.forward_loss(batch, s);
    LBC_CUDA(cudaStreamSynchronize(s));
    float r2[2] = {0.f, 0.f};
    LBC_CUDA(cudaMemcpy(r2, m2.metrics_device(), sizeof(r2), cudaMemcpyDeviceToHost));
    const double rel = std::abs(double(r1[0]) - double(r2[0])) / std::max(1e-12, std::abs(double(r2[0])));
    std::printf("carry check: %d rows, T %d, bank chunk %d, %d targets in the second half\n", B, T, c1.bank_chunk,
                targets1);
    std::printf("  carried (2 x %d): loss %.6f  accuracy %.5f\n  one row (%d):    loss %.6f  accuracy %.5f\n", T, r1[0],
                r1[1], 2 * T, r2[0], r2[1]);
    // An untrained model's loss barely depends on the context; the keys and reads of every layer do. Relative L2
    // distance to the one-row model over the second half of the first B / 2 rows, for the carrying model and for a
    // control without any carry: the same 2T rows through a model of B / 2 rows (other GEMM heights). Rounding alone
    // (a few bf16 ulps flipping top-2 routes and gates near their thresholds) moves the control as much; a wrong
    // carry would move the carrying model by O(1) from layer 0 on.
    Config c3 = c2;
    c3.B = B / 2;
    Model m3(c3, seed);
    LBC_CUDA(cudaMemcpy(m3.params(), m1.params(), m1.num_params() * sizeof(float), cudaMemcpyDeviceToDevice));
    m3.prepare_weights(s);
    m3.forward_loss(batch, s);  // the first B / 2 rows of the joined batch
    LBC_CUDA(cudaStreamSynchronize(s));
    const auto HD = size_t(c1.HD());
    double worst = 0.0, worst_ctl = 0.0;
    const char* names[5] = {"q1", "q2", "read 1", "read 2", "keys"};
    std::printf("  relative distance to the one-row model, carried / control without carry:\n");
    for (int n = 0; n < c1.L; ++n) {
        std::printf("  layer %d:", n);
        for (const int which : {4, 2, 3}) {
            const std::vector<bf16> x1 = m1.download_read(n, which, s), x2 = m2.download_read(n, which, s),
                                    x3 = m3.download_read(n, which, s);
            if (x1.empty()) { continue; }
            double d1 = 0.0, d3 = 0.0, nn = 0.0;
            for (int b = 0; b < B / 2; ++b) {
                for (int t = 0; t < T; ++t) {
                    for (size_t j = 0; j < HD; ++j) {
                        const size_t at = (size_t(b) * 2 * T + T + t) * HD + j;
                        const double u = __bfloat162float(x1[(size_t(b) * T + t) * HD + j]);
                        const double v = __bfloat162float(x2[at]), w = __bfloat162float(x3[at]);
                        d1 += (u - v) * (u - v);
                        d3 += (w - v) * (w - v);
                        nn += v * v;
                    }
                }
            }
            const double r1d = std::sqrt(d1 / std::max(nn, 1e-30)), r3d = std::sqrt(d3 / std::max(nn, 1e-30));
            worst = std::max(worst, r1d);
            worst_ctl = std::max(worst_ctl, r3d);
            std::printf("  %s %.2e / %.2e", names[which], r1d, r3d);
        }
        std::printf("\n");
    }
    // the carried model may not be farther from the one-row model than rounding moves the control (2x margin, and an
    // absolute floor for layers where the control happens to round identically)
    const bool ok = rel < 2e-3 && worst < std::max(2.0 * worst_ctl, 2e-2);
    std::printf("%s (relative loss difference %.2e, tolerance 2e-3; largest activation distance %.2e, control %.2e)\n",
                ok ? "PASS" : "FAIL", rel, worst, worst_ctl);
    LBC_CUDA(cudaFree(d_ids));
    LBC_CUDA(cudaFree(d_y));
    LBC_CUDA(cudaFree(d_reset));
    LBC_CUDA(cudaFree(d_inv));
    LBC_CUDA(cudaStreamDestroy(s));
    return ok ? 0 : 1;
}

}  // namespace

static int run(int argc, char** argv) {
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
        rc = with_keys(rc, a.sets);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "%s\n", e.what());
        return 2;
    }
    const Config c = rc.model_config();

    TokenFile const file(a.data);
    const int stream = std::max(1, rc.data.stream_chunks);
    if (a.carry_check) { return carry_check(c, file, a.seed, stream); }
    const int nc = rc.training.unlikelihood > 0 ? rc.training.unlikelihood_candidates : 0;
    const auto ul = float(rc.training.unlikelihood);
    Loader loader(file, c.B, c.T, a.seed, 0, 0, -1, -1, false, 3, {}, stream, nc);
    Model model(c, a.seed);
    cudaStream_t s = nullptr;
    LBC_CUDA(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
    const size_t ntok = size_t(c.B) * c.T;
    int *d_ids = nullptr, *d_y = nullptr, *d_cand = nullptr;
    uint8_t* d_reset = nullptr;
    float* d_inv_n = nullptr;
    LBC_CUDA(cudaMalloc(&d_inv_n, sizeof(float)));
    LBC_CUDA(cudaMalloc(&d_ids, ntok * sizeof(int)));
    LBC_CUDA(cudaMalloc(&d_y, ntok * sizeof(int)));
    LBC_CUDA(cudaMalloc(&d_reset, ntok));
    if (nc > 0) { LBC_CUDA(cudaMalloc(&d_cand, ntok * size_t(nc) * sizeof(int))); }

    size_t free_b = 0, total_b = 0;
    LBC_CUDA(cudaMemGetInfo(&free_b, &total_b));
    std::printf("model: %zu params, %d layers (hops", model.num_params(), c.L);
    for (int l = 0; l < c.L; ++l) { std::printf("%s%d", (l != 0) ? "," : " ", c.hops_of(l)); }
    std::printf("), D %d, LRU %d complex, %d heads x %d, %d slots, FFN %d, vocab %d x %d (%s, %s)\n", c.D, c.N, c.H,
                c.dh, SLOTS, c.F, c.V, c.E0, c.factorized ? "factorized" : "not factorized",
                c.tied ? "tied" : "untied");
    std::printf("batch %d x %d = %zu tokens per step; gate biases write %.2f read %.2f; %zu tokens on disk\n", c.B, c.T,
                ntok, c.gate_bias_w, c.gate_bias_r, file.size());
    std::printf("GPU memory in use after allocation: %.2f GiB of %.2f\n", (total_b - free_b) / 1073741824.0,
                total_b / 1073741824.0);

    auto copy_batch = [&](HostBatch* hb) {
        LBC_CUDA(cudaMemcpyAsync(d_ids, hb->ids, ntok * sizeof(int), cudaMemcpyHostToDevice, s));
        LBC_CUDA(cudaMemcpyAsync(d_y, hb->y, ntok * sizeof(int), cudaMemcpyHostToDevice, s));
        LBC_CUDA(cudaMemcpyAsync(d_reset, hb->reset, ntok, cudaMemcpyHostToDevice, s));
        LBC_CUDA(cudaMemcpyAsync(d_inv_n, hb->inv_n, sizeof(float), cudaMemcpyHostToDevice, s));
        Batch b{d_ids, d_y, d_reset, d_inv_n};
        if (nc > 0) {
            LBC_CUDA(cudaMemcpyAsync(d_cand, hb->cand, ntok * size_t(nc) * sizeof(int), cudaMemcpyHostToDevice, s));
            b.cand = d_cand;
            b.nc = nc;
            b.ul = ul;
        }
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
        HostBatch const keep = *hb;  // the pinned buffers stay valid until the slot is refilled (after this run)
        const Batch b = copy_batch(hb);
        std::vector<std::vector<float>> carried;  // the step's starting states: it overwrites them
        if (c.carry) {
            for (int l = 0; l < c.L; ++l) {
                carried.push_back(model.carried(l, 0, s));
                carried.push_back(model.carried(l, 1, s));
            }
        }
        model.forward_backward(b, s);
        LBC_CUDA(cudaStreamSynchronize(s));
        dump(a.dump, model, keep, s, carried, nc, ul);
        loader.release(hb, s);
        return 0;
    }

    const int total = a.warmup + a.steps;
    float* d_hist = nullptr;
    LBC_CUDA(cudaMalloc(&d_hist, total * sizeof(float)));
    cudaEvent_t t0 = nullptr, t1 = nullptr;
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
            if (step_graph == nullptr) {
                cudaGraph_t g = nullptr;
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
    for (int i = 0; i < total; i += std::max(1, total / 12)) { std::printf(" [%d] %.3f", i + 1, hist[i]); }
    std::printf(" [%d] %.3f\n", total, hist[total - 1]);

    const std::vector<float> g = model.gate_stats(s);
    std::printf("open gates (last step), per layer write / read 1 / read 2:");
    for (int n = 0; n < c.L; ++n) {
        std::printf("  L%d %.2f %.2f %.2f", n, g[3 * size_t(n)], g[3 * n + 1], g[3 * n + 2]);
    }
    std::printf("\n");

    // the last step's read gates: how often all heads of a token are closed (rows a GEMM could skip)
    if (a.gate_report) {
        std::printf("gate report (last step): per layer, hop: open reads, tokens with every head closed\n");
        for (int n = 0; n < c.L; ++n) {
            std::vector<uint8_t> ridx, gw, gr, gr2;
            model.decisions(n, ridx, gw, gr, gr2, s);
            for (int hop = 0; hop < 2; ++hop) {
                const std::vector<uint8_t>& gates = hop == 0 ? gr : gr2;
                size_t open = 0, all_closed = 0;
                for (size_t r = 0; r < ntok; ++r) {
                    int o = 0;
                    for (int h = 0; h < c.H; ++h) { o += gates[r * c.H + h]; }
                    open += o;
                    all_closed += static_cast<size_t>(o == 0);
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
        double total_sq = 0;
        for (const auto& t : model.tensors()) {
            double ss = 0;
            for (size_t i = 0; i < size_t(t.rows) * t.cols; ++i) { ss += double(gh[t.off + i]) * gh[t.off + i]; }
            total_sq += ss;
            norms.emplace_back(std::sqrt(ss), t.name);
        }
        std::ranges::sort(std::ranges::reverse_view(norms));
        std::printf("gradient norm (profiled step) %.4g; largest tensors:", std::sqrt(total_sq));
        for (size_t i = 0; i < std::min<size_t>(14, norms.size()); ++i) {
            std::printf(" %s %.3g", norms[i].second.c_str(), norms[i].first);
        }
        std::printf("\n");
        // W_s's columns: [key (H*DH) | query (H*DH) | read gate (H) | write gate (H)]
        const int HD = c.H * c.dh;
        const int groups[5] = {0, HD, 2 * HD, 2 * HD + c.H, 2 * HD + 2 * c.H};
        const char* gname[4] = {"key", "query", "read gate", "write gate"};
        for (const auto& t : model.tensors()) {
            if (t.name.size() < 3 || !t.name.ends_with("_Ws")) { continue; }
            std::printf("  %s by columns:", t.name.c_str());
            for (int k = 0; k < 4; ++k) {
                double ss = 0;
                for (int r = 0; r < t.rows; ++r) {
                    for (int j = groups[k]; j < groups[k + 1]; ++j) {
                        ss += double(gh[t.off + size_t(r) * t.cols + j]) * gh[t.off + size_t(r) * t.cols + j];
                    }
                }
                std::printf(" %s %.3g", gname[k], std::sqrt(ss));
            }
            std::printf("\n");
        }
    }
    float sum = 0.f;
    const auto prof = model.profile_ms();
    for (const auto& [cat, t] : prof) { sum += t; }
    std::printf("GPU time by category (one profiled step, %.1f ms in kernels):\n", sum);
    for (const auto& [cat, t] : prof) { std::printf("  %-18s %8.2f ms  %5.1f%%\n", cat.c_str(), t, 100.f * t / sum); }
    LBC_CUDA(cudaMemGetInfo(&free_b, &total_b));
    std::printf("GPU memory in use: %.2f GiB\n", (total_b - free_b) / 1073741824.0);
    return 0;
}

int main(int argc, char** argv) {
    try {
        return run(argc, argv);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "train_bench: %s\n", e.what());
        return 1;
    }
}
