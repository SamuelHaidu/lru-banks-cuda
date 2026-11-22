// RL subcommands of `lbc`: test-calcx (the ports' oracles), decode-check (incremental decoding vs the training
// forward), grpo (the RL trainer).

#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <type_traits>
#include <stdexcept>
#include <string>
#include <vector>

#include "../src/calcx.hpp"
#include "../src/checkpoint.hpp"
#include "../src/decode.hpp"
#include "../src/run_config.hpp"
#include "lbc/check.hpp"
#include "../src/data.hpp"
#include "../src/json.hpp"
#include "../src/tokenizer.hpp"
#include "lbc_rl.hpp"

namespace lbc {
namespace {

// a little-endian int64 2-D .npy (the categories files: start, length, category)
std::vector<int64_t> read_npy_i64(const std::string& path, int& cols) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot read " + path);
    char magic[8];
    f.read(magic, 8);
    uint16_t hl = 0;
    if (magic[6] == 1) {
        f.read(reinterpret_cast<char*>(&hl), 2);
    } else {
        uint32_t h4 = 0;
        f.read(reinterpret_cast<char*>(&h4), 4);
        hl = uint16_t(h4);
    }
    std::string header(hl, ' ');
    f.read(header.data(), hl);
    if (header.find("<i8") == std::string::npos) throw std::runtime_error(path + ": not int64");
    const size_t a = header.find('(') + 1, b = header.find(')');
    const std::string shape = header.substr(a, b - a);
    long rows = std::atol(shape.c_str());
    const size_t comma = shape.find(',');
    cols = comma != std::string::npos && comma + 1 < shape.size() ? std::atoi(shape.c_str() + comma + 1) : 1;
    if (cols <= 0) cols = 1;
    std::vector<int64_t> v(static_cast<size_t>(rows) * size_t(cols));
    f.read(reinterpret_cast<char*>(v.data()), std::streamsize(v.size() * 8));
    return v;
}

std::vector<std::string> doc_texts(const std::string& path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("cannot read " + path);
    std::vector<std::string> out;
    std::string line;
    while (std::getline(f, line))
        if (!line.empty()) out.push_back(json::parse(line).find("text")->str);
    return out;
}

}  // namespace

// lbc test-calcx DATASET_DIR TOKENIZER_JSON [--split train] [--limit N]
int cmd_test_calcx(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: lbc test-calcx DATASET_DIR TOKENIZER_JSON [--split S] [--limit N]\n");
        return 2;
    }
    const std::string dir = argv[0];
    std::string split = "train";
    long limit = 0;
    for (int i = 2; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--split" && i + 1 < argc) split = argv[++i];
        else if (a == "--limit" && i + 1 < argc) limit = std::atol(argv[++i]);
    }
    const auto t0 = std::chrono::steady_clock::now();
    Tokenizer tok(argv[1]);
    const std::vector<std::string> texts = doc_texts(dir + "/" + split + ".docs.jsonl");
    int cols = 0;
    const std::vector<int64_t> cat = read_npy_i64(dir + "/" + split + ".categories.npy", cols);
    TokenFile tf(dir + "/" + split + ".npy");
    const size_t n = limit > 0 ? std::min(texts.size(), size_t(limit)) : texts.size();
    if (cat.size() / size_t(cols) != texts.size()) throw std::runtime_error("docs and categories differ in count");
    size_t dec_bad = 0, enc_bad = 0, calls = 0, calc_bad = 0, ans_bad = 0, rew_bad = 0;
    std::vector<std::string> examples;
    for (size_t d = 0; d < n; ++d) {
        const int64_t st = cat[d * size_t(cols)], len = cat[d * size_t(cols) + 1];
        std::vector<int> ids(size_t(len - 2));
        for (int64_t k = 0; k < len - 2; ++k) ids[size_t(k)] = tf.data()[st + 1 + k];
        const std::string& t = texts[d];
        if (tok.decode(ids) != t) {
            ++dec_bad;
            if (examples.size() < 6) examples.push_back("decode differs: doc " + std::to_string(d));
        }
        if (tok.encode(t) != ids) {
            ++enc_bad;
            if (examples.size() < 6) {
                const std::vector<int> e = tok.encode(t);
                size_t k = 0;
                while (k < e.size() && k < ids.size() && e[k] == ids[k]) ++k;
                const size_t a = k >= 3 ? k - 3 : 0;
                std::vector<int> want(ids.begin() + long(a), ids.begin() + long(std::min(ids.size(), k + 4)));
                std::vector<int> got(e.begin() + long(a), e.begin() + long(std::min(e.size(), k + 4)));
                examples.push_back("encode differs: doc " + std::to_string(d) + " near [" + tok.decode(want) +
                                   "] vs [" + tok.decode(got) + "]");
            }
        }
        // every calculator call and its recorded result
        size_t p = 0;
        while ((p = t.find("<call>", p)) != std::string::npos) {
            const size_t e = t.find("</call></assistant><result>", p);
            if (e == std::string::npos) break;
            const size_t r0 = e + 27, r1 = t.find("</result>", r0);
            ++calls;
            try {
                const auto expr = calcx::find_call(t.substr(p, e + 7 - p));
                const std::string got = expr ? calcx::calculator(*expr) : "<no call>";
                if (got != t.substr(r0, r1 - r0)) {
                    ++calc_bad;
                    if (examples.size() < 12)
                        examples.push_back("calculator(" + (expr ? *expr : "?") + ") = " + got + ", recorded " +
                                           t.substr(r0, r1 - r0));
                }
            } catch (const std::exception& ex) {
                ++calc_bad;
            }
            p = r1;
        }
    }
    // reward: each reference solution, as a model output, is correct
    const std::vector<calcx::Question> qs = calcx::load_questions(dir + "/" + split + ".docs.jsonl");
    const size_t nq = limit > 0 ? std::min(qs.size(), size_t(limit)) : qs.size();
    for (size_t i = 0; i < nq; ++i) {
        const std::string& t = texts[i];
        const std::string out = t.substr(t.find("</user>") + 7 + 11);
        int c = 0;
        for (size_t p = 0; (p = out.find("<call>", p)) != std::string::npos; ++p) ++c;
        const auto pred = calcx::answer(out);
        if (!calcx::correct(pred, qs[i].gold)) ++ans_bad;
        if (calcx::reward(pred, out, c, qs[i], 0.5) != 1.0) ++rew_bad;
    }
    const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::printf("test-calcx %s/%s: %zu documents (%.1f s)\n", dir.c_str(), split.c_str(), n, secs);
    std::printf("  tokenizer: decode mismatches %zu, encode mismatches %zu\n", dec_bad, enc_bad);
    std::printf("  calculator: %zu calls, %zu results differ\n", calls, calc_bad);
    std::printf("  reward: %zu questions, %zu reference answers not correct, %zu rewards != 1\n", nq, ans_bad, rew_bad);
    for (const auto& e : examples) std::printf("    %s\n", e.c_str());
    return (dec_bad || enc_bad || calc_bad || ans_bad) ? 1 : 0;
}

// lbc decode-check RUN_DIR [--checkpoint PATH] [--rows B]: per-token loss of validation rows from the training
// forward and from incremental decoding (one token at a time, state carried), compared position by position
int cmd_decode_check(int argc, char** argv) {
    if (argc < 1) {
        std::fprintf(stderr, "usage: lbc decode-check RUN_DIR [--checkpoint PATH] [--rows B]\n");
        return 2;
    }
    const std::string run = argv[0];
    std::string path = run + "/model.lbc";
    int rows = 8;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--checkpoint" && i + 1 < argc) path = argv[++i];
        else if (a == "--rows" && i + 1 < argc) rows = std::atoi(argv[++i]);
    }
    const CheckpointInfo info = read_checkpoint_info(path);
    const RunConfig rc = RunConfig::from_json(info.config);
    const DatasetMeta meta = DatasetMeta::load(rc.data.dataset);
    Config c = rc.model_config();
    c.B = rows;
    const int T = c.T, V = c.V;
    cudaStream_t s;
    LBC_CUDA(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
    Model model(c, uint64_t(rc.data.seed));
    load_checkpoint(path, model, s);
    model.keep_row_loss();
    TokenFile file(meta.validation_file);
    ValidationSet vs(file, c.B, T, 1, uint64_t(rc.data.seed), meta.doc_token_id, meta.score_after_token,
                     meta.score_prev_token, meta.row_length > 0,
                     Span{meta.span_open, meta.span_close});
    const HostBatch& hb = vs.batch(0);
    const size_t ntok = size_t(c.B) * T;
    int *d_ids, *d_y, *d_yt;
    uint8_t* d_reset;
    float *d_inv_n, *d_one, *d_scr, *d_rl;
    LBC_CUDA(cudaMalloc(&d_ids, ntok * 4));
    LBC_CUDA(cudaMalloc(&d_y, ntok * 4));
    LBC_CUDA(cudaMalloc(&d_yt, size_t(rows) * 4));
    LBC_CUDA(cudaMalloc(&d_reset, ntok));
    LBC_CUDA(cudaMalloc(&d_inv_n, 4));
    LBC_CUDA(cudaMalloc(&d_one, 4));
    LBC_CUDA(cudaMalloc(&d_scr, 8));
    LBC_CUDA(cudaMalloc(&d_rl, size_t(rows) * 4));
    const float one = 1.f;
    LBC_CUDA(cudaMemcpy(d_ids, hb.ids, ntok * 4, cudaMemcpyHostToDevice));
    LBC_CUDA(cudaMemcpy(d_y, hb.y, ntok * 4, cudaMemcpyHostToDevice));
    LBC_CUDA(cudaMemcpy(d_reset, hb.reset, ntok, cudaMemcpyHostToDevice));
    LBC_CUDA(cudaMemcpy(d_inv_n, hb.inv_n, 4, cudaMemcpyHostToDevice));
    LBC_CUDA(cudaMemcpy(d_one, &one, 4, cudaMemcpyHostToDevice));
    model.forward_loss(Batch{d_ids, d_y, d_reset, d_inv_n}, s);
    const std::vector<float> ref = model.row_loss(s);

    if (std::getenv("LBC_FWD_B1")) {  // the training forward with one row per batch, against the B-row forward
        Config c1 = c;
        c1.B = 1;
        Model m1(c1, uint64_t(rc.data.seed));
        load_checkpoint(path, m1, s);
        m1.keep_row_loss();
        double sd = 0, md = 0;
        long nn = 0, bg = 0;
        for (int b = 0; b < rows; ++b) {
            const size_t o = size_t(b) * T;
            const float inv = 1.f;
            LBC_CUDA(cudaMemcpy(d_inv_n, &inv, 4, cudaMemcpyHostToDevice));
            m1.forward_loss(Batch{d_ids + o, d_y + o, d_reset + o, d_inv_n}, s);
            const std::vector<float> r1 = m1.row_loss(s);
            for (int t = 0; t < T; ++t)
                if (hb.y[o + size_t(t)] >= 0) {
                    const double d = std::fabs(double(r1[size_t(t)]) - ref[o + size_t(t)]);
                    sd += d;
                    md = std::max(md, d);
                    bg += d > 0.05;
                    ++nn;
                }
        }
        std::printf("  forward B=1 vs B=%d: |diff| mean %.5f, max %.4f, > 0.05: %.2f%%\n", rows, sd / nn, md,
                    100.0 * bg / nn);
    }
    Decoder dec(model, rows, rows);
    std::printf("decode-check %s (epoch %d): %d rows x %d tokens; decoder state %.1f MB\n", path.c_str(), info.epoch,
                rows, T, dec.state_bytes() / 1e6);
    const size_t nr = static_cast<size_t>(rows);
    std::vector<int> ids(nr), slot(nr), tpos(nr), yt(nr);
    std::vector<uint8_t> rst(nr);
    std::vector<float> got(nr);
    double max_d = 0, sum_d = 0, sum_ref = 0;
    long n = 0, big = 0;
    const auto t0 = std::chrono::steady_clock::now();
    for (int t = 0; t < T; ++t) {
        for (int b = 0; b < rows; ++b) {
            const size_t i = size_t(b) * T + t;
            ids[size_t(b)] = hb.ids[i];
            slot[size_t(b)] = b;
            tpos[size_t(b)] = t;
            rst[size_t(b)] = hb.reset[i];
            yt[size_t(b)] = hb.y[i];
        }
        dec.step(ids.data(), slot.data(), tpos.data(), rst.data(), rows, 1.f, nullptr, nullptr, nullptr, nullptr, s);
        LBC_CUDA(cudaMemcpyAsync(d_yt, yt.data(), size_t(rows) * 4, cudaMemcpyHostToDevice, s));
        ce_rows(dec.logits(), d_yt, nullptr, d_scr, d_scr + 1, rows, V, d_one, s, d_rl);
        LBC_CUDA(cudaMemcpyAsync(got.data(), d_rl, size_t(rows) * 4, cudaMemcpyDeviceToHost, s));
        LBC_CUDA(cudaStreamSynchronize(s));
        for (int b = 0; b < rows; ++b) {
            if (yt[size_t(b)] < 0) continue;
            const double d = std::fabs(double(got[size_t(b)]) - double(ref[size_t(b) * T + t]));
            max_d = std::max(max_d, d);
            sum_d += d;
            sum_ref += ref[size_t(b) * T + t];
            big += d > 0.05;
            ++n;
            if (std::getenv("LBC_DEBUG") && b == 0 && (t < 12 || d > 0.3))
                std::printf("    row 0 t %d: forward %.4f decode %.4f\n", t, ref[size_t(b) * T + t], got[size_t(b)]);
        }
    }
    const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::printf("  %ld targets: mean nll %.4f; |decode - forward| mean %.5f, max %.4f, > 0.05: %ld (%.2f%%)\n", n,
                sum_ref / std::max(1L, n), sum_d / std::max(1L, n), max_d, big, 100.0 * big / std::max(1L, n));
    std::printf("  decoding: %.2f s for %d steps (%.2f ms/step at %d rows)\n", secs, T, 1e3 * secs / T, rows);
    // pass: within the training forward's own sensitivity (the binary gates flip on last-bit GEMM differences:
    // the same forward at 1 row vs 8 rows differs by a mean 0.043 per token on the calcx30 checkpoint)
    return sum_d / std::max(1L, n) < 0.05 ? 0 : 1;
}

// ---- GRPO -----------------------------------------------------------------------------------------------------
//
// lbc grpo RUN_DIR: RUN_DIR/grpo.json holds the settings (below, every key optional but init_from / data paths);
// RUN_DIR gets config.json (the init checkpoint's), grpo_metrics.csv, eval.csv, model.lbc (latest), checkpoints/.
// Resumes from model.lbc (its "epoch" field counts the updates done).

namespace {

struct GrpoCfg {
    std::string init_from, train_docs, test_docs, tokenizer, pretrain_dataset;
    std::vector<std::string> sources = {"gsm8k", "mawps", "ape210k"};
    int updates = 100, questions = 512, group = 8, chunk = 512, max_new = 200, micro_rows = 64;
    int target_mixed = 0;          // dynamic sampling: roll out more chunks until this many mixed groups (0: off)
    int max_chunks = 4;
    double temperature = 1.0, lr = 5e-5, beta2 = 0.99, clip = 1.0, partial = 0.5, tis_cap = 2.0;
    int warmup = 3;
    std::string mode = "grpo";     // "grpo": r - mean(group); "raft": weight 1 on correct rollouts only (no negatives)
    int ppo_epochs = 1;            // optimizer steps per rollout batch (> 1: PPO-clipped reuse of the samples)
    double clip_low = 0.2, clip_high = 0.28;  // with ppo_epochs > 1 (DAPO clip-higher)
    bool no_std = true;            // Dr. GRPO: r - mean (else / max(std, std_floor))
    double std_floor = 0.1;
    bool greso = true;             // skip questions whose recent groups had no reward variance
    double greso_p = 0.5, greso_max = 0.9;
    int eval_every = 5, eval_per_source = 100, pretrain_val_steps = 8, checkpoint_every = 10;
    bool save = true;              // false: no checkpoints (ablations: the results are in the CSVs)
    // collapse guards
    int max_repeat = 0;            // > 0: a rollout with a run of more than this many identical generated tokens gets
                                   // reward 0 and is never a positive (RAFT: dropped; GRPO: pushed down)
    int sample_questions = 0;      // > 0: no training; write the model's clean, correct, unique solutions on that many
    std::string sample_out;        // training questions to this jsonl (source, id, text = prompt + generation)
    double novelty = 0;            // > 0: a correct solution's reward + novelty / (copies of it in the group)
    bool unique = false;           // positives with the same calculator-call sequence and answer count once per group
    int replay_rows = 0;           // pretraining rows (a multiple of micro_rows) added to every update's gradient
    double replay_coef = 0.1;      // their loss weight relative to the RL loss (per token, scaled to the RL tokens)
    uint64_t seed = 0;
};

GrpoCfg load_grpo(const std::string& path) {
    const json::Value v = json::read_file(path);
    GrpoCfg c;
    for (const auto& [k, x] : v.obj) {
        auto str = [&](std::string& o) { o = x.str; };
        auto num = [&](auto& o) { o = static_cast<std::decay_t<decltype(o)>>(x.num); };
        if (k == "init_from") str(c.init_from);
        else if (k == "train_docs") str(c.train_docs);
        else if (k == "test_docs") str(c.test_docs);
        else if (k == "tokenizer") str(c.tokenizer);
        else if (k == "pretrain_dataset") str(c.pretrain_dataset);
        else if (k == "sources") {
            c.sources.clear();
            for (const auto& e : x.arr) c.sources.push_back(e.str);
        } else if (k == "updates") num(c.updates);
        else if (k == "questions") num(c.questions);
        else if (k == "group") num(c.group);
        else if (k == "chunk") num(c.chunk);
        else if (k == "max_new") num(c.max_new);
        else if (k == "micro_rows") num(c.micro_rows);
        else if (k == "target_mixed") num(c.target_mixed);
        else if (k == "max_chunks") num(c.max_chunks);
        else if (k == "temperature") num(c.temperature);
        else if (k == "lr") num(c.lr);
        else if (k == "beta2") num(c.beta2);
        else if (k == "clip") num(c.clip);
        else if (k == "partial") num(c.partial);
        else if (k == "tis_cap") num(c.tis_cap);
        else if (k == "warmup") num(c.warmup);
        else if (k == "mode") str(c.mode);
        else if (k == "ppo_epochs") num(c.ppo_epochs);
        else if (k == "clip_low") num(c.clip_low);
        else if (k == "clip_high") num(c.clip_high);
        else if (k == "no_std") c.no_std = x.b;
        else if (k == "std_floor") num(c.std_floor);
        else if (k == "greso") c.greso = x.b;
        else if (k == "greso_p") num(c.greso_p);
        else if (k == "greso_max") num(c.greso_max);
        else if (k == "eval_every") num(c.eval_every);
        else if (k == "eval_per_source") num(c.eval_per_source);
        else if (k == "pretrain_val_steps") num(c.pretrain_val_steps);
        else if (k == "checkpoint_every") num(c.checkpoint_every);
        else if (k == "seed") num(c.seed);
        else if (k == "save") c.save = x.b;
        else if (k == "max_repeat") num(c.max_repeat);
        else if (k == "unique") c.unique = x.b;
        else if (k == "novelty") num(c.novelty);
        else if (k == "sample_questions") num(c.sample_questions);
        else if (k == "sample_out") str(c.sample_out);
        else if (k == "replay_rows") num(c.replay_rows);
        else if (k == "replay_coef") num(c.replay_coef);
        else throw std::runtime_error(path + ": unknown key " + k);
    }
    if (c.init_from.empty() || c.train_docs.empty() || c.test_docs.empty() || c.tokenizer.empty())
        throw std::runtime_error(path + ": init_from, train_docs, test_docs and tokenizer are required");
    return c;
}

// the text without abandoned calls: a <call> followed by another <call> before any </call> (the model gave up on
// it and started over) is cut up to the next <call>
std::string strip_abandoned(const std::string& text) {
    std::string out = text;
    for (bool changed = true; changed;) {
        changed = false;
        for (size_t p = out.find("<call>"); p != std::string::npos; p = out.find("<call>", p + 1)) {
            const size_t q = out.find("<call>", p + 6), e = out.find("</call>", p + 6);
            if (q != std::string::npos && (e == std::string::npos || q < e)) {
                out.erase(p, q - p);
                changed = true;
                break;
            }
        }
    }
    return out;
}

// a solution's identity: its complete calculator calls' expressions (spaces removed) and the final answer; abandoned
// or unclosed calls do not count (otherwise junk text in a broken call makes every rollout look new: log T44)
std::string solution_sig(const std::string& raw) {
    const std::string text = strip_abandoned(raw);
    std::string sig;
    for (size_t p0 = text.find("<call>calculator(expression=\""); p0 != std::string::npos;
         p0 = text.find("<call>calculator(expression=\"", p0 + 1)) {
        const size_t a = p0 + 29, b = text.find('"', a);
        if (b == std::string::npos || text.compare(b, 9, "\")</call>") != 0) continue;
        for (size_t i = a; i < b; ++i)
            if (text[i] != ' ') sig += text[i];
        sig += ';';
    }
    const auto ans = calcx::answer(text);
    return sig + (ans ? *ans : std::string("-"));
}

uint64_t hash3(uint64_t a, uint64_t b, uint64_t c) {
    uint64_t z = a * 0x9E3779B97F4A7C15ULL ^ (b + 0x632BE59BD9B4E019ULL) * 0xC2B2AE3D27D4EB4FULL ^ c;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

double secs_since(std::chrono::steady_clock::time_point t) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t).count();
}

struct Rollout {
    int q = 0;                   // question index
    std::vector<int> ids;        // <bos>, the prompt, everything after it
    int n_prompt = 0;
    std::vector<uint8_t> own;    // per generated token: sampled by the model (not a forced tool result)
    std::vector<float> lp;       // per generated token: the rollout's log-probability (own tokens)
    int calls = 0;
    bool finished = false, bad_call = false, degen = false;
    double ent_sum = 0;
    int ent_n = 0;
    std::string text;
    double reward = 0;
    bool ok = false;
};

struct Timers {
    double prefill = 0, decode = 0, host = 0;
};

// The agent loop of model/calcx.py:rollout for len(qs) x group rollouts, through the decoder: each question's prompt
// (but its last token) is prefilled once in slot q * group, copied to the group's other slots, then every row feeds
// the last prompt token and samples on.
std::vector<Rollout> rollout(Decoder& dec, const Tokenizer& tok, const std::vector<const calcx::Question*>& qs,
                             const std::vector<int>& qidx, int group, int max_new, float temperature, uint64_t seed,
                             cudaStream_t s, Timers& tm) {
    const int Q = int(qs.size()), N = Q * group;
    const int bos = 0, eos = 1, a_close = tok.id("</assistant>");
    std::vector<Rollout> out(static_cast<size_t>(N));
    std::vector<std::vector<int>> prompt(static_cast<size_t>(Q));
    auto t0 = std::chrono::steady_clock::now();
    for (int q = 0; q < Q; ++q) {
        prompt[size_t(q)] = {bos};
        const std::vector<int> e = tok.encode(qs[size_t(q)]->prompt);
        prompt[size_t(q)].insert(prompt[size_t(q)].end(), e.begin(), e.end());
    }
    std::vector<int> ids, slot, tpos, tok_out;
    std::vector<uint8_t> rst;
    std::vector<float> lp_out, ent_out;
    std::vector<uint64_t> key;
    // prefill: every prompt token but the last, one slot per question
    int Lmax = 0;
    for (auto& p : prompt) Lmax = std::max(Lmax, int(p.size()) - 1);
    for (int t = 0; t < Lmax; ++t) {
        ids.clear();
        slot.clear();
        tpos.clear();
        rst.clear();
        for (int q = 0; q < Q; ++q)
            if (t < int(prompt[size_t(q)].size()) - 1) {
                ids.push_back(prompt[size_t(q)][size_t(t)]);
                slot.push_back(q * group);
                tpos.push_back(t);
                rst.push_back(t == 0 ? 1 : 0);
            }
        dec.step(ids.data(), slot.data(), tpos.data(), rst.data(), int(ids.size()), 1.f, nullptr, nullptr, nullptr,
                 nullptr, s);
    }
    std::vector<int2> pairs;
    for (int q = 0; q < Q; ++q)
        for (int g = 1; g < group; ++g) pairs.push_back(make_int2(q * group + g, q * group));
    dec.copy_slots(pairs, s);
    LBC_CUDA(cudaStreamSynchronize(s));
    tm.prefill += secs_since(t0);

    // generation
    std::vector<int> next(static_cast<size_t>(N)), pos(static_cast<size_t>(N)), span(static_cast<size_t>(N));
    std::vector<std::vector<int>> forced(static_cast<size_t>(N));
    std::vector<std::vector<int>> gen(static_cast<size_t>(N));
    std::vector<uint8_t> done(static_cast<size_t>(N), 0);
    for (int r = 0; r < N; ++r) {
        const int q = r / group;
        out[size_t(r)].q = qidx[size_t(q)];
        out[size_t(r)].n_prompt = int(prompt[size_t(q)].size());
        next[size_t(r)] = prompt[size_t(q)].back();
        pos[size_t(r)] = int(prompt[size_t(q)].size()) - 1;
    }
    std::vector<int> live;
    for (int step = 0; step < max_new; ++step) {
        live.clear();
        for (int r = 0; r < N; ++r)
            if (!done[size_t(r)]) live.push_back(r);
        if (live.empty()) break;
        const int R = int(live.size());
        ids.resize(size_t(R));
        slot.resize(size_t(R));
        tpos.resize(size_t(R));
        rst.assign(size_t(R), 0);
        key.resize(size_t(R));
        tok_out.resize(size_t(R));
        lp_out.resize(size_t(R));
        ent_out.resize(size_t(R));
        for (int j = 0; j < R; ++j) {
            const int r = live[size_t(j)];
            ids[size_t(j)] = next[size_t(r)];
            slot[size_t(j)] = r;
            tpos[size_t(j)] = pos[size_t(r)];
            key[size_t(j)] = hash3(seed, uint64_t(r), uint64_t(step));
        }
        auto td = std::chrono::steady_clock::now();
        dec.step(ids.data(), slot.data(), tpos.data(), rst.data(), R, temperature, key.data(), tok_out.data(),
                 lp_out.data(), ent_out.data(), s);
        tm.decode += secs_since(td);
        auto th = std::chrono::steady_clock::now();
        for (int j = 0; j < R; ++j) {
            const int r = live[size_t(j)];
            Rollout& ro = out[size_t(r)];
            auto& f = forced[size_t(r)];
            const bool own = f.empty();
            int t;
            if (own) {
                t = tok_out[size_t(j)];
                if (t < 0) {  // the decoder produced no finite logits for this row
                    static int reported = 0;
                    if (reported++ < 5)
                        std::fprintf(stderr, "warning: non-finite logits: row %d step %d pos %d (gen %zu)\n", r,
                                     step, pos[size_t(r)], gen[size_t(r)].size());
                    done[size_t(r)] = 1;
                    continue;
                }
            } else {
                t = f.front();
                f.erase(f.begin());
            }
            gen[size_t(r)].push_back(t);
            ro.own.push_back(own ? 1 : 0);
            ro.lp.push_back(own ? lp_out[size_t(j)] : 0.f);
            if (own) {
                ro.ent_sum += ent_out[size_t(j)];
                ++ro.ent_n;
            }
            next[size_t(r)] = t;
            ++pos[size_t(r)];
            if (t == eos || t == bos) {
                done[size_t(r)] = 1;
                ro.finished = t == eos;
            } else if (t == a_close && own) {
                const auto& g = gen[size_t(r)];
                const std::string turn = tok.decode(g.data() + span[size_t(r)], g.size() - size_t(span[size_t(r)]));
                std::optional<std::string> expr;
                bool bad = false;
                try {
                    expr = calcx::find_call(turn);
                } catch (const std::exception&) {
                    bad = true;
                }
                if (expr || bad) {
                    const std::string res = bad ? "Error: bad call" : calcx::calc(*expr);
                    ++ro.calls;
                    f = tok.encode("<result>" + res + "</result><assistant>");
                    span[size_t(r)] = int(g.size() + f.size());
                } else {
                    ro.bad_call = turn.find("<call>") != std::string::npos;
                    done[size_t(r)] = 1;
                    ro.finished = true;
                }
            }
            if (int(gen[size_t(r)].size()) >= max_new) done[size_t(r)] = 1;
        }
        tm.host += secs_since(th);
    }
    for (int r = 0; r < N; ++r) {
        Rollout& ro = out[size_t(r)];
        ro.ids = prompt[size_t(r / group)];
        ro.ids.insert(ro.ids.end(), gen[size_t(r)].begin(), gen[size_t(r)].end());
        ro.text = tok.decode(gen[size_t(r)]);
    }
    return out;
}

// the longest run of one generated (own) token in a rollout
int longest_run(const Rollout& r) {
    int run = 0, best = 0, prev = -1;
    for (size_t i = 0; i < r.own.size(); ++i) {
        const int t = r.ids[size_t(r.n_prompt) + i];
        if (!r.own[i]) {
            prev = -1;
            run = 0;
            continue;
        }
        run = t == prev ? run + 1 : 1;
        prev = t;
        best = std::max(best, run);
    }
    return best;
}

size_t count_of(const std::string& s, const std::string& sub) {
    size_t n = 0;
    for (size_t p = s.find(sub); p != std::string::npos; p = s.find(sub, p + 1)) ++n;
    return n;
}

double pretrain_val(Model& model, ValidationSet& vs, int* d_ids, int* d_y, uint8_t* d_reset, float* d_inv_n,
                    cudaStream_t s, int T, int B) {
    double tot = 0;
    long n = 0;
    const size_t ntok = size_t(B) * T;
    for (int k = 0; k < vs.steps(); ++k) {
        const HostBatch& hb = vs.batch(k);
        LBC_CUDA(cudaMemcpyAsync(d_ids, hb.ids, ntok * 4, cudaMemcpyHostToDevice, s));
        LBC_CUDA(cudaMemcpyAsync(d_y, hb.y, ntok * 4, cudaMemcpyHostToDevice, s));
        LBC_CUDA(cudaMemcpyAsync(d_reset, hb.reset, ntok, cudaMemcpyHostToDevice, s));
        LBC_CUDA(cudaMemcpyAsync(d_inv_n, hb.inv_n, 4, cudaMemcpyHostToDevice, s));
        model.forward_loss(Batch{d_ids, d_y, d_reset, d_inv_n}, s);
        float r[2];
        LBC_CUDA(cudaMemcpyAsync(r, model.metrics_device(), 8, cudaMemcpyDeviceToHost, s));
        LBC_CUDA(cudaStreamSynchronize(s));
        tot += double(r[0]) * hb.n_targets;
        n += hb.n_targets;
    }
    return tot / std::max(1L, n);
}

}  // namespace

int cmd_grpo(int argc, char** argv) {
    if (argc < 1) {
        std::fprintf(stderr, "usage: lbc grpo RUN_DIR   (RUN_DIR/grpo.json: the settings)\n");
        return 2;
    }
    namespace fs = std::filesystem;
    const std::string run = argv[0];
    const GrpoCfg g = load_grpo(run + "/grpo.json");
    const std::string latest = run + "/model.lbc";
    const bool resume = fs::exists(latest);
    const CheckpointInfo info = read_checkpoint_info(resume ? latest : g.init_from);
    const RunConfig rc = RunConfig::from_json(info.config);
    const json::Value config_json = rc.to_json();
    if (!fs::exists(run + "/config.json")) json::write_file(run + "/config.json", config_json);
    Config c = rc.model_config();
    c.B = g.micro_rows;
    c.lr = float(g.lr);
    c.warmup = g.warmup;
    c.decay_start = c.decay_steps = 0;
    c.wd = 0.f;
    c.beta2 = float(g.beta2);
    c.clip = float(g.clip);
    const int T = c.T;
    cudaStream_t s;
    LBC_CUDA(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
    Model model(c, uint64_t(rc.data.seed));
    int done_updates = 0;
    double best_acc = -1;
    if (resume) {
        load_checkpoint(latest, model, s);
        done_updates = info.epoch;
        best_acc = info.best_val_loss;
    } else {
        load_weights(g.init_from, model, s);
    }
    Tokenizer tok(g.tokenizer);
    const std::vector<calcx::Question> train = calcx::load_questions(g.train_docs, g.sources, true);
    std::vector<calcx::Question> tests;
    {
        const std::vector<calcx::Question> all = calcx::load_questions(g.test_docs);
        std::map<std::string, std::vector<const calcx::Question*>> by;
        for (const auto& q : all) by[q.source].push_back(&q);
        for (auto& [src, v] : by) {  // a fixed sample per source
            std::vector<size_t> idx(v.size());
            for (size_t i = 0; i < idx.size(); ++i) idx[i] = i;
            std::sort(idx.begin(), idx.end(), [&](size_t a, size_t b) {
                return hash3(7, a, 0) < hash3(7, b, 0);
            });
            for (size_t i = 0; i < idx.size() && int(i) < g.eval_per_source; ++i) tests.push_back(*v[idx[i]]);
        }
    }
    const int slots = g.chunk * g.group;
    Decoder dec(model, std::max(slots, int(tests.size())), std::max(slots, int(tests.size())));
    std::unique_ptr<TokenFile> pfile;
    std::unique_ptr<ValidationSet> pval;
    if (!g.pretrain_dataset.empty() && g.pretrain_val_steps > 0) {
        const DatasetMeta pm = DatasetMeta::load(g.pretrain_dataset);
        pfile = std::make_unique<TokenFile>(pm.validation_file);
        pval = std::make_unique<ValidationSet>(*pfile, c.B, T, g.pretrain_val_steps, 42, pm.doc_token_id);
    }
    const size_t ntok = size_t(c.B) * T;
    std::unique_ptr<TokenFile> rfile;  // pretraining replay
    std::unique_ptr<Loader> rload;
    std::vector<float> hw_rep(ntok);
    if (g.replay_rows > 0) {
        if (g.pretrain_dataset.empty()) throw std::runtime_error("replay_rows needs pretrain_dataset");
        if (g.replay_rows % c.B) throw std::runtime_error("replay_rows must be a multiple of micro_rows");
        const DatasetMeta pm = DatasetMeta::load(g.pretrain_dataset);
        rfile = std::make_unique<TokenFile>(pm.train_file);
        rload = std::make_unique<Loader>(*rfile, c.B, T, g.seed + 11 + uint64_t(done_updates), pm.doc_token_id);
    }
    int *d_ids, *d_y;
    uint8_t* d_reset;
    float *d_inv_n, *d_w, *d_lp;
    LBC_CUDA(cudaMalloc(&d_ids, ntok * 4));
    LBC_CUDA(cudaMalloc(&d_y, ntok * 4));
    LBC_CUDA(cudaMalloc(&d_reset, ntok));
    LBC_CUDA(cudaMalloc(&d_inv_n, 4));
    LBC_CUDA(cudaMalloc(&d_w, ntok * 4));
    LBC_CUDA(cudaMalloc(&d_lp, ntok * 4));
    std::printf("GRPO %s from %s: %zu training questions (%zu sources), %zu test questions; %d x %d rollouts per chunk; "
                "decoder state %.2f GB; updates %d -> %d\n",
                run.c_str(), resume ? latest.c_str() : g.init_from.c_str(), train.size(), g.sources.size(),
                tests.size(), g.chunk, g.group, dec.state_bytes() / 1e9, done_updates, g.updates);
    std::fflush(stdout);

    auto evaluate = [&](int update) {
        const auto t0 = std::chrono::steady_clock::now();
        std::vector<const calcx::Question*> qs;
        std::vector<int> qi;
        for (size_t i = 0; i < tests.size(); ++i) {
            qs.push_back(&tests[i]);
            qi.push_back(int(i));
        }
        Timers tm;
        const std::vector<Rollout> ro = rollout(dec, tok, qs, qi, 1, g.max_new, 0.f, 0, s, tm);
        std::map<std::string, std::pair<int, int>> by;
        int ok = 0, answered = 0;
        for (const auto& r : ro) {
            const auto pred = calcx::answer(r.text);
            const bool k = calcx::correct(pred, tests[size_t(r.q)].gold);
            ok += k;
            answered += bool(pred);
            auto& b = by[tests[size_t(r.q)].source];
            b.first += k;
            ++b.second;
        }
        const double pv = pval ? pretrain_val(model, *pval, d_ids, d_y, d_reset, d_inv_n, s, T, c.B) : -1;
        const double acc = double(ok) / std::max<size_t>(1, ro.size());
        std::string srcs;
        for (auto& [k, v] : by) {
            char b[64];
            std::snprintf(b, sizeof b, " %s %.3f", k.c_str(), double(v.first) / v.second);
            srcs += b;
        }
        std::printf("[eval %d] accuracy %.4f (%s ) answered %.3f pretrain_val %.4f (%.0f s)\n", update, acc,
                    srcs.c_str(), double(answered) / ro.size(), pv, secs_since(t0));
        std::fflush(stdout);
        const bool header = !fs::exists(run + "/eval.csv");
        std::ofstream f(run + "/eval.csv", std::ios::app);
        if (header) {
            f << "update,accuracy,answered,pretrain_val";
            for (auto& [k, v] : by) f << "," << k;
            f << ",seconds\n";
        }
        f << update << "," << acc << "," << double(answered) / ro.size() << "," << pv;
        for (auto& [k, v] : by) f << "," << double(v.first) / v.second;
        f << "," << secs_since(t0) << "\n";
        {  // every eval's outputs, for inspection
            std::ofstream d(run + "/eval_" + std::to_string(update) + ".jsonl");
            for (const auto& r : ro) {
                std::string t;
                for (const char ch : r.text) {
                    if (ch == '"' || ch == '\\') t += '\\', t += ch;
                    else if (ch == '\n') t += "\\n";
                    else if (static_cast<unsigned char>(ch) < 0x20) t += ' ';
                    else t += ch;
                }
                d << "{\"source\": \"" << tests[size_t(r.q)].source << "\", \"ok\": "
                  << (calcx::correct(calcx::answer(r.text), tests[size_t(r.q)].gold) ? "true" : "false")
                  << ", \"finished\": " << (r.finished ? "true" : "false") << ", \"calls\": " << r.calls
                  << ", \"gen\": " << (long(r.ids.size()) - r.n_prompt) << ", \"text\": \"" << t << "\"}\n";
            }
        }
        return acc;
    };

    // per question: consecutive groups without reward variance (GRESO-style skipping)
    std::vector<int> zv(train.size(), 0);
    std::vector<size_t> order(train.size());
    size_t cursor = 0;
    uint64_t pass = 0;
    auto reshuffle = [&]() {
        for (size_t i = 0; i < order.size(); ++i) order[i] = i;
        std::sort(order.begin(), order.end(),
                  [&](size_t a, size_t b) { return hash3(g.seed + 11, pass, a) < hash3(g.seed + 11, pass, b); });
        ++pass;
        cursor = 0;
    };
    reshuffle();
    if (g.sample_questions > 0) {  // self-distillation data: the model's own clean, correct, unique solutions
        if (g.sample_out.empty()) throw std::runtime_error("sample_questions needs sample_out");
        std::ofstream f(g.sample_out);
        long rolls = 0, ok = 0, kept = 0, dirty = 0;
        const auto t0 = std::chrono::steady_clock::now();
        for (int done = 0; done < g.sample_questions;) {
            std::vector<const calcx::Question*> qs;
            std::vector<int> qi;
            while (int(qs.size()) < std::min(g.chunk, g.sample_questions - done)) {
                if (cursor >= order.size()) reshuffle();
                const size_t i = order[cursor++];
                qs.push_back(&train[i]);
                qi.push_back(int(i));
            }
            Timers tm;
            const std::vector<Rollout> ro = rollout(dec, tok, qs, qi, g.group, g.max_new, float(g.temperature),
                                                    hash3(g.seed + 5, uint64_t(done), 0), s, tm);
            for (size_t q = 0; q < qs.size(); ++q) {
                std::set<std::string> seen;
                for (int k = 0; k < g.group; ++k) {
                    const Rollout& r = ro[q * size_t(g.group) + size_t(k)];
                    ++rolls;
                    if (!r.finished || !calcx::correct(calcx::answer(r.text), qs[q]->gold)) continue;
                    ++ok;
                    // clean: abandoned calls cut out; then no long token runs and every <call> closed
                    const std::string clean = strip_abandoned(r.text);
                    if (longest_run(r) > std::max(1, g.max_repeat) ||
                        count_of(clean, "<call>") != count_of(clean, "</call>")) {
                        ++dirty;
                        continue;
                    }
                    if (!seen.insert(solution_sig(clean)).second) continue;
                    ++kept;
                    std::string t;
                    for (const char ch : qs[q]->prompt + clean) {
                        if (ch == '"' || ch == '\\') t += '\\', t += ch;
                        else if (ch == '\n') t += "\\n";
                        else if (ch == '\t') t += "\\t";
                        else if (static_cast<unsigned char>(ch) < 0x20) t += ' ';
                        else t += ch;
                    }
                    f << "{\"source\": \"" << qs[q]->source << "\", \"id\": \"" << qs[q]->id << "\", \"text\": \"" << t << "\"}\n";
                }
            }
            done += int(qs.size());
            std::printf("sampled %d questions: %ld rollouts, %ld correct, %ld dirty, %ld kept (%.0f s)\n", done, rolls,
                        ok, dirty, kept, secs_since(t0));
            std::fflush(stdout);
        }
        return 0;
    }
    // resume: skip the questions earlier updates drew (approximately: questions x updates)
    for (long k = 0; k < long(done_updates) * g.questions; ++k)
        if (++cursor >= order.size()) reshuffle();

    if (done_updates == 0) {
        const double a = evaluate(0);
        if (best_acc < 0) best_acc = a;
    }
    const bool header = !fs::exists(run + "/grpo_metrics.csv");
    std::ofstream csv(run + "/grpo_metrics.csv", std::ios::app);
    if (header)
        csv << "update,reward,accuracy,mixed_groups,solved_groups,questions,skipped,chunks,rollouts,train_rows,"
               "train_tokens,gen_tokens,calls,answered,entropy,loss,grad_norm,lr,t_prefill,t_decode,t_host,t_train,"
               "t_total\n";
    for (int u = done_updates; u < g.updates; ++u) {
        const auto t0 = std::chrono::steady_clock::now();
        Timers tm;
        std::vector<Rollout> all;
        std::vector<int> group_mixed;
        int skipped = 0, chunks = 0, questions = 0, mixed = 0, solved = 0;
        while (chunks < g.max_chunks) {
            std::vector<const calcx::Question*> qs;
            std::vector<int> qi;
            const int want = chunks == 0 ? g.questions : g.chunk;
            while (int(qs.size()) < std::min(want, g.chunk)) {
                if (cursor >= order.size()) reshuffle();
                const size_t i = order[cursor++];
                if (g.greso && zv[i] > 0) {
                    const double p = std::min(g.greso_max, 1.0 - std::pow(1.0 - g.greso_p, zv[i]));
                    const double u01 = double(hash3(g.seed + 3, i, cursor + pass * 1000003) >> 11) * 0x1.0p-53;
                    if (u01 < p) {
                        ++skipped;
                        continue;
                    }
                }
                qs.push_back(&train[i]);
                qi.push_back(int(i));
            }
            std::vector<Rollout> ro = rollout(dec, tok, qs, qi, g.group, g.max_new, float(g.temperature),
                                              hash3(g.seed, uint64_t(u), uint64_t(chunks)), s, tm);
            ++chunks;
            questions += int(qs.size());
            for (size_t q = 0; q < qs.size(); ++q) {
                double mn = 1e9, mx = -1e9;
                bool all_ok = true;
                for (int k = 0; k < g.group; ++k) {
                    Rollout& r = ro[q * size_t(g.group) + size_t(k)];
                    const auto pred = calcx::answer(r.text);
                    r.ok = calcx::correct(pred, train[size_t(r.q)].gold);
                    r.reward = calcx::reward(pred, r.text, r.calls, train[size_t(r.q)], g.partial);
                    if (g.max_repeat > 0) {  // degenerate: a long token run, or an abandoned / unclosed call
                        if (longest_run(r) > g.max_repeat ||
                            count_of(r.text, "<call>") != count_of(r.text, "</call>")) {
                            r.degen = true;
                            r.reward = 0;
                        }
                    }
                    mn = std::min(mn, r.reward);
                    mx = std::max(mx, r.reward);
                    all_ok &= r.ok;
                }
                if (g.novelty > 0) {  // correct solutions: + novelty / (how many of the group found the same one)
                    std::map<std::string, int> cnt;
                    std::vector<std::string> sigs(size_t(g.group));
                    for (int k = 0; k < g.group; ++k) {
                        Rollout& r = ro[q * size_t(g.group) + size_t(k)];
                        if (r.ok && !r.degen) ++cnt[sigs[size_t(k)] = solution_sig(r.text)];
                    }
                    mn = 1e9, mx = -1e9;
                    for (int k = 0; k < g.group; ++k) {
                        Rollout& r = ro[q * size_t(g.group) + size_t(k)];
                        if (r.ok && !r.degen) r.reward += g.novelty / cnt[sigs[size_t(k)]];
                        mn = std::min(mn, r.reward);
                        mx = std::max(mx, r.reward);
                    }
                }
                const bool mix = mx - mn > 1e-6;
                zv[size_t(qi[q])] = mix ? 0 : zv[size_t(qi[q])] + 1;
                mixed += mix;
                solved += all_ok;
                group_mixed.push_back(mix);
            }
            all.insert(all.end(), std::make_move_iterator(ro.begin()), std::make_move_iterator(ro.end()));
            if (g.target_mixed <= 0 || mixed >= g.target_mixed) break;
        }
        // advantages; the update's rows: packed sequences of the mixed groups' rollouts
        const auto ttr = std::chrono::steady_clock::now();
        std::vector<int> hid, hy;
        std::vector<uint8_t> hres;
        std::vector<float> hw, hlp;
        long own_tokens = 0;
        int rows = 0;
        size_t fill = 0;
        auto new_row = [&]() {
            hid.resize(hid.size() + size_t(T), 0);
            hy.resize(hy.size() + size_t(T), -1);
            hres.resize(hres.size() + size_t(T), 1);
            hw.resize(hw.size() + size_t(T), 0.f);
            hlp.resize(hlp.size() + size_t(T), 0.f);
            ++rows;
            fill = 0;
        };
        double rsum = 0, gen_tok = 0, calls = 0, ent = 0, okc = 0, answered = 0;
        int ent_n = 0;
        int degen = 0, dupes = 0;
        for (size_t q = 0; q < group_mixed.size(); ++q) {
            std::set<std::string> seen;  // the group's positive solutions so far (unique)
            double mean = 0, sq = 0;
            for (int k = 0; k < g.group; ++k) mean += all[q * size_t(g.group) + size_t(k)].reward;
            mean /= g.group;
            for (int k = 0; k < g.group; ++k) {
                const double d = all[q * size_t(g.group) + size_t(k)].reward - mean;
                sq += d * d;
            }
            const double sd = std::sqrt(sq / g.group);
            for (int k = 0; k < g.group; ++k) {
                Rollout& r = all[q * size_t(g.group) + size_t(k)];
                rsum += r.reward;
                okc += r.ok;
                answered += bool(calcx::answer(r.text));
                gen_tok += double(r.ids.size()) - r.n_prompt;
                calls += r.calls;
                if (r.ent_n) {
                    ent += r.ent_sum / r.ent_n;
                    ++ent_n;
                }
                degen += r.degen;
                const bool raft = g.mode == "raft";
                if (raft ? (!r.ok || r.degen) : !group_mixed[q]) continue;
                const double adv = raft ? 1.0 : g.no_std ? r.reward - mean : (r.reward - mean) / std::max(sd, g.std_floor);
                if (g.unique && adv > 0) {
                    if (!seen.insert(solution_sig(r.text)).second) {
                        ++dupes;
                        continue;
                    }
                }
                const size_t L = r.ids.size();
                if (L > size_t(T)) continue;  // (cannot happen with prompts + max_new under T)
                if (rows == 0 || fill + L > size_t(T)) new_row();
                const size_t base = size_t(rows - 1) * T + fill;
                for (size_t i = 0; i < L; ++i) {
                    hid[base + i] = r.ids[i];
                    hres[base + i] = r.ids[i] == 0;
                }
                for (size_t i = 0; i + 1 < L; ++i) {
                    const long gi = long(i + 1) - r.n_prompt;  // the generated token that ids[i + 1] is
                    if (gi < 0 || !r.own[size_t(gi)]) continue;
                    hy[base + i] = r.ids[i + 1];
                    hw[base + i] = float(adv);
                    hlp[base + i] = r.lp[size_t(gi)];
                    ++own_tokens;
                }
                fill += L;
            }
        }
        const int nro = int(all.size());
        float loss = 0, gnorm = 0, lr = 0;
        if (own_tokens > 0) {
            while (rows % c.B) new_row();  // whole micro-batches (padding rows: no targets)
            const float inv = 1.f / float(own_tokens);
            LBC_CUDA(cudaMemcpyAsync(d_inv_n, &inv, 4, cudaMemcpyHostToDevice, s));
            float lsum = 0;
            for (int ep = 0; ep < std::max(1, g.ppo_epochs); ++ep) {
            lsum = 0;
            for (int b0 = 0; b0 < rows; b0 += c.B) {
                const size_t o = size_t(b0) * T;
                LBC_CUDA(cudaMemcpyAsync(d_ids, hid.data() + o, ntok * 4, cudaMemcpyHostToDevice, s));
                LBC_CUDA(cudaMemcpyAsync(d_y, hy.data() + o, ntok * 4, cudaMemcpyHostToDevice, s));
                LBC_CUDA(cudaMemcpyAsync(d_reset, hres.data() + o, ntok, cudaMemcpyHostToDevice, s));
                LBC_CUDA(cudaMemcpyAsync(d_w, hw.data() + o, ntok * 4, cudaMemcpyHostToDevice, s));
                LBC_CUDA(cudaMemcpyAsync(d_lp, hlp.data() + o, ntok * 4, cudaMemcpyHostToDevice, s));
                Batch b{d_ids, d_y, d_reset, d_inv_n};
                b.w = d_w;
                b.lp_old = g.tis_cap > 0 ? d_lp : nullptr;  // tis_cap <= 0: plain on-policy weights (no TIS)
                b.tis_cap = float(g.tis_cap);
                if (g.ppo_epochs > 1) {
                    b.clip_low = float(g.clip_low);
                    b.clip_high = float(g.clip_high);
                }
                model.forward_backward(b, s, b0 > 0);
                float r0;
                LBC_CUDA(cudaMemcpyAsync(&r0, model.metrics_device(), 4, cudaMemcpyDeviceToHost, s));
                LBC_CUDA(cudaStreamSynchronize(s));  // the host buffers' next slice is copied next
                lsum += r0;
            }
            for (int k = 0; rload && k < g.replay_rows / c.B; ++k) {  // pretraining rows, weight replay_coef overall
                HostBatch* hb = rload->next();
                std::fill(hw_rep.begin(), hw_rep.end(),
                          float(g.replay_coef * double(own_tokens) / std::max(1, hb->n_targets)));
                LBC_CUDA(cudaMemcpyAsync(d_ids, hb->ids, ntok * 4, cudaMemcpyHostToDevice, s));
                LBC_CUDA(cudaMemcpyAsync(d_y, hb->y, ntok * 4, cudaMemcpyHostToDevice, s));
                LBC_CUDA(cudaMemcpyAsync(d_reset, hb->reset, ntok, cudaMemcpyHostToDevice, s));
                LBC_CUDA(cudaMemcpyAsync(d_w, hw_rep.data(), ntok * 4, cudaMemcpyHostToDevice, s));
                rload->release(hb, s);
                Batch b{d_ids, d_y, d_reset, d_inv_n};
                b.w = d_w;
                model.forward_backward(b, s, true);
                LBC_CUDA(cudaStreamSynchronize(s));
            }
            model.optimizer_step(s);
            }
            std::vector<float> rec(size_t(model.metrics_count()));
            LBC_CUDA(cudaMemcpyAsync(rec.data(), model.metrics_device(), rec.size() * 4, cudaMemcpyDeviceToHost, s));
            LBC_CUDA(cudaStreamSynchronize(s));
            const StepMetrics sm = model.decode_metrics(rec.data());
            loss = lsum;
            gnorm = sm.grad_norm;
            lr = sm.lr;
        }
        const double t_train = secs_since(ttr), t_total = secs_since(t0);
        const int ng = int(group_mixed.size());
        csv << u + 1 << "," << rsum / nro << "," << okc / nro << "," << double(mixed) / ng << ","
            << double(solved) / ng << "," << questions << "," << skipped << "," << chunks << "," << nro << "," << rows
            << "," << own_tokens << "," << gen_tok / nro << "," << calls / nro << "," << answered / nro << ","
            << (ent_n ? ent / ent_n : 0) << "," << loss << "," << gnorm << "," << lr << "," << tm.prefill << ","
            << tm.decode << "," << tm.host << "," << t_train << "," << t_total << "\n";
        csv.flush();
        std::printf("update %d: reward %.3f acc %.3f mixed %.2f solved %.2f skipped %d rollouts %d len %.0f calls %.2f "
                    "ent %.3f degen %d dupes %d tokens %ld rows %d |g| %.3f lr %.1e (prefill %.1f decode %.1f host %.1f train %.1f = "
                    "%.1f s)\n",
                    u + 1, rsum / nro, okc / nro, double(mixed) / ng, double(solved) / ng, skipped, nro,
                    gen_tok / nro, calls / nro, ent_n ? ent / ent_n : 0, degen, dupes, own_tokens, rows, gnorm, lr, tm.prefill,
                    tm.decode, tm.host, t_train, t_total);
        std::fflush(stdout);
        const bool do_eval = (u + 1) % g.eval_every == 0 || u + 1 == g.updates;
        double acc = -1;
        if (do_eval) acc = evaluate(u + 1);
        const bool improved = do_eval && acc > best_acc;
        if (improved) best_acc = acc;
        if (g.save && (do_eval || (u + 1) % g.checkpoint_every == 0 || u + 1 == g.updates)) {
            save_checkpoint(latest, model, config_json, u + 1, best_acc, s);
            if (improved) {
                fs::create_directories(run + "/checkpoints");
                fs::copy_file(latest, run + "/checkpoints/best.lbc", fs::copy_options::overwrite_existing);
            }
        }
    }
    return 0;
}

}  // namespace lbc

