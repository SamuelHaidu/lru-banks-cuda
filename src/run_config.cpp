#include "run_config.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <stdexcept>

namespace lbc {

namespace {

[[noreturn]] void bad(const std::string& key, const std::string& what) {
    throw std::runtime_error("config: " + key + " " + what);
}

json::Value to_value(int x) { return json::Value(x); }
json::Value to_value(double x) { return json::Value(x); }
json::Value to_value(bool x) { return json::Value(x); }
json::Value to_value(const std::string& x) { return json::Value(x); }
template <class T>
json::Value to_value(const std::vector<T>& xs) {
    json::Value a = json::Value::array();
    for (const T& x : xs) a.push(to_value(x));
    return a;
}

void from_value(const json::Value& v, int& x, const std::string& key) {
    if (v.type != json::Value::Number || v.num != std::floor(v.num) || std::fabs(v.num) > 2e9)
        bad(key, "must be an integer");
    x = int(v.num);
}
void from_value(const json::Value& v, double& x, const std::string& key) {
    if (v.is_null()) {  // e.g. global_clipnorm: null = off
        x = 0.0;
        return;
    }
    if (v.type != json::Value::Number) bad(key, "must be a number");
    x = v.num;
}
void from_value(const json::Value& v, bool& x, const std::string& key) {
    if (v.type != json::Value::Bool) bad(key, "must be true or false");
    x = v.b;
}
void from_value(const json::Value& v, std::string& x, const std::string& key) {
    if (v.type != json::Value::String) bad(key, "must be a string");
    x = v.str;
}
template <class T>
void from_value(const json::Value& v, std::vector<T>& xs, const std::string& key) {
    if (v.type != json::Value::Array) bad(key, "must be a list");
    xs.clear();
    for (size_t i = 0; i < v.arr.size(); ++i) {
        T x{};
        from_value(v.arr[i], x, key + "[" + std::to_string(i) + "]");
        xs.push_back(x);
    }
}

struct Writer {
    json::Value root = json::Value::object();
    json::Value* cur = nullptr;
    void begin(const char* section) {
        root[section] = json::Value::object();
        cur = &root[section];
    }
    template <class T>
    void operator()(const char* key, const T& x) {
        (*cur)[key] = to_value(x);
    }
    void end() {}
};

struct Reader {
    explicit Reader(const json::Value& r) : root(r) {}
    const json::Value& root;
    const json::Value* cur = nullptr;
    std::string section;
    std::vector<std::string> seen, sections;
    void begin(const char* s) {
        section = s;
        sections.push_back(s);
        seen.clear();
        cur = root.find(s);
        if (cur && !cur->is_object()) bad(section, "must be an object");
    }
    template <class T>
    void operator()(const char* key, T& x) {
        seen.push_back(key);
        if (!cur) return;
        if (const json::Value* v = cur->find(key)) from_value(*v, x, section + "." + key);
    }
    void end() {
        if (!cur) return;
        for (const auto& [k, v] : cur->obj)
            if (std::find(seen.begin(), seen.end(), k) == seen.end()) {
                std::string allowed;
                for (const auto& s : seen) allowed += (allowed.empty() ? "" : ", ") + s;
                bad(section + "." + k, "is not a known key (known: " + allowed + ")");
            }
    }
};

// one list of every key, for writing and reading
template <class V, class RC>
void visit(V& v, RC& c) {
    v.begin("arch_params");
    v("vocabulary_size", c.arch.vocabulary_size);
    v("dim", c.arch.dim);
    v("embed_dim", c.arch.embed_dim);
    v("factorized", c.arch.factorized);
    v("tied_embeddings", c.arch.tied_embeddings);
    v("num_layers", c.arch.num_layers);
    v("hops", c.arch.hops);
    v("lru_dim", c.arch.lru_dim);
    v("num_heads", c.arch.num_heads);
    v("ffn_mult", c.arch.ffn_mult);
    v("lru_half_life", c.arch.lru_half_life);
    v("slot_half_life", c.arch.slot_half_life);
    v("tau", c.arch.tau);
    v("gate_bias_write", c.arch.gate_bias_write);
    v("gate_bias_read", c.arch.gate_bias_read);
    v("gate_budget", c.arch.gate_budget);
    v("gate_budget_target", c.arch.gate_budget_target);
    v("window_size", c.arch.window_size);
    v("bank_slots", c.arch.bank_slots);
    v("head_dim", c.arch.head_dim);
    v("topk", c.arch.topk);
    v.end();
    v.begin("data");
    v("dataset", c.data.dataset);
    v("batch_size", c.data.batch_size);
    v("epoch_passes", c.data.epoch_passes);
    v("seed", c.data.seed);
    v.end();
    v.begin("training");
    v("learning_rate", c.training.learning_rate);
    v("warmup_steps", c.training.warmup_steps);
    v("lr_decay_start", c.training.lr_decay_start);
    v("lr_decay_steps", c.training.lr_decay_steps);
    v("lr_min_ratio", c.training.lr_min_ratio);
    v("weight_decay", c.training.weight_decay);
    v("beta_1", c.training.beta_1);
    v("beta_2", c.training.beta_2);
    v("epsilon", c.training.epsilon);
    v("global_clipnorm", c.training.global_clipnorm);
    v("epochs", c.training.epochs);
    v("validation_steps", c.training.validation_steps);
    v("checkpoint_every", c.training.checkpoint_every);
    v.end();
    v.begin("runtime");
    v("bank_chunk", c.runtime.bank_chunk);
    v("ce_chunk", c.runtime.ce_chunk);
    v("cuda_graph", c.runtime.cuda_graph);
    v.end();
}

void check(bool ok, const std::string& key, const std::string& what) {
    if (!ok) bad(key, what);
}

}  // namespace

json::Value RunConfig::to_json() const {
    Writer w;
    visit(w, *this);
    return w.root;
}

RunConfig RunConfig::from_json(const json::Value& v) {
    if (!v.is_object()) throw std::runtime_error("config: the top level must be an object");
    RunConfig rc;
    Reader r(v);
    visit(r, rc);
    for (const auto& [k, x] : v.obj)
        if (std::find(r.sections.begin(), r.sections.end(), k) == r.sections.end())
            bad(k, "is not a known section (arch_params, data, training, runtime)");
    rc.validate();
    return rc;
}

int RunConfig::ffn_dim() const { return int(std::lround(arch.ffn_mult * arch.dim)); }

void RunConfig::validate() const {
    const ArchCfg& a = arch;
    check(a.vocabulary_size >= 8 && a.vocabulary_size <= 65536 && a.vocabulary_size % 8 == 0,
          "arch_params.vocabulary_size", "must be a multiple of 8 in [8, 65536] (uint16 tokens)");
    check(a.dim >= 32 && a.dim <= 1024 && a.dim % 32 == 0, "arch_params.dim",
          "must be a multiple of 32 in [32, 1024] (the norm backward)");
    if (a.factorized)
        check(a.embed_dim >= 8 && a.embed_dim % 8 == 0, "arch_params.embed_dim", "must be a positive multiple of 8");
    check(a.num_layers >= 1, "arch_params.num_layers", "must be at least 1");
    check(a.hops.empty() || int(a.hops.size()) == a.num_layers, "arch_params.hops",
          "must have one entry per layer (num_layers " + std::to_string(a.num_layers) + "), or be [] (2 everywhere)");
    for (int h : a.hops) check(h == 1 || h == 2, "arch_params.hops", "entries must be 1 or 2");
    check(a.lru_dim >= 4 && a.lru_dim % 4 == 0, "arch_params.lru_dim", "must be a positive multiple of 4");
    check(a.num_heads >= 1, "arch_params.num_heads", "must be at least 1");
    check(ffn_dim() >= 8 && ffn_dim() % 8 == 0, "arch_params.ffn_mult", "x dim must be a positive multiple of 8");
    for (const auto* hl : {&a.lru_half_life, &a.slot_half_life})
        check(hl->size() == 2 && (*hl)[0] > 0 && (*hl)[0] <= (*hl)[1],
              hl == &a.lru_half_life ? "arch_params.lru_half_life" : "arch_params.slot_half_life",
              "must be [min, max] with 0 < min <= max");
    check(a.tau > 0, "arch_params.tau", "must be positive");
    check(a.gate_budget >= 0, "arch_params.gate_budget", "must be >= 0");
    check(a.window_size >= 8 && a.window_size % 8 == 0, "arch_params.window_size", "must be a positive multiple of 8");
    check(a.head_dim == 32 || a.head_dim == 64, "arch_params.head_dim", "must be 32 or 64 (the kernels' sizes)");
    check(a.bank_slots.empty() || int(a.bank_slots.size()) == a.num_layers, "arch_params.bank_slots",
          "must have one entry per layer (num_layers " + std::to_string(a.num_layers) + "), or be [] (128 everywhere)");
    for (int n : a.bank_slots)
        check(n >= 2 && n <= SLOTS, "arch_params.bank_slots", "entries must be in [2, " + std::to_string(SLOTS) + "]");
    check(a.topk == 2, "arch_params.topk", "is fixed at 2 by the kernels");
    check(data.batch_size >= 1, "data.batch_size", "must be at least 1");
    check(data.epoch_passes > 0, "data.epoch_passes", "must be positive");
    const TrainCfg& t = training;
    check(t.learning_rate > 0, "training.learning_rate", "must be positive");
    check(t.warmup_steps >= 0 && t.lr_decay_start >= 0 && t.lr_decay_steps >= 0, "training.warmup_steps / lr_decay_*",
          "must be >= 0");
    check(t.lr_min_ratio >= 0 && t.lr_min_ratio <= 1, "training.lr_min_ratio", "must be in [0, 1]");
    check(t.weight_decay >= 0, "training.weight_decay", "must be >= 0");
    check(t.beta_1 >= 0 && t.beta_1 < 1 && t.beta_2 >= 0 && t.beta_2 < 1, "training.beta_1 / beta_2",
          "must be in [0, 1)");
    check(t.epsilon > 0, "training.epsilon", "must be positive");
    check(t.global_clipnorm >= 0, "training.global_clipnorm", "must be >= 0 (0 or null: off)");
    check(t.epochs >= 1, "training.epochs", "must be at least 1");
    check(t.validation_steps >= 0, "training.validation_steps", "must be >= 0");
    check(t.checkpoint_every >= 0, "training.checkpoint_every", "must be >= 0");
    check(runtime.bank_chunk >= 8 && runtime.bank_chunk % 8 == 0, "runtime.bank_chunk",
          "must be a positive multiple of 8");
    check(runtime.ce_chunk >= 1, "runtime.ce_chunk", "must be positive");
}

Config RunConfig::model_config() const {
    Config c;
    c.V = arch.vocabulary_size;
    c.D = arch.dim;
    c.factorized = arch.factorized;
    c.E0 = arch.factorized ? arch.embed_dim : arch.dim;
    c.tied = arch.tied_embeddings;
    c.L = arch.num_layers;
    c.hops = arch.hops.empty() ? std::vector<int>(size_t(arch.num_layers), 2) : arch.hops;
    c.slots = arch.bank_slots;
    c.dh = arch.head_dim;
    c.N = arch.lru_dim;
    c.H = arch.num_heads;
    c.F = ffn_dim();
    c.B = data.batch_size;
    c.T = arch.window_size;
    c.bank_chunk = runtime.bank_chunk;
    c.ce_chunk = runtime.ce_chunk;
    c.lr = float(training.learning_rate);
    c.warmup = training.warmup_steps;
    c.decay_start = training.lr_decay_start;
    c.decay_steps = training.lr_decay_steps;
    c.min_ratio = float(training.lr_min_ratio);
    c.wd = float(training.weight_decay);
    c.beta1 = float(training.beta_1);
    c.beta2 = float(training.beta_2);
    c.eps = float(training.epsilon);
    c.clip = float(training.global_clipnorm);
    c.tau = float(arch.tau);
    c.lru_hl_min = float(arch.lru_half_life[0]);
    c.lru_hl_max = float(arch.lru_half_life[1]);
    c.slot_hl_min = float(arch.slot_half_life[0]);
    c.slot_hl_max = float(arch.slot_half_life[1]);
    c.gate_bias_w = float(arch.gate_bias_write);
    c.gate_bias_r = float(arch.gate_bias_read);
    c.budget = float(arch.gate_budget);
    c.budget_target = float(arch.gate_budget_target);
    return c;
}

RunConfig load_run_config(const std::string& run_dir) {
    const std::string path = run_dir + "/config.json";
    if (!std::filesystem::exists(path))
        throw std::runtime_error("no config at " + path + ": create the run first (lbc create " + run_dir +
                                 " --dataset DIR)");
    try {
        return RunConfig::from_json(json::read_file(path));
    } catch (const std::exception& e) {
        throw std::runtime_error(path + ": " + e.what());
    }
}

void save_run_config(const std::string& run_dir, const RunConfig& rc) {
    rc.validate();
    std::filesystem::create_directories(run_dir);
    json::write_file(run_dir + "/config.json", rc.to_json());
}

RunConfig with_key(const RunConfig& rc, const std::string& dotted, const std::string& raw) {
    json::Value v = rc.to_json();
    if (!json::get(v, dotted)) throw std::runtime_error("config: no key " + dotted);
    json::Value x;
    try {
        x = json::parse(raw);
    } catch (const std::exception&) {
        x = json::Value(raw);  // not JSON: a plain string (e.g. a dataset path)
    }
    json::set(v, dotted, std::move(x));
    return RunConfig::from_json(v);
}

DatasetMeta DatasetMeta::load(const std::string& dir) {
    const json::Value m = json::read_file(dir + "/metadata.json");
    DatasetMeta d;
    d.dir = dir;
    const json::Value* name = m.find("name");
    d.name = name && name->type == json::Value::String ? name->str : std::filesystem::path(dir).filename().string();
    auto integer = [&](const char* key) {
        const json::Value* v = m.find(key);
        if (!v || v->type != json::Value::Number) throw std::runtime_error(dir + "/metadata.json: no " + key);
        return int(v->num);
    };
    d.vocabulary_size = integer("vocabulary_size");
    d.doc_token_id = integer("doc_token_id");
    double avg = 0.0;
    if (const json::Value* a = m.find("avg_chars_per_token"); a && a->type == json::Value::Number) avg = a->num;
    auto split = [&](const char* s, std::string& file, double& cpt) {
        const json::Value* sp = m.find(s);
        const json::Value* f = sp ? sp->find("file") : nullptr;
        file = dir + "/" + (f && f->type == json::Value::String ? f->str : std::string(s) + ".npy");
        const json::Value* c = sp ? sp->find("chars_per_token") : nullptr;
        cpt = c && c->type == json::Value::Number ? c->num : avg;
    };
    split("train", d.train_file, d.train_chars_per_token);
    split("validation", d.validation_file, d.validation_chars_per_token);
    return d;
}

}  // namespace lbc
