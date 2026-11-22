#include "checkpoint.hpp"

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <vector>

#include "lbc/check.hpp"

namespace lbc {

namespace {

constexpr char MAGIC[8] = {'L', 'B', 'C', 'K', 'P', 'T', '0', '1'};
constexpr size_t ALIGN = 256;

json::Value manifest(const Model& model) {
    json::Value t = json::Value::array();
    for (const auto& x : model.tensors()) {
        json::Value e = json::Value::array();
        e.push(x.name);
        e.push(json::Value(static_cast<unsigned long long>(x.off)));
        e.push(x.rows);
        e.push(x.cols);
        t.push(std::move(e));
    }
    return t;
}

struct Header {
    json::Value h;
    size_t data_offset = 0;
};

Header read_header(const std::string& path, std::ifstream& f) {
    char magic[8];
    uint64_t len = 0;
    f.read(magic, 8);
    f.read(reinterpret_cast<char*>(&len), 8);
    if (!f || std::memcmp(magic, MAGIC, 8) != 0) { throw std::runtime_error(path + " is not an lbc checkpoint"); }
    std::string text(len, '\0');
    f.read(text.data(), std::streamsize(len));
    if (!f) { throw std::runtime_error(path + ": truncated header"); }
    return {json::parse(text), (16 + len + ALIGN - 1) / ALIGN * ALIGN};
}

}  // namespace

void save_checkpoint(const std::string& path,
                     Model& model,
                     const json::Value& config,
                     int epoch,
                     double best_val_loss,
                     cudaStream_t s) {
    const size_t n = model.num_params();
    int step = 0;
    std::vector<float> buf(3 * n);
    LBC_CUDA(cudaStreamSynchronize(s));
    LBC_CUDA(cudaMemcpy(buf.data(), model.params(), n * 4, cudaMemcpyDeviceToHost));
    LBC_CUDA(cudaMemcpy(buf.data() + n, model.adam_m(), n * 4, cudaMemcpyDeviceToHost));
    LBC_CUDA(cudaMemcpy(buf.data() + 2 * n, model.adam_v(), n * 4, cudaMemcpyDeviceToHost));
    LBC_CUDA(cudaMemcpy(&step, model.step_device(), sizeof(int), cudaMemcpyDeviceToHost));

    json::Value h = json::Value::object();
    h["format"] = 1;
    h["epoch"] = epoch;
    h["step"] = step;
    h["best_val_loss"] = best_val_loss >= 0 ? json::Value(best_val_loss) : json::Value();
    h["num_params"] = json::Value(static_cast<unsigned long long>(n));
    h["tensors"] = manifest(model);
    h["config"] = config;
    const std::string text = json::dump(h, 0);
    const uint64_t len = text.size();
    const size_t data_offset = (16 + len + ALIGN - 1) / ALIGN * ALIGN;

    std::filesystem::create_directories(std::filesystem::path(path).parent_path());
    const std::string tmp = path + ".tmp";
    {
        std::ofstream f(tmp, std::ios::binary);
        f.write(MAGIC, 8);
        f.write(reinterpret_cast<const char*>(&len), 8);
        f.write(text.data(), std::streamsize(len));
        const std::vector<char> pad(data_offset - 16 - len, 0);
        f.write(pad.data(), std::streamsize(pad.size()));
        f.write(reinterpret_cast<const char*>(buf.data()), std::streamsize(buf.size() * 4));
        f.flush();
        if (!f) { throw std::runtime_error("cannot write " + tmp); }
    }
    std::filesystem::rename(tmp, path);
}

CheckpointInfo read_checkpoint_info(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { throw std::runtime_error("cannot read " + path); }
    const Header hd = read_header(path, f);
    CheckpointInfo info;
    if (const json::Value* v = hd.h.find("epoch")) { info.epoch = int(v->num); }
    if (const json::Value* v = hd.h.find("step")) { info.step = static_cast<long long>(v->num); }
    if (const json::Value* v = hd.h.find("best_val_loss"); v && !v->is_null()) { info.best_val_loss = v->num; }
    if (const json::Value* v = hd.h.find("config")) { info.config = *v; }
    return info;
}

void load_checkpoint(const std::string& path, Model& model, cudaStream_t s) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { throw std::runtime_error("cannot read " + path); }
    const Header hd = read_header(path, f);
    const json::Value* tensors = hd.h.find("tensors");
    if ((tensors == nullptr) || *tensors != manifest(model)) {
        throw std::runtime_error(path + ": its tensors (names, shapes, layout) differ from this model's");
    }
    const size_t n = model.num_params();
    std::vector<float> buf(3 * n);
    f.seekg(std::streamoff(hd.data_offset));
    f.read(reinterpret_cast<char*>(buf.data()), std::streamsize(buf.size() * 4));
    if (!f) { throw std::runtime_error(path + ": truncated data"); }
    const int step = int(hd.h.find("step") ? hd.h.find("step")->num : 0);
    LBC_CUDA(cudaStreamSynchronize(s));
    LBC_CUDA(cudaMemcpy(model.params(), buf.data(), n * 4, cudaMemcpyHostToDevice));
    LBC_CUDA(cudaMemcpy(model.adam_m(), buf.data() + n, n * 4, cudaMemcpyHostToDevice));
    LBC_CUDA(cudaMemcpy(model.adam_v(), buf.data() + 2 * n, n * 4, cudaMemcpyHostToDevice));
    LBC_CUDA(cudaMemcpy(model.step_device(), &step, sizeof(int), cudaMemcpyHostToDevice));
    model.prepare_weights(s);
    LBC_CUDA(cudaStreamSynchronize(s));
}

void load_weights(const std::string& path, Model& model, cudaStream_t s) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { throw std::runtime_error("cannot read " + path); }
    const Header hd = read_header(path, f);
    const json::Value* tensors = hd.h.find("tensors");
    if ((tensors == nullptr) || *tensors != manifest(model)) {
        throw std::runtime_error(path + ": its tensors (names, shapes, layout) differ from this model's");
    }
    std::vector<float> buf(model.num_params());
    f.seekg(std::streamoff(hd.data_offset));
    f.read(reinterpret_cast<char*>(buf.data()), std::streamsize(buf.size() * 4));
    if (!f) { throw std::runtime_error(path + ": truncated data"); }
    LBC_CUDA(cudaStreamSynchronize(s));
    LBC_CUDA(cudaMemcpy(model.params(), buf.data(), buf.size() * 4, cudaMemcpyHostToDevice));
    model.prepare_weights(s);
    LBC_CUDA(cudaStreamSynchronize(s));
}

}  // namespace lbc
