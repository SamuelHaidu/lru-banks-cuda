#include "tokenizer.hpp"

#include <algorithm>
#include <climits>
#include <stdexcept>

#include "json.hpp"

namespace lbc {
namespace {

void put_utf8(std::string& s, uint32_t c) {
    if (c < 0x80) {
        s += char(c);
    } else if (c < 0x800) {
        s += char(0xC0 | (c >> 6));
        s += char(0x80 | (c & 0x3F));
    } else if (c < 0x10000) {
        s += char(0xE0 | (c >> 12));
        s += char(0x80 | ((c >> 6) & 0x3F));
        s += char(0x80 | (c & 0x3F));
    } else {
        s += char(0xF0 | (c >> 18));
        s += char(0x80 | ((c >> 12) & 0x3F));
        s += char(0x80 | ((c >> 6) & 0x3F));
        s += char(0x80 | (c & 0x3F));
    }
}

// next code point at s[i] (advances i); invalid bytes come back as themselves (0x80..0xFF)
uint32_t next_cp(const std::string& s, size_t& i) {
    const auto b = [&](size_t k) { return uint8_t(s[k]); };
    const uint8_t c = b(i);
    int n = 0;  // the sequence's length from its lead byte (0: not a lead byte)
    if (c < 0x80) {
        n = 1;
    } else if ((c >> 5) == 6) {
        n = 2;
    } else if ((c >> 4) == 14) {
        n = 3;
    } else if ((c >> 3) == 30) {
        n = 4;
    }
    if (n == 0 || i + size_t(n) > s.size()) {
        ++i;
        return c;
    }
    static constexpr uint8_t lead_mask[5] = {0, 0xFF, 0x1F, 0x0F, 0x07};  // the lead byte's payload bits
    uint32_t cp = c & lead_mask[n];
    for (int k = 1; k < n; ++k) {
        if ((b(i + size_t(k)) & 0xC0) != 0x80) {
            ++i;
            return c;
        }
        cp = (cp << 6) | (b(i + size_t(k)) & 0x3F);
    }
    i += size_t(n);
    return cp;
}

bool is_space(uint32_t c) {
    return (c >= 9 && c <= 13) || c == 32 || c == 0x85 || c == 0xA0 || c == 0x1680 || (c >= 0x2000 && c <= 0x200A)
           || c == 0x2028 || c == 0x2029 || c == 0x202F || c == 0x205F || c == 0x3000;
}

bool is_number(uint32_t c) {
    if (c < 0x80) { return c >= '0' && c <= '9'; }
    return c == 0xB2 || c == 0xB3 || c == 0xB9 || (c >= 0xBC && c <= 0xBE) || (c >= 0x660 && c <= 0x669)
           || (c >= 0x6F0 && c <= 0x6F9) || (c >= 0x966 && c <= 0x96F)
           || (c >= 0x2070 && c <= 0x2079 && c != 0x2071 && c != 0x2072 && c != 0x2073) || (c >= 0x2080 && c <= 0x2089)
           || (c >= 0x2150 && c <= 0x2189) || (c >= 0x2460 && c <= 0x249B) || (c >= 0x24EA && c <= 0x24FF)
           || (c >= 0x2776 && c <= 0x2793) || c == 0x3007 || (c >= 0x3021 && c <= 0x3029)
           || (c >= 0xFF10 && c <= 0xFF19);
}

bool is_letter(uint32_t c) {
    if (c < 0x80) { return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z'); }
    if (is_space(c) || is_number(c)) { return false; }
    if (c < 0xC0) { return c == 0xAA || c == 0xB5 || c == 0xBA; }
    if (c == 0xD7 || c == 0xF7) { return false; }
    if (c >= 0x2B0 && c <= 0x2FF) {
        return c <= 0x2C1 || (c >= 0x2C6 && c <= 0x2D1) || (c >= 0x2E0 && c <= 0x2E4) || c == 0x2EC
               || c == 0x2EE;  // modifier letters vs modifier symbols
    }
    if (c >= 0x300 && c <= 0x36F) {
        return false;  // combining marks
    }
    if (c == 0x37E || c == 0x387 || c == 0x375 || c == 0x384 || c == 0x385) { return false; }
    if (c >= 0x483 && c <= 0x489) { return false; }
    if (c >= 0x591 && c <= 0x5C7) {
        return false;  // Hebrew points
    }
    if (c >= 0x5F3 && c <= 0x5F4) { return false; }
    if (c >= 0x600 && c <= 0x61F) { return false; }
    if (c >= 0x64B && c <= 0x65F) { return false; }
    if (c == 0x66A || c == 0x66B || c == 0x66C || c == 0x66D || c == 0x6D4) { return false; }
    if (c >= 0x900 && c <= 0x903) { return false; }
    if (c >= 0x93A && c <= 0x94F && c != 0x93D) { return false; }
    if (c >= 0x2000 && c <= 0x2BFF) {
        return (c >= 0x2071 && c <= 0x207F && c != 0x207A && c != 0x207B && c != 0x207C && c != 0x207D && c != 0x207E)
               || (c >= 0x2090 && c <= 0x209C) || c == 0x2102 || c == 0x2107 || (c >= 0x210A && c <= 0x2113)
               || c == 0x2115 || (c >= 0x2119 && c <= 0x211D) || c == 0x2124 || c == 0x2126 || c == 0x2128
               || (c >= 0x212A && c <= 0x212D) || (c >= 0x212F && c <= 0x2139) || (c >= 0x213C && c <= 0x213F)
               || (c >= 0x2145 && c <= 0x2149) || c == 0x214E;
    }
    if (c >= 0x2E00 && c <= 0x2E7F) {
        return false;  // supplemental punctuation
    }
    if (c >= 0x3000 && c <= 0x303F) {
        return c == 0x3005 || c == 0x3006 || (c >= 0x3031 && c <= 0x3035) || c == 0x303B || c == 0x303C;
    }
    if (c == 0x30FB) { return false; }
    if (c >= 0xFE00 && c <= 0xFE0F) {
        return false;  // variation selectors
    }
    if (c >= 0xFE10 && c <= 0xFE6F) {
        return false;  // vertical / small forms
    }
    if (c >= 0xFF00 && c <= 0xFF65) { return (c >= 0xFF21 && c <= 0xFF3A) || (c >= 0xFF41 && c <= 0xFF5A); }
    if (c >= 0xFFE0 && c <= 0xFFFF) { return false; }
    if (c >= 0x1F000 && c <= 0x1FAFF) {
        return false;  // emoji, symbols
    }
    if (c >= 0xE0000) { return false; }
    return true;
}

enum Cls { SPACE, LETTER, NUMBER, OTHER };

Cls cls(uint32_t c) {
    if (is_space(c)) { return SPACE; }
    if (is_letter(c)) { return LETTER; }
    if (is_number(c)) { return NUMBER; }
    return OTHER;
}

// the pre-tokenizer: pieces as byte ranges of `s`
void pretokenize(const std::string& s, std::vector<std::pair<size_t, size_t>>& out) {
    std::vector<uint32_t> cp;
    std::vector<size_t> at;  // byte offset of each code point, plus the end
    for (size_t i = 0; i < s.size();) {
        at.push_back(i);
        cp.push_back(next_cp(s, i));
    }
    at.push_back(s.size());
    const size_t n = cp.size();
    std::vector<uint8_t> c(n);
    for (size_t i = 0; i < n; ++i) { c[i] = uint8_t(cls(cp[i])); }
    size_t i = 0;
    while (i < n) {
        size_t j = i;
        // 's 't 're 've 'm 'll 'd
        if (cp[i] == '\'' && i + 1 < n) {
            const uint32_t a = cp[i + 1], b = i + 2 < n ? cp[i + 2] : 0;
            if (a == 's' || a == 't' || a == 'm' || a == 'd') {
                j = i + 2;
            } else if ((a == 'r' && b == 'e') || (a == 'v' && b == 'e') || (a == 'l' && b == 'l')) {
                j = i + 3;
            }
        }
        if (j == i) {
            // ` ?\p{L}+`, ` ?\p{N}+`, ` ?[^\s\p{L}\p{N}]+`: an optional literal space, then a run of one class
            const size_t k = (cp[i] == ' ' && i + 1 < n) ? i + 1 : i;
            if (c[k] != SPACE) {
                const uint8_t want = c[k];
                j = k;
                while (j < n && c[j] == want) { ++j; }
            } else {
                // `\s+(?!\S)` (a run, minus its last space when a non-space follows), else `\s+`
                j = i;
                while (j < n && c[j] == SPACE) { ++j; }
                if (j < n && j - i > 1) { --j; }
            }
        }
        out.emplace_back(at[i], at[j]);
        i = j;
    }
}

}  // namespace

Tokenizer::Tokenizer(const std::string& path) {
    const json::Value t = json::read_file(path);
    const json::Value* model = t.find("model");
    if ((model == nullptr) || (model->find("vocab") == nullptr) || (model->find("merges") == nullptr)) {
        throw std::runtime_error(path + ": no BPE model");
    }
    for (const auto& [tok, idv] : model->find("vocab")->obj) {
        const int i = int(idv.num);
        vocab_[tok] = i;
        if (size_t(i) >= id_to_token_.size()) { id_to_token_.resize(size_t(i) + 1); }
        id_to_token_[size_t(i)] = tok;
    }
    int r = 0;
    for (const auto& m : model->find("merges")->arr) {
        if (m.type == json::Value::Array) {
            ranks_[m.arr[0].str + " " + m.arr[1].str] = r++;
        } else {
            ranks_[m.str] = r++;
        }
    }
    is_added_.assign(id_to_token_.size(), false);
    if (const json::Value* add = t.find("added_tokens")) {
        for (const auto& a : add->arr) {
            const int i = int(a.find("id")->num);
            const std::string& content = a.find("content")->str;
            added_[content] = i;
            if (size_t(i) >= id_to_token_.size()) {
                id_to_token_.resize(size_t(i) + 1);
                is_added_.resize(size_t(i) + 1, false);
            }
            id_to_token_[size_t(i)] = content;
            is_added_[size_t(i)] = true;
            added_list_.push_back(content);
        }
    }
    std::ranges::sort(added_list_, [](auto& a, auto& b) { return a.size() > b.size(); });
    // GPT-2 bytes_to_unicode
    int n = 0;
    for (int b = 0; b < 256; ++b) {
        const bool keep = (b >= '!' && b <= '~') || (b >= 0xA1 && b <= 0xAC) || (b >= 0xAE && b <= 0xFF);
        const uint32_t u = keep ? uint32_t(b) : uint32_t(256 + n++);
        put_utf8(byte_to_char_[b], u);
        char_to_byte_[u] = uint8_t(b);
    }
}

int Tokenizer::id(const std::string& token) const {
    if (auto it = added_.find(token); it != added_.end()) { return it->second; }
    if (auto it = vocab_.find(token); it != vocab_.end()) { return it->second; }
    return -1;
}

void Tokenizer::bpe(const std::string& piece, std::vector<int>& out) const {
    if (auto it = cache_.find(piece); it != cache_.end()) {
        out.insert(out.end(), it->second.begin(), it->second.end());
        return;
    }
    std::vector<std::string> w;
    for (unsigned char const b : piece) { w.push_back(byte_to_char_[b]); }
    while (w.size() > 1) {
        int best = INT_MAX;
        size_t bi = 0;
        for (size_t i = 0; i + 1 < w.size(); ++i) {
            auto it = ranks_.find(w[i] + " " + w[i + 1]);
            if (it != ranks_.end() && it->second < best) {
                best = it->second;
                bi = i;
            }
        }
        if (best == INT_MAX) { break; }
        const std::string a = w[bi], b = w[bi + 1];
        std::vector<std::string> nw;
        for (size_t i = 0; i < w.size();) {
            if (i + 1 < w.size() && w[i] == a && w[i + 1] == b) {
                nw.push_back(a + b);
                i += 2;
            } else {
                nw.push_back(w[i++]);
            }
        }
        w.swap(nw);
    }
    std::vector<int> ids;
    for (const auto& x : w) {
        auto it = vocab_.find(x);
        if (it == vocab_.end()) { throw std::runtime_error("tokenizer: no vocab entry for a BPE piece"); }
        ids.push_back(it->second);
    }
    if (cache_.size() < 200000) { cache_[piece] = ids; }
    out.insert(out.end(), ids.begin(), ids.end());
}

std::vector<int> Tokenizer::encode(const std::string& text) const {
    std::vector<int> out;
    std::vector<std::pair<size_t, size_t>> pieces;
    size_t i = 0, seg = 0;
    auto flush = [&](size_t end) {
        if (end <= seg) { return; }
        const std::string part = text.substr(seg, end - seg);
        pieces.clear();
        pretokenize(part, pieces);
        for (auto [a, b] : pieces) { bpe(part.substr(a, b - a), out); }
    };
    while (i < text.size()) {
        bool hit = false;
        if (text[i] == '<') {
            for (const auto& a : added_list_) {
                if (text.compare(i, a.size(), a) == 0) {
                    flush(i);
                    out.push_back(added_.at(a));
                    i += a.size();
                    seg = i;
                    hit = true;
                    break;
                }
            }
        }
        if (!hit) { ++i; }
    }
    flush(text.size());
    return out;
}

std::string Tokenizer::decode(const int* ids, size_t n) const {
    std::string out;
    for (size_t k = 0; k < n; ++k) {
        const int i = ids[k];
        if (i < 0 || size_t(i) >= id_to_token_.size()) { continue; }
        const std::string& t = id_to_token_[size_t(i)];
        if (is_added_[size_t(i)]) {
            out += t;
            continue;
        }
        for (size_t j = 0; j < t.size();) {
            const uint32_t cp = next_cp(t, j);
            auto it = char_to_byte_.find(cp);
            if (it != char_to_byte_.end()) { out += char(it->second); }
        }
    }
    return out;
}

std::string Tokenizer::decode(const std::vector<int>& ids) const { return decode(ids.data(), ids.size()); }

}  // namespace lbc
