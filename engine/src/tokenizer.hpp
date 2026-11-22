#pragma once
// The byte-level BPE tokenizer of a Hugging Face `tokenizer.json` (GPT-2 style: no normalizer, the ByteLevel
// pre-tokenizer with its regex, BPE merges, added tokens matched first), encode and decode. The pre-tokenizer regex
//   's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
// is matched by hand: exact for ASCII; outside ASCII the letter / number / space classes come from Unicode ranges
// (tokenizer.cpp: is_letter, is_number, is_space), which cover the scripts of our corpora.

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace lbc {

class Tokenizer {
  public:
    explicit Tokenizer(const std::string& path);
    std::vector<int> encode(const std::string& text) const;  // added tokens (e.g. <call>) are recognized in the text
    std::string decode(const std::vector<int>& ids) const;   // special tokens included
    std::string decode(const int* ids, size_t n) const;
    int id(const std::string& token) const;                  // -1 when unknown (added tokens and vocab entries)
    int vocab_size() const { return int(id_to_token_.size()); }

  private:
    void bpe(const std::string& piece, std::vector<int>& out) const;
    std::unordered_map<std::string, int> vocab_, ranks_, added_;
    std::vector<std::string> id_to_token_;
    std::vector<bool> is_added_;
    std::vector<std::string> added_list_;  // longest first
    std::string byte_to_char_[256];
    std::unordered_map<uint32_t, uint8_t> char_to_byte_;
    mutable std::unordered_map<std::string, std::vector<int>> cache_;
};

}  // namespace lbc
