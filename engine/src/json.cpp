#include "json.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace lbc::json {

const Value* Value::find(const std::string& key) const {
    if (type != Object) { return nullptr; }
    for (const auto& [k, v] : obj) {
        if (k == key) { return &v; }
    }
    return nullptr;
}

Value* Value::find(const std::string& key) {
    if (type != Object) { return nullptr; }
    for (auto& [k, v] : obj) {
        if (k == key) { return &v; }
    }
    return nullptr;
}

Value& Value::operator[](const std::string& key) {
    if (type == Null) { type = Object; }
    if (type != Object) { throw std::runtime_error("json: [\"" + key + "\"] on a non-object"); }
    if (Value* v = find(key)) { return *v; }
    obj.emplace_back(key, Value());
    return obj.back().second;
}

bool Value::operator==(const Value& o) const {
    if (type != o.type) { return false; }
    switch (type) {
        case Null: return true;
        case Bool: return b == o.b;
        case Number: return num == o.num;
        case String: return str == o.str;
        case Array: return arr == o.arr;
        case Object:
            if (obj.size() != o.obj.size()) { return false; }
            for (const auto& [k, v] : obj) {
                const Value* w = o.find(k);
                if (!w || !(*w == v)) { return false; }
            }
            return true;
    }
    return false;
}

namespace {

struct Parser {
    const std::string& s;
    size_t i = 0;

    [[noreturn]] void error(const std::string& what) const {
        throw std::runtime_error("json: " + what + " at offset " + std::to_string(i));
    }

    void ws() {
        while (i < s.size() && (s[i] == ' ' || s[i] == '\t' || s[i] == '\n' || s[i] == '\r')) { ++i; }
    }

    bool literal(const char* word) {
        const size_t n = std::char_traits<char>::length(word);
        if (s.compare(i, n, word) != 0) { return false; }
        i += n;
        return true;
    }

    static void utf8(std::string& out, unsigned cp) {
        if (cp < 0x80) {
            out += char(cp);
        } else if (cp < 0x800) {
            out += char(0xC0 | (cp >> 6));
            out += char(0x80 | (cp & 0x3F));
        } else if (cp < 0x10000) {
            out += char(0xE0 | (cp >> 12));
            out += char(0x80 | ((cp >> 6) & 0x3F));
            out += char(0x80 | (cp & 0x3F));
        } else {
            out += char(0xF0 | (cp >> 18));
            out += char(0x80 | ((cp >> 12) & 0x3F));
            out += char(0x80 | ((cp >> 6) & 0x3F));
            out += char(0x80 | (cp & 0x3F));
        }
    }

    unsigned hex4() {
        if (i + 4 > s.size()) { error("short \\u escape"); }
        unsigned v = 0;
        for (int k = 0; k < 4; ++k) {
            const char c = s[i++];
            v <<= 4;
            if (c >= '0' && c <= '9') {
                v |= unsigned(c - '0');
            } else if (c >= 'a' && c <= 'f') {
                v |= unsigned(c - 'a' + 10);
            } else if (c >= 'A' && c <= 'F') {
                v |= unsigned(c - 'A' + 10);
            } else {
                error("bad \\u escape");
            }
        }
        return v;
    }

    std::string string() {
        if (s[i] != '"') { error("expected a string"); }
        ++i;
        std::string out;
        while (i < s.size() && s[i] != '"') {
            char c = s[i++];
            if (c != '\\') {
                out += c;
                continue;
            }
            if (i >= s.size()) { error("unterminated escape"); }
            c = s[i++];
            switch (c) {
                case '"': out += '"'; break;
                case '\\': out += '\\'; break;
                case '/': out += '/'; break;
                case 'b': out += '\b'; break;
                case 'f': out += '\f'; break;
                case 'n': out += '\n'; break;
                case 'r': out += '\r'; break;
                case 't': out += '\t'; break;
                case 'u': {
                    unsigned cp = hex4();
                    if (cp >= 0xD800 && cp < 0xDC00 && s.compare(i, 2, "\\u") == 0) {
                        i += 2;
                        const unsigned lo = hex4();
                        cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                    }
                    utf8(out, cp);
                    break;
                }
                default: error("bad escape");
            }
        }
        if (i >= s.size()) { error("unterminated string"); }
        ++i;
        return out;
    }

    Value value() {
        ws();
        if (i >= s.size()) { error("unexpected end"); }
        const char c = s[i];
        if (c == '{') {
            ++i;
            Value v = Value::object();
            ws();
            if (i < s.size() && s[i] == '}') {
                ++i;
                return v;
            }
            for (;;) {
                ws();
                std::string k = string();
                ws();
                if (i >= s.size() || s[i] != ':') { error("expected ':'"); }
                ++i;
                if (v.find(k)) { error("duplicate key \"" + k + "\""); }
                v.obj.emplace_back(std::move(k), value());
                ws();
                if (i < s.size() && s[i] == ',') {
                    ++i;
                    continue;
                }
                if (i < s.size() && s[i] == '}') {
                    ++i;
                    return v;
                }
                error("expected ',' or '}'");
            }
        }
        if (c == '[') {
            ++i;
            Value v = Value::array();
            ws();
            if (i < s.size() && s[i] == ']') {
                ++i;
                return v;
            }
            for (;;) {
                v.arr.push_back(value());
                ws();
                if (i < s.size() && s[i] == ',') {
                    ++i;
                    continue;
                }
                if (i < s.size() && s[i] == ']') {
                    ++i;
                    return v;
                }
                error("expected ',' or ']'");
            }
        }
        if (c == '"') { return Value(string()); }
        if (literal("true")) { return Value(true); }
        if (literal("false")) { return Value(false); }
        if (literal("null")) { return Value(); }
        if (c == '-' || (c >= '0' && c <= '9')) {
            const char* start = s.c_str() + i;
            char* end = nullptr;
            const double d = std::strtod(start, &end);
            if (end == start) { error("bad number"); }
            i += size_t(end - start);
            return Value(d);
        }
        error(std::string("unexpected '") + c + "'");
    }
};

std::string number(double d) {
    if (!std::isfinite(d)) { return "null"; }
    if (d == std::floor(d) && std::fabs(d) < 1e15) {
        char buf[32];
        std::snprintf(buf, sizeof buf, "%.0f", d);
        return buf;
    }
    char buf[32];
    std::snprintf(buf, sizeof buf, "%.15g", d);  // the shortest of 15 / 17 digits that reads back exactly
    if (std::strtod(buf, nullptr) != d) { std::snprintf(buf, sizeof buf, "%.17g", d); }
    return buf;
}

std::string quote(const std::string& s) {
    std::string out = "\"";
    for (const char c : s) {
        switch (c) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:
                if (static_cast<unsigned char>(c) < 0x20) {
                    char buf[8];
                    std::snprintf(buf, sizeof buf, "\\u%04x", unsigned(c));
                    out += buf;
                } else {
                    out += c;
                }
        }
    }
    return out + "\"";
}

bool scalar_array(const Value& v) {
    for (const auto& x : v.arr) {
        if (x.type == Value::Array || x.type == Value::Object) { return false; }
    }
    return true;
}

void emit(std::string& out, const Value& v, int indent, int depth) {
    const std::string pad(size_t(indent) * (depth + 1), ' '), end_pad(size_t(indent) * depth, ' ');
    switch (v.type) {
        case Value::Null: out += "null"; break;
        case Value::Bool: out += v.b ? "true" : "false"; break;
        case Value::Number: out += number(v.num); break;
        case Value::String: out += quote(v.str); break;
        case Value::Array:
            if (v.arr.empty()) {
                out += "[]";
            } else if (scalar_array(v)) {  // short lists of numbers stay on one line
                out += "[";
                for (size_t k = 0; k < v.arr.size(); ++k) {
                    if (k) { out += ", "; }
                    emit(out, v.arr[k], indent, depth + 1);
                }
                out += "]";
            } else {
                out += "[\n";
                for (size_t k = 0; k < v.arr.size(); ++k) {
                    out += pad;
                    emit(out, v.arr[k], indent, depth + 1);
                    out += k + 1 < v.arr.size() ? ",\n" : "\n";
                }
                out += end_pad + "]";
            }
            break;
        case Value::Object:
            if (v.obj.empty()) {
                out += "{}";
                break;
            }
            out += "{\n";
            for (size_t k = 0; k < v.obj.size(); ++k) {
                out += pad + quote(v.obj[k].first) + ": ";
                emit(out, v.obj[k].second, indent, depth + 1);
                out += k + 1 < v.obj.size() ? ",\n" : "\n";
            }
            out += end_pad + "}";
            break;
    }
}

}  // namespace

Value parse(const std::string& text) {
    Parser p{text};
    Value v = p.value();
    p.ws();
    if (p.i != text.size()) { p.error("trailing characters"); }
    return v;
}

std::string dump(const Value& v, int indent) {
    std::string out;
    emit(out, v, indent, 0);
    return out;
}

Value read_file(const std::string& path) {
    std::ifstream f(path);
    if (!f) { throw std::runtime_error("cannot read " + path); }
    std::stringstream ss;
    ss << f.rdbuf();
    try {
        return parse(ss.str());
    } catch (const std::exception& e) { throw std::runtime_error(path + ": " + e.what()); }
}

void write_file(const std::string& path, const Value& v) {
    const std::string tmp = path + ".tmp";
    {
        std::ofstream f(tmp);
        if (!f) { throw std::runtime_error("cannot write " + tmp); }
        f << dump(v) << "\n";
        if (!f) { throw std::runtime_error("cannot write " + tmp); }
    }
    std::filesystem::rename(tmp, path);
}

const Value* get(const Value& root, const std::string& dotted) {
    const Value* node = &root;
    size_t start = 0;
    for (;;) {
        const size_t dot = dotted.find('.', start);
        node = node->find(dotted.substr(start, dot == std::string::npos ? std::string::npos : dot - start));
        if (!node || dot == std::string::npos) { return node; }
        start = dot + 1;
    }
}

bool set(Value& root, const std::string& dotted, Value v) {
    Value* node = &root;
    size_t start = 0;
    for (;;) {
        const size_t dot = dotted.find('.', start);
        const std::string part = dotted.substr(start, dot == std::string::npos ? std::string::npos : dot - start);
        if (dot == std::string::npos) {
            if (!node->is_object()) { return false; }
            (*node)[part] = std::move(v);
            return true;
        }
        node = node->find(part);
        if (!node) { return false; }
        start = dot + 1;
    }
}

}  // namespace lbc::json
