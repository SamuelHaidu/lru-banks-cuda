#pragma once
// A minimal JSON value for run configs and checkpoint headers: parse, pretty-print (objects keep their key order),
// dotted-key access. No external dependency.

#include <string>
#include <utility>
#include <vector>

namespace lbc::json {

struct Value {
    enum Type { Null, Bool, Number, String, Array, Object } type = Null;
    bool b = false;
    double num = 0.0;
    std::string str;
    std::vector<Value> arr;
    std::vector<std::pair<std::string, Value>> obj;

    Value() = default;
    Value(bool v) : type(Bool), b(v) {}
    Value(int v) : type(Number), num(v) {}
    Value(long v) : type(Number), num(double(v)) {}
    Value(long long v) : type(Number), num(double(v)) {}
    Value(unsigned long v) : type(Number), num(double(v)) {}
    Value(unsigned long long v) : type(Number), num(double(v)) {}
    Value(double v) : type(Number), num(v) {}
    Value(float v) : type(Number), num(v) {}
    Value(const char* v) : type(String), str(v) {}
    Value(std::string v) : type(String), str(std::move(v)) {}
    static Value object() {
        Value v;
        v.type = Object;
        return v;
    }
    static Value array() {
        Value v;
        v.type = Array;
        return v;
    }

    bool is_null() const { return type == Null; }
    bool is_object() const { return type == Object; }
    const Value* find(const std::string& key) const;
    Value* find(const std::string& key);
    Value& operator[](const std::string& key);  // objects: inserted (null) when missing
    void push(Value v) { arr.push_back(std::move(v)); }
    bool operator==(const Value& o) const;
    bool operator!=(const Value& o) const { return !(*this == o); }
};

Value parse(const std::string& text);  // throws std::runtime_error (with the position) on malformed input
std::string dump(const Value& v, int indent = 2);
Value read_file(const std::string& path);
void write_file(const std::string& path, const Value& v);  // written to path.tmp, then renamed

// "a.b.c" in nested objects; null when missing
const Value* get(const Value& root, const std::string& dotted);
// sets "a.b.c" (the parent objects must exist); returns false when a parent is missing
bool set(Value& root, const std::string& dotted, Value v);

}  // namespace lbc::json
