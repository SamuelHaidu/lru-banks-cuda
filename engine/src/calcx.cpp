#include "calcx.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <memory>
#include <stdexcept>

#include "json.hpp"

namespace lbc::calcx {
namespace {

using i128 = __int128;

// ---- Python repr of floats (shortest round trip; exponent form below 1e-4 or from 1e16) -------------------------
std::string float_repr(double x, bool point_zero) {
    if (std::isnan(x)) { return "nan"; }
    if (std::isinf(x)) { return x < 0 ? "-inf" : "inf"; }
    if (x == 0.0) { return std::signbit(x) ? (point_zero ? "-0.0" : "-0") : (point_zero ? "0.0" : "0"); }
    char buf[64];
    int p = 1;
    for (; p <= 17; ++p) {
        std::snprintf(buf, sizeof buf, "%.*e", p - 1, x);
        if (std::strtod(buf, nullptr) == x) { break; }
    }
    std::string s = buf, digits;
    const bool neg = s[0] == '-';
    size_t e = s.find('e');
    for (size_t i = neg ? 1 : 0; i < e; ++i) {
        if (s[i] != '.') { digits += s[i]; }
    }
    const int exp = std::atoi(s.c_str() + e + 1);
    while (digits.size() > 1 && digits.back() == '0') { digits.pop_back(); }
    std::string out = neg ? "-" : "";
    if (exp >= -4 && exp < 16) {
        if (exp < 0) {
            out += "0." + std::string(size_t(-exp - 1), '0') + digits;
        } else if (size_t(exp) + 1 >= digits.size()) {
            out += digits + std::string(size_t(exp) + 1 - digits.size(), '0') + (point_zero ? ".0" : "");
        } else {
            out += digits.substr(0, size_t(exp) + 1) + "." + digits.substr(size_t(exp) + 1);
        }
    } else {
        out += digits.substr(0, 1);
        if (digits.size() > 1) { out += "." + digits.substr(1); }
        char eb[16];
        std::snprintf(eb, sizeof eb, "e%c%02d", exp < 0 ? '-' : '+', std::abs(exp));
        out += eb;
    }
    return out;
}

std::string i128_str(i128 v) {
    if (v == 0) { return "0"; }
    const bool neg = v < 0;
    unsigned __int128 u = neg ? (unsigned __int128)(-(v + 1)) + 1 : (unsigned __int128)v;
    std::string s;
    while (u) {
        s += char('0' + int(u % 10));
        u /= 10;
    }
    if (neg) { s += '-'; }
    return std::string(s.rbegin(), s.rend());
}

// ---- values -------------------------------------------------------------------------------------------------------
struct Num {
    enum Kind { Int, Float, Complex } k = Int;

    i128 i = 0;
    double f = 0, im = 0;  // Float: f; Complex: f + im j

    double as_f() const { return k == Int ? double(i) : f; }
};

struct DivZero {};

struct Unsupported {};

struct Overflow {};

Num mk_f(double x) { return Num{Num::Float, 0, x, 0}; }

Num mk_i(i128 x) { return Num{Num::Int, x, 0, 0}; }

i128 floordiv_i(i128 a, i128 b) {
    i128 q = a / b;
    if ((a % b != 0) && ((a < 0) != (b < 0))) { --q; }
    return q;
}

// CPython float_divmod
void float_divmod(double vx, double wx, double& floordiv, double& mod) {
    mod = std::fmod(vx, wx);
    double div = (vx - mod) / wx;
    if (mod != 0.0) {
        if ((wx < 0) != (mod < 0)) {
            mod += wx;
            div -= 1.0;
        }
    } else {
        mod = std::copysign(0.0, wx);
    }
    if (div != 0.0) {
        floordiv = std::floor(div);
        if (div - floordiv > 0.5) { floordiv += 1.0; }
    } else {
        floordiv = std::copysign(0.0, vx / wx);
    }
}

Num binop(char op, const Num& a, const Num& b) {
    if (a.k == Num::Complex || b.k == Num::Complex) {
        throw Unsupported{};  // (not reached by the dataset)
    }
    if (op == 'p') {  // power
        const double bf = b.as_f();
        if (std::fabs(bf) > 100) {
            throw Unsupported{};  // "exponent too large" (a ValueError)
        }
        if (a.k == Num::Int && b.k == Num::Int && b.i >= 0) {
            i128 r = 1, base = a.i;
            bool over = false;
            for (i128 e = b.i; e > 0; --e) {
                if (__builtin_mul_overflow(r, base, &r)) {
                    over = true;
                    break;
                }
            }
            if (!over) { return mk_i(r); }
            const double x = std::pow(double(a.i), double(b.i));
            if (std::isinf(x)) { throw Overflow{}; }
            return mk_f(x);
        }
        const double x = a.as_f(), y = bf;
        if (x == 0.0 && y < 0) { throw DivZero{}; }
        if (x < 0 && y != std::floor(y)) {  // Python 3: a complex result
            const double mag = std::pow(-x, y), ang = 3.14159265358979323846 * y;
            Num c;
            c.k = Num::Complex;
            c.f = mag * std::cos(ang);
            c.im = mag * std::sin(ang);
            return c;
        }
        const double r = std::pow(x, y);
        if (std::isinf(r) && std::isfinite(x) && std::isfinite(y)) { throw Overflow{}; }
        return mk_f(r);
    }
    if (a.k == Num::Int && b.k == Num::Int) {
        i128 r;
        switch (op) {
            case '+':
                if (!__builtin_add_overflow(a.i, b.i, &r)) { return mk_i(r); }
                return mk_f(double(a.i) + double(b.i));
            case '-':
                if (!__builtin_sub_overflow(a.i, b.i, &r)) { return mk_i(r); }
                return mk_f(double(a.i) - double(b.i));
            case '*':
                if (!__builtin_mul_overflow(a.i, b.i, &r)) { return mk_i(r); }
                return mk_f(double(a.i) * double(b.i));
            case '/':  // Python rounds the exact rational once
                if (b.i == 0) { throw DivZero{}; }
                if (a.i % b.i == 0) { return mk_f(double(a.i / b.i)); }
                return mk_f(double((long double)a.i / (long double)b.i));
            case 'f':
                if (b.i == 0) { throw DivZero{}; }
                return mk_i(floordiv_i(a.i, b.i));
            case '%': {
                if (b.i == 0) { throw DivZero{}; }
                return mk_i(a.i - floordiv_i(a.i, b.i) * b.i);
            }
        }
    }
    const double x = a.as_f(), y = b.as_f();
    switch (op) {
        case '+': return mk_f(x + y);
        case '-': return mk_f(x - y);
        case '*': return mk_f(x * y);
        case '/':
            if (y == 0.0) { throw DivZero{}; }
            return mk_f(x / y);
        case 'f':
        case '%': {
            if (y == 0.0) { throw DivZero{}; }
            double fd, md;
            float_divmod(x, y, fd, md);
            return mk_f(op == 'f' ? fd : md);
        }
    }
    throw Unsupported{};
}

// ---- parser: the AST first (so evaluation order is Python's), then the evaluation --------------------------------
struct Node {
    char op = 0;  // 0: number, 'n': negate, 'u': unary plus, else a binary operator (+ - * / f % p)
    Num v;
    std::unique_ptr<Node> a, b;
};

struct SyntaxErr {};

struct Parser {
    const std::string& s;
    size_t i = 0;

    explicit Parser(const std::string& str) : s(str) {}

    void ws() {
        while (i < s.size() && (s[i] == ' ' || s[i] == '\t' || s[i] == '\f')) { ++i; }
    }

    bool eat(const char* t) {
        ws();
        const size_t n = std::char_traits<char>::length(t);
        if (s.compare(i, n, t) == 0) {
            if (n == 1 && (t[0] == '*' || t[0] == '/') && i + 1 < s.size() && s[i + 1] == t[0]) { return false; }
            i += n;
            return true;
        }
        return false;
    }

    std::unique_ptr<Node> bin(char op, std::unique_ptr<Node> a, std::unique_ptr<Node> b) {
        auto n = std::make_unique<Node>();
        n->op = op;
        n->a = std::move(a);
        n->b = std::move(b);
        return n;
    }

    std::unique_ptr<Node> expr() {
        auto n = term();
        for (;;) {
            if (eat("+")) {
                n = bin('+', std::move(n), term());
            } else if (eat("-")) {
                n = bin('-', std::move(n), term());
            } else {
                return n;
            }
        }
    }

    std::unique_ptr<Node> term() {
        auto n = factor();
        for (;;) {
            if (eat("//")) {
                n = bin('f', std::move(n), factor());
            } else if (eat("*")) {
                n = bin('*', std::move(n), factor());
            } else if (eat("/")) {
                n = bin('/', std::move(n), factor());
            } else if (eat("%")) {
                n = bin('%', std::move(n), factor());
            } else {
                return n;
            }
        }
    }

    std::unique_ptr<Node> factor() {
        if (eat("-")) { return bin('n', factor(), nullptr); }
        if (eat("+")) { return bin('u', factor(), nullptr); }
        return power();
    }

    std::unique_ptr<Node> power() {
        auto n = atom();
        if (eat("**")) { return bin('p', std::move(n), factor()); }
        return n;
    }

    std::unique_ptr<Node> atom() {
        ws();
        if (eat("(")) {
            auto n = expr();
            if (!eat(")")) { throw SyntaxErr{}; }
            return n;
        }
        return number();
    }

    std::unique_ptr<Node> number() {
        ws();
        const size_t st = i;
        auto n = std::make_unique<Node>();
        auto isd = [&](size_t k) { return k < s.size() && s[k] >= '0' && s[k] <= '9'; };
        if (s.compare(i, 2, "0x") == 0 || s.compare(i, 2, "0X") == 0 || s.compare(i, 2, "0o") == 0
            || s.compare(i, 2, "0O") == 0 || s.compare(i, 2, "0b") == 0 || s.compare(i, 2, "0B") == 0) {
            const char kind = char(std::tolower(s[i + 1]));
            const int base = kind == 'x' ? 16 : kind == 'o' ? 8 : 2;
            i += 2;
            i128 v = 0;
            size_t nd = 0;
            for (;; ++i) {
                if (i < s.size() && s[i] == '_' && nd) { continue; }
                int d = -1;
                if (i < s.size()) {
                    const char ch = char(std::tolower(s[i]));
                    if (ch >= '0' && ch <= '9') {
                        d = ch - '0';
                    } else if (ch >= 'a' && ch <= 'f') {
                        d = 10 + ch - 'a';
                    }
                }
                if (d < 0 || d >= base) { break; }
                if (__builtin_mul_overflow(v, base, &v) || __builtin_add_overflow(v, d, &v)) { throw Overflow{}; }
                ++nd;
            }
            if (!nd) { throw SyntaxErr{}; }
            n->v = mk_i(v);
            return n;
        }
        std::string digits;
        bool is_float = false;
        auto run = [&]() {
            size_t k = 0;
            while (isd(i) || (i < s.size() && s[i] == '_' && k && isd(i + 1))) {
                if (s[i] != '_') {
                    digits += s[i];
                    ++k;
                }
                ++i;
            }
            return k;
        };
        const size_t int_digits = run();
        if (i < s.size() && s[i] == '.') {
            is_float = true;
            digits += '.';
            ++i;
            const size_t frac = run();
            if (!int_digits && !frac) { throw SyntaxErr{}; }
        } else if (!int_digits) {
            throw SyntaxErr{};
        }
        if (i < s.size() && (s[i] == 'e' || s[i] == 'E')) {
            size_t k = i + 1;
            if (k < s.size() && (s[k] == '+' || s[k] == '-')) { ++k; }
            if (isd(k)) {
                is_float = true;
                digits += 'e';
                if (s[i + 1] == '-' || s[i + 1] == '+') { digits += s[i + 1]; }
                i = k;
                run();
            }
        }
        if (i < s.size() && (s[i] == 'j' || s[i] == 'J')) {
            throw Unsupported{};  // an imaginary literal
        }
        if (i < s.size() && (std::isalnum(static_cast<unsigned char>(s[i])) || s[i] == '_')) { throw SyntaxErr{}; }
        if (is_float) {
            n->v = mk_f(std::strtod(digits.c_str(), nullptr));
        } else {
            if (digits.size() > 1 && digits[0] == '0' && digits.find_first_not_of('0') != std::string::npos) {
                throw SyntaxErr{};  // leading zeros in a decimal int
            }
            i128 v = 0;
            bool over = false;
            for (char ch : digits) {
                if (__builtin_mul_overflow(v, 10, &v) || __builtin_add_overflow(v, ch - '0', &v)) { over = true; }
            }
            n->v = over ? mk_f(std::strtod(digits.c_str(), nullptr)) : mk_i(v);
        }
        (void)st;
        return n;
    }
};

Num eval(const Node& n) {
    if (n.op == 0) { return n.v; }
    if (n.op == 'n' || n.op == 'u') {
        Num v = eval(*n.a);
        if (n.op == 'u') { return v; }
        if (v.k == Num::Int) {
            if (v.i == std::numeric_limits<i128>::min()) { return mk_f(-double(v.i)); }
            v.i = -v.i;
        } else {
            v.f = -v.f;
            v.im = -v.im;
        }
        return v;
    }
    const Num a = eval(*n.a), b = eval(*n.b);
    return binop(n.op, a, b);
}

std::string fmt_number(const Num& x) {
    if (x.k == Num::Int) { return i128_str(x.i); }
    if (x.k == Num::Complex) {  // str(complex)
        const std::string re = float_repr(x.f, false), im = float_repr(x.im, false);
        if (x.f == 0.0 && !std::signbit(x.f)) { return im + "j"; }
        return "(" + re + (std::signbit(x.im) || std::isnan(x.im) ? "" : "+") + im + "j)";
    }
    const double v = x.f;
    if (std::isfinite(v) && v == std::floor(v) && std::fabs(v) < 1e15) { return i128_str(i128(v)); }
    if (std::isnan(v)) { return "nan"; }
    if (std::isinf(v)) { return v < 0 ? "-inf" : "inf"; }
    char buf[512];
    std::snprintf(buf, sizeof buf, "%.6f", v);
    std::string s = buf;
    while (!s.empty() && s.back() == '0') { s.pop_back(); }
    while (!s.empty() && s.back() == '.') { s.pop_back(); }
    if (s == "-0") {
        s = "-0";  // Python keeps the sign: f"{-1e-9:.6f}" -> "-0.000000" -> "-0"
    }
    return s;
}

std::string replace_all(std::string s, const std::string& a, const std::string& b) {
    for (size_t p = 0; (p = s.find(a, p)) != std::string::npos; p += b.size()) { s.replace(p, a.size(), b); }
    return s;
}

bool close(double a, double b) { return std::fabs(a - b) <= 1e-4 * std::max(1.0, std::fabs(b)); }

}  // namespace

std::string py_repr(const std::string& s) {
    const bool dq = s.find('\'') != std::string::npos && s.find('"') == std::string::npos;
    const char q = dq ? '"' : '\'';
    std::string out(1, q);
    for (size_t i = 0; i < s.size(); ++i) {
        const unsigned char c = static_cast<unsigned char>(s[i]);
        if (c == '\\') {
            out += "\\\\";
        } else if (c == q) {
            out += std::string("\\") + char(q);
        } else if (c == '\n') {
            out += "\\n";
        } else if (c == '\r') {
            out += "\\r";
        } else if (c == '\t') {
            out += "\\t";
        } else if (c < 0x20 || c == 0x7f) {
            char b[8];
            std::snprintf(b, sizeof b, "\\x%02x", c);
            out += b;
        } else if (c == 0xC2 && i + 1 < s.size() && static_cast<unsigned char>(s[i + 1]) < 0xA0) {
            char b[8];  // U+0080..U+009F (C1 controls) are not printable
            std::snprintf(b, sizeof b, "\\x%02x", static_cast<unsigned char>(s[i + 1]) - 0x80 + 0x80);
            out += b;
            ++i;
        } else {
            out += char(c);
        }
    }
    return out + q;
}

std::string calculator(const std::string& expression) {
    const std::string e = replace_all(expression, "^", "**");
    try {
        Parser p(e);
        // Python's eval mode accepts surrounding spaces and a trailing newline
        auto root = p.expr();
        p.ws();
        while (p.i < e.size() && (e[p.i] == '\n' || e[p.i] == ' ' || e[p.i] == '\t')) { ++p.i; }
        if (p.i != e.size()) { throw SyntaxErr{}; }
        return fmt_number(eval(*root));
    } catch (const DivZero&) { return "Error: division by zero"; } catch (const Unsupported&) {
        return "Error: cannot evaluate " + py_repr(expression);
    } catch (const SyntaxErr&) { return "Error: cannot evaluate " + py_repr(expression); }
    // Overflow propagates (Python raises OverflowError here: calc() catches it)
}

std::string calc(const std::string& expression) {
    size_t pows = 0;
    for (size_t p = 0; (p = expression.find("**", p)) != std::string::npos; p += 2) { ++pows; }
    for (char ch : expression) { pows += ch == '^'; }
    if (expression.size() > 200 || pows > 2) { return "Error: cannot evaluate " + py_repr(expression); }
    try {
        return calculator(expression);
    } catch (...) { return "Error: cannot evaluate " + py_repr(expression); }
}

std::optional<double> number(const std::string& s) {
    std::string t = replace_all(replace_all(s, ",", ""), "$", "");
    const size_t a = t.find_first_not_of(" \t\n\r\f\v");
    if (a == std::string::npos) { return std::nullopt; }
    t = t.substr(a, t.find_last_not_of(" \t\n\r\f\v") - a + 1);
    std::string r;
    try {
        r = calculator(t);
    } catch (...) { return std::nullopt; }
    // float(r): r is a formatted number, "nan", "inf", a complex string or an error
    char* end = nullptr;
    const double v = std::strtod(r.c_str(), &end);
    if (r.empty() || end != r.c_str() + r.size()) { return std::nullopt; }
    return v;
}

std::optional<std::string> answer(const std::string& output) {
    // re.findall(r"The answer is (.+?)\.?</assistant>"): non-overlapping, no newline inside the match
    static const std::string key = "The answer is ", close_tag = "</assistant>";
    std::optional<std::string> last;
    size_t p = 0;
    while ((p = output.find(key, p)) != std::string::npos) {
        const size_t st = p + key.size();
        size_t q = output.find(close_tag, st + 1);
        const size_t nl = output.find('\n', st);
        if (q == std::string::npos) { break; }
        if (nl != std::string::npos && nl < q) {  // the lazy group cannot cross a newline
            p = p + 1;
            continue;
        }
        std::string g = output.substr(st, q - st);
        if (g.size() > 1 && g.back() == '.') {
            g.pop_back();  // `\.?` takes the period when the group keeps a char
        }
        last = g;
        p = q + close_tag.size();
    }
    return last;
}

bool is_choice(const std::string& gold) {
    return gold.size() >= 2 && gold[0] >= 'A' && gold[0] <= 'E' && gold[1] == ')';
}

bool correct(const std::optional<std::string>& pred, const std::string& gold) {
    if (!pred) { return false; }
    if (is_choice(gold)) {
        const size_t a = pred->find_first_not_of(" \t\n\r\f\v");
        return a != std::string::npos && (*pred)[a] == gold[0];
    }
    const auto a = number(*pred), b = number(gold);
    return a && b && close(*a, *b);
}

std::vector<double> step_values(const std::string& text) {
    std::vector<double> out;
    size_t p = 0;
    while ((p = text.find("<result>", p)) != std::string::npos) {
        const size_t st = p + 8, q = text.find("</result>", st);
        if (q == std::string::npos) { break; }
        const std::string r = text.substr(st, q - st);
        if (r.rfind("Error", 0) != 0) {
            if (auto v = number(r)) { out.push_back(*v); }
        }
        p = q + 9;
    }
    return out;
}

double step_credit(const std::string& output, int calls, const Question& q) {
    std::vector<double> gold;
    for (double v : q.steps) {
        bool given = false;
        for (double x : q.qnums) { given |= close(v, x); }
        if (!given) { gold.push_back(v); }
    }
    if (gold.empty()) { return 0.0; }
    int matched = 0;
    for (double v : step_values(output)) {
        for (size_t j = 0; j < gold.size(); ++j) {
            if (close(v, gold[j])) {
                ++matched;
                gold.erase(gold.begin() + long(j));
                break;
            }
        }
    }
    return double(matched) / double(std::max({calls, int(q.steps.size()), 1}));
}

double reward(const std::optional<std::string>& pred,
              const std::string& output,
              int calls,
              const Question& q,
              double partial) {
    if (correct(pred, q.gold)) { return 1.0; }
    return partial > 0 ? partial * step_credit(output, calls, q) : 0.0;
}

std::vector<Question> load_questions(const std::string& path,
                                     const std::vector<std::string>& sources,
                                     bool exclude_choice) {
    std::ifstream f(path);
    if (!f) { throw std::runtime_error("cannot read " + path); }
    std::vector<Question> out;
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty()) { continue; }
        const json::Value d = json::parse(line);
        Question q;
        q.source = d.find("source")->str;
        if (!sources.empty() && std::find(sources.begin(), sources.end(), q.source) == sources.end()) { continue; }
        q.id = d.find("id")->str;
        const std::string& t = d.find("text")->str;
        const auto g = answer(t);
        if (!g) { continue; }
        q.gold = *g;
        if (exclude_choice && is_choice(q.gold)) { continue; }
        const size_t cut = t.find("</user>") + 7, u = t.find("<user>") + 6;
        const std::string question = t.substr(u, cut - 7 - u);
        q.prompt = t.substr(0, cut) + "<assistant>";
        q.steps = step_values(t.substr(cut));
        const std::string qn = replace_all(question, ",", "");
        for (size_t i = 0; i < qn.size();) {  // \d+(?:\.\d+)?
            if (qn[i] >= '0' && qn[i] <= '9') {
                size_t j = i;
                while (j < qn.size() && qn[j] >= '0' && qn[j] <= '9') { ++j; }
                if (j + 1 < qn.size() && qn[j] == '.' && qn[j + 1] >= '0' && qn[j + 1] <= '9') {
                    ++j;
                    while (j < qn.size() && qn[j] >= '0' && qn[j] <= '9') { ++j; }
                }
                q.qnums.push_back(std::strtod(qn.substr(i, j - i).c_str(), nullptr));
                i = j;
            } else {
                ++i;
            }
        }
        out.push_back(std::move(q));
    }
    return out;
}

std::optional<std::string> find_call(const std::string& turn) {
    static const std::string head = "<call>calculator(expression=\"";
    size_t p = 0;
    while ((p = turn.find(head, p)) != std::string::npos) {
        const size_t st = p + head.size() - 1;  // the opening quote
        size_t i = st + 1;
        bool ok = false;
        while (i < turn.size()) {
            if (turn[i] == '\\') {
                if (i + 1 >= turn.size()) { break; }
                i += 2;
                continue;
            }
            if (turn[i] == '"') {
                ok = true;
                break;
            }
            ++i;
        }
        if (ok && turn.compare(i + 1, 8, ")</call>") == 0) {
            const json::Value v = json::parse(turn.substr(st, i + 1 - st));  // throws on a bad string
            if (v.type != json::Value::String) { throw std::runtime_error("bad call"); }
            return v.str;
        }
        p += 1;
    }
    return std::nullopt;
}

}  // namespace lbc::calcx
