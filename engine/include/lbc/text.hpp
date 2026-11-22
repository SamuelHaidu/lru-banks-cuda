#pragma once
// Text with checked results: printf-style formatting into a std::string (no fixed buffers, nothing truncated) and
// number parsing that rejects malformed input (std::atoi turns "--rows 1O" into 1 and "abc" into 0, silently).

#include <charconv>
#include <cstdarg>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>

namespace lbc {

// printf into a std::string; the compiler checks the format against the arguments (-Wformat=2)
// (C varargs on purpose: the format attribute, hence the compile-time check, needs them)
// NOLINTNEXTLINE(cert-dcl50-cpp)
[[gnu::format(printf, 1, 2)]] inline std::string strformat(const char* fmt, ...) {
    va_list args{};
    va_start(args, fmt);
    va_list again{};
    va_copy(again, args);
    const int n = std::vsnprintf(nullptr, 0, fmt, args);
    va_end(args);
    std::string out(n > 0 ? size_t(n) : 0, '\0');
    const int written = n > 0 ? std::vsnprintf(out.data(), out.size() + 1, fmt, again) : n;
    va_end(again);
    if (n < 0 || written != n) { throw std::runtime_error(std::string("cannot format \"") + fmt + "\""); }
    return out;
}

// the whole of s as a T (int, long, float, double, ...), or an error naming what was being read
template <class T>
T parse_number(std::string_view s, std::string_view what) {
    T value{};
    const char* end = s.data() + s.size();
    // NOLINTNEXTLINE(bugprone-suspicious-stringview-data-usage): from_chars takes the end, no terminator needed
    const auto [stop, ec] = std::from_chars(s.data(), end, value);
    if (s.empty() || ec != std::errc() || stop != end) {
        throw std::invalid_argument(std::string(what) + ": not a valid number: '" + std::string(s) + "'");
    }
    return value;
}

inline int parse_int(std::string_view s, std::string_view what) { return parse_number<int>(s, what); }

inline long parse_long(std::string_view s, std::string_view what) { return parse_number<long>(s, what); }

inline float parse_float(std::string_view s, std::string_view what) { return parse_number<float>(s, what); }

inline double parse_double(std::string_view s, std::string_view what) { return parse_number<double>(s, what); }

}  // namespace lbc
