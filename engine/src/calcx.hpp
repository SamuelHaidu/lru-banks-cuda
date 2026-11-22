#pragma once
// The Calc-X calculator task, ported from rnn-research (datasets/build_agent_shell.py: calculator, fmt_number;
// model/calcx.py: calc, answer, correct, step_values, step_credit, reward, load_questions).
//
// The calculator evaluates what Python's `ast` would accept of an arithmetic expression: numbers, + - * / // % **,
// unary + -, parentheses, `^` read as `**`; ints are exact up to 127 bits (Python's are unbounded: past that they
// become floats here), the exponent is limited to |b| <= 100, results are formatted as fmt_number (an integral float
// under 1e15 as an int, else rounded to 6 decimals with the trailing zeros removed).

#include <optional>
#include <string>
#include <vector>

namespace lbc::calcx {

std::string calculator(const std::string& expression);  // the dataset's tool: a result or "Error: ..."
std::string calc(const std::string& expression);        // calcx.calc: with the sampling guards
std::optional<double> number(const std::string& s);     // a text as a number (through the calculator)
std::string py_repr(const std::string& s);              // Python's repr() of a str

struct Question {
    std::string source, id, gold, prompt;  // prompt: the document up to and including the first <assistant>
    std::vector<double> steps, qnums;      // the reference's calculator results; the numbers in the question
};

std::optional<std::string> answer(const std::string& output);  // the last "The answer is X"
bool correct(const std::optional<std::string>& pred, const std::string& gold);
bool is_choice(const std::string& gold);
std::vector<double> step_values(const std::string& text);
double step_credit(const std::string& output, int calls, const Question& q);
double reward(const std::optional<std::string>& pred, const std::string& output, int calls, const Question& q,
              double partial);
// `{split}.docs.jsonl`: sources (empty: all), multiple choice dropped when exclude_choice
std::vector<Question> load_questions(const std::string& path, const std::vector<std::string>& sources = {},
                                     bool exclude_choice = false);
// the first `<call>calculator(expression="...")</call>` in a turn: the JSON-decoded expression; nullopt when there
// is no call; throws std::runtime_error when the string does not decode ("Error: bad call")
std::optional<std::string> find_call(const std::string& turn);

}  // namespace lbc::calcx
