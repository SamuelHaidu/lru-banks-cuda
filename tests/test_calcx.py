"""`model/calcx.py`: the answer check and the step-level partial credit used by `model/grpo.py`."""

import unittest

import tests.helpers  # noqa: F401  (pins the device before keras/torch use)

from model import calcx


def turn(expr: str, res: str) -> str:
    return f'<call>calculator(expression="{expr}")</call></assistant><result>{res}</result><assistant>'


# svamp chal-1: "Each pack of dvds costs 76 dollars. If there is a discount of 25 dollars on each pack, ..."; the
# reference solution is one call, 76 - 25 = 51
DVD = {"gold": "51", "steps": [51.0], "qnums": [76.0, 25.0]}
# a three-step reference
THREE = {"gold": "18", "steps": [13.0, 9.0, 18.0], "qnums": [16.0, 3.0, 4.0, 2.0]}


class TestStepCredit(unittest.TestCase):
    @unittest.skipUnless((calcx.DATA / "test.docs.jsonl").exists(),
                         "needs the Calc-X dataset: uv run --with pyarrow python datasets/build_calc_x.py --download")
    def test_reference_steps_are_read_from_the_document(self):
        items = calcx.load_questions(calcx.DATA / "test.docs.jsonl", ["svamp"], with_steps=True)
        dvd = next(it for it in items if it["id"] == "svamp__chal-1")
        self.assertEqual(dvd["steps"], [51.0])
        self.assertEqual(dvd["qnums"], [76.0, 25.0])

    def test_right_step_then_an_extra_call(self):
        out = turn("76 - 25", "51") + turn("51 * 76", "3876") + "The answer is 3876.</assistant>"
        self.assertAlmostEqual(calcx.step_credit(out, 2, DVD), 0.5)  # 1 match / max(2 calls, 1 step)
        self.assertAlmostEqual(calcx.reward("3876", out, 2, DVD, 0.5), 0.25)
        self.assertEqual(calcx.reward("3876", out, 2, DVD, 0.0), 0.0)  # binary

    def test_correct_answer_is_one(self):
        out = turn("76 - 25", "51") + "The answer is 51.</assistant>"
        self.assertEqual(calcx.reward("51", out, 1, DVD, 0.5), 1.0)

    def test_restating_a_question_number_earns_nothing(self):
        item = {"gold": "100", "steps": [76.0, 100.0], "qnums": [76.0, 24.0]}
        out = turn("76", "76") + "The answer is 7.</assistant>"
        self.assertEqual(calcx.step_credit(out, 1, item), 0.0)

    def test_each_reference_step_matches_once(self):
        out = turn("16 - 3", "13") + turn("10 + 3", "13") + turn("13 - 4", "9") + "The answer is 9.</assistant>"
        self.assertAlmostEqual(calcx.step_credit(out, 3, THREE), 2 / 3)

    def test_extra_calls_lower_the_credit(self):
        base = turn("16 - 3", "13")
        spam = base + "".join(turn(f"{i} + 1", str(i + 1)) for i in range(20, 25))
        self.assertGreater(calcx.step_credit(base, 1, THREE), calcx.step_credit(spam, 6, THREE))

    def test_tool_errors_are_not_steps(self):
        out = turn("1/0", "Error: division by zero") + "The answer is 5.</assistant>"
        self.assertEqual(calcx.step_credit(out, 1, THREE), 0.0)


if __name__ == "__main__":
    unittest.main()
