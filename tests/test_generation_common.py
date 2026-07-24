from __future__ import annotations

import unittest

from opsd_research.generation_common import extract_last_boxed, split_thinking
from opsd_research.prompts import lcb_messages, math_messages, render_thinking_prompt


class FakeQwenTokenizer:
    def apply_chat_template(self, _messages, *, enable_thinking, **_kwargs):
        if enable_thinking:
            return "<assistant>"
        return "<assistant><think>\n\n</think>\n\n"


class GenerationTests(unittest.TestCase):
    def test_nested_last_boxed(self):
        text = r"first \boxed{1}, final \boxed{\frac{3}{x+{1}}}"
        self.assertEqual(extract_last_boxed(text), r"\frac{3}{x+{1}}")

    def test_incomplete_box_is_not_an_answer(self):
        self.assertIsNone(extract_last_boxed(r"\boxed{123"))

    def test_thinking_is_split(self):
        reasoning, final, present = split_thinking("<think>work</think>\\boxed{2}")
        self.assertEqual(reasoning, "work")
        self.assertEqual(final, r"\boxed{2}")
        self.assertTrue(present)

    def test_prompts_request_expected_output(self):
        self.assertIn(r"\boxed{}", math_messages("x")[0]["content"])
        messages = lcb_messages("solve", "")
        self.assertIn("Python code block", messages[1]["content"])

    def test_qwen_thinking_switch_is_checked(self):
        self.assertEqual(
            render_thinking_prompt(FakeQwenTokenizer(), math_messages("x")),
            "<assistant>",
        )


if __name__ == "__main__":
    unittest.main()
