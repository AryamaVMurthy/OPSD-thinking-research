from __future__ import annotations


MATH_SUFFIX = (
    "Please reason step by step, and put your final answer within \\boxed{}."
)

LCB_SYSTEM = (
    "You are an expert Python programmer. You will be given a question "
    "(problem specification) and will generate a correct Python program that "
    "matches the specification and passes all tests."
)


def math_messages(problem: str) -> list[dict[str, str]]:
    return [{"role": "user", "content": f"{problem}\n\n{MATH_SUFFIX}"}]


def lcb_messages(question_content: str, starter_code: str = "") -> list[dict[str, str]]:
    if starter_code:
        format_instruction = (
            "You will use the following starter code to write the solution and "
            "enclose your code within delimiters.\n"
            f"```python\n{starter_code}\n```"
        )
    else:
        format_instruction = (
            "Read inputs from stdin, solve the problem, and write the answer to "
            "stdout. Enclose the complete program in a Python code block."
        )
    user = (
        f"### Question:\n{question_content}\n\n"
        f"### Format:\n{format_instruction}\n\n"
        "### Answer: (use the provided format with backticks)"
    )
    return [
        {"role": "system", "content": LCB_SYSTEM},
        {"role": "user", "content": user},
    ]
