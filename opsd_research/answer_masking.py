"""Conservative surface-equivalence checks for answer-masked artifacts."""

from __future__ import annotations

import re
from fractions import Fraction


PROBLEM_ONLY_GUIDANCE_PROTOCOL = "problem-only-v1"
ANSWER_LEAKAGE_PROTOCOL = "surface-equivalence-and-result-claim-v2"

_LATEX_FRACTION = re.compile(
    r"\\(?:dfrac|tfrac|frac)\s*\{([^{}]+)\}\s*\{([^{}]+)\}"
)
_LATEX_COMPACT_FRACTION = re.compile(
    r"\\(?:dfrac|tfrac|frac)\s*([+-]?\d)\s*([+-]?\d)"
)
_LATEX_SQUARE_ROOT = re.compile(r"\\sqrt\s*\{([^{}]+)\}")
_NUMERICAL_EXPRESSION_START = (
    r"(?:[-+]?(?:\d+(?:\.\d+)?|\.\d+)|"
    r"\\(?:dfrac|tfrac|frac|sqrt)\b|√)"
)
_RESULT_RELATION = re.compile(
    rf"(?:"
    rf"\b(?:is|are|equals?|gives?|yields?|becomes?|obtains?)\b|"
    rf"\b(?:equal|evaluates?|simplifies?)\s+to\b|"
    rf"(?<![<>!])=(?!=)"
    rf")\s*(?:exactly\s+|approximately\s+|about\s+)?"
    rf"{_NUMERICAL_EXPRESSION_START}",
    re.IGNORECASE,
)
_NUMERIC_TOKEN = re.compile(
    r"(?<![A-Za-z0-9_.])"
    r"[-+]?(?:\d+(?:\.\d+)?|\.\d+)"
    r"(?:\s*/\s*[-+]?(?:\d+(?:\.\d+)?|\.\d+))?%?"
    r"(?![A-Za-z0-9_.])"
)


def _plain_math(text: str) -> str:
    """Canonicalize common LaTeX presentation without evaluating expressions."""
    value = str(text)
    value = value.replace("−", "-").replace("–", "-")
    value = re.sub(r"\\(?:left|right)\b", "", value)
    previous = None
    while previous != value:
        previous = value
        value = _LATEX_FRACTION.sub(r"(\1)/(\2)", value)
        value = _LATEX_SQUARE_ROOT.sub(r"sqrt(\1)", value)
    value = _LATEX_COMPACT_FRACTION.sub(r"(\1)/(\2)", value)
    value = re.sub(
        r"√\s*\{?([A-Za-z0-9.+-]+)\}?",
        r"sqrt(\1)",
        value,
    )
    value = re.sub(r"\\(?:quad|qquad)\b", "", value)
    value = re.sub(r"\\[,!;:]", "", value)
    value = value.replace(r"\%", "%")
    value = value.replace("$", "")
    value = re.sub(r"\s+", " ", value).strip().lower()
    value = re.sub(r"\s*([/=+*^,])\s*", r"\1", value)
    value = re.sub(r"\(\s*", "(", value)
    value = re.sub(r"\s*\)", ")", value)
    value = re.sub(r"\s+-\s+", "-", value)
    # Parentheses introduced solely around atomic fraction terms do not alter
    # the expression and otherwise defeat a literal surface check.
    value = re.sub(r"\(([+-]?(?:\d+(?:\.\d+)?|[a-z]))\)", r"\1", value)
    return value


def contains_reference_answer(text: str, reference_answer: str) -> bool:
    """Return whether text contains a common surface-equivalent answer form."""
    answer = _plain_math(reference_answer)
    candidate = _plain_math(text)
    if not answer:
        return False
    if re.search(
        rf"(?<![A-Za-z0-9]){re.escape(answer)}(?![A-Za-z0-9])",
        candidate,
    ) is not None:
        return True
    answer_value = _numeric_value(answer)
    if answer_value is None:
        return False
    return any(
        _numeric_value(match.group(0)) == answer_value
        for match in _NUMERIC_TOKEN.finditer(candidate)
    )


def _numeric_value(text: str) -> Fraction | None:
    """Parse one exact decimal/fraction/percent token without executing code."""
    token = re.sub(r"\s+", "", str(text))
    percent = token.endswith("%")
    if percent:
        token = token[:-1]
    if not re.fullmatch(
        r"[-+]?(?:\d+(?:\.\d+)?|\.\d+)"
        r"(?:/[-+]?(?:\d+(?:\.\d+)?|\.\d+))?",
        token,
    ):
        return None
    try:
        if "/" in token:
            numerator, denominator = token.split("/", 1)
            value = Fraction(numerator) / Fraction(denominator)
        else:
            value = Fraction(token)
    except (ValueError, ZeroDivisionError):
        return None
    return value / 100 if percent else value


def contains_explicit_numerical_result(text: str) -> bool:
    """Detect an asserted computed value while allowing procedural constants."""
    return _RESULT_RELATION.search(str(text)) is not None
