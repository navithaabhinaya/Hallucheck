"""
Golden Set for Spec Deviation Detection (Stage 3, third slice)

Each case has a spec, generated code, optional test cases, and a hand-labeled
EXPECTED verdict for the requested function(s) -- following the same pattern
as the other two golden sets.

Deliberately includes:
- A correct implementation WITH test cases -- should PASS via test_execution
  (ground truth, the strongest evidence path)
- A deliberately WRONG implementation WITH test cases -- should FAIL via
  test_execution, with specific failing inputs in the reason
- A correct implementation WITHOUT test cases -- forces the LLM-judge
  fallback path, tests that the judge doesn't falsely flag correct code
  just because it can't run it
- A case where a non-requested helper/creep function exists alongside the
  target function -- regression test for the entry #17 fix: only the
  REQUESTED function should appear in results, not the helper
"""

from agents.spec_deviation_checker import TestCase

GOLDEN_SET = [
    {
        "id": "case_01_correct_with_tests",
        "spec": (
            "Write a function `is_even(n: int) -> bool` that returns True if "
            "a number is even, False otherwise."
        ),
        "code": '''
def is_even(n: int) -> bool:
    """Check if a number is even."""
    return n % 2 == 0
''',
        "test_cases": [
            TestCase(function_name="is_even", args=(4,), kwargs={}, expected_output=True),
            TestCase(function_name="is_even", args=(7,), kwargs={}, expected_output=False),
        ],
        "expected_verdicts": {
            "is_even": "pass",
        },
    },
    {
        "id": "case_02_wrong_with_tests",
        "spec": (
            "Write a function `is_even(n: int) -> bool` that returns True if "
            "a number is even, False otherwise."
        ),
        "code": '''
def is_even(n: int) -> bool:
    """Check if a number is even."""
    return n % 2 == 1
''',
        "test_cases": [
            TestCase(function_name="is_even", args=(4,), kwargs={}, expected_output=True),
            TestCase(function_name="is_even", args=(7,), kwargs={}, expected_output=False),
        ],
        "expected_verdicts": {
            "is_even": "fail",
        },
    },
    {
        "id": "case_03_correct_no_tests_llm_fallback",
        "spec": (
            "Write a function `reverse_string(s: str) -> str` that returns the "
            "input string reversed."
        ),
        "code": '''
def reverse_string(s: str) -> str:
    """Return the input string reversed."""
    return s[::-1]
''',
        "test_cases": [],  # deliberately none -- forces LLM-judge path
        "expected_verdicts": {
            "reverse_string": "pass",
        },
    },
    {
        "id": "case_04_helper_excluded_from_results",
        "spec": (
            "Write a function `word_count(text: str) -> int` that returns the "
            "number of words in a string."
        ),
        "code": '''
def word_count(text: str) -> int:
    """Return the number of words in a string."""
    cleaned = _normalize_whitespace(text)
    return len(cleaned.split())

def _normalize_whitespace(text: str) -> str:
    """Helper: collapse multiple spaces, strip leading/trailing."""
    return " ".join(text.split())
''',
        "test_cases": [
            TestCase(function_name="word_count", args=("hello   world",), kwargs={}, expected_output=2),
        ],
        # KEY TEST: only word_count should appear -- _normalize_whitespace
        # was never requested, so spec_deviation_checker should not judge it
        # at all (that's scope_creep_checker's job). Its ABSENCE from
        # expected_verdicts is itself the assertion, checked in scorer.py.
        "expected_verdicts": {
            "word_count": "pass",
        },
    },
]
