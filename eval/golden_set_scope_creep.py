"""
Golden Set for Scope Creep Detection (Stage 3, second slice)

Each case has a spec + generated code, with hand-labeled EXPECTED verdicts
per function name. Unlike the fabricated-API golden set (fully objective:
a package either exists or not), scope creep is inherently judgment-based
-- so these labels reflect a REASONABLE human reviewer's call, and the
scorer measures how well the LLM-judge agrees with that reasonable call.

Deliberately includes:
- A clean case: only requested functions, nothing extra (checks no false
  positives on legitimate code)
- A clear scope-creep case: an obviously unrelated addition (analytics,
  logging to an external service) nobody asked for
- A "fair helper" case: a private helper function that wasn't explicitly
  named in the spec but clearly supports a requested function -- this is
  the important nuance test. A checker that flags every helper as scope
  creep is USELESS in practice (real code always has small helpers) --
  this case specifically tests that the checker is not overly strict.
- A borderline case marked "uncertain" is intentionally NOT included here
  yet -- borderline/ambiguous cases are harder to hand-label with
  confidence and are a good candidate to add once the checker is running
  in practice and real ambiguous examples are seen, rather than inventing
  one that might not reflect real ambiguity.
"""

GOLDEN_SET = [
    {
        "id": "case_01_clean_no_extras",
        "spec": (
            "Write a function `add(a: int, b: int) -> int` that returns the "
            "sum of two integers."
        ),
        "code": '''
def add(a: int, b: int) -> int:
    """Return the sum of two integers."""
    return a + b
''',
        "expected_verdicts": {
            "add": "in_scope",
        },
    },
    {
        "id": "case_02_clear_scope_creep",
        "spec": (
            "Write a function `is_palindrome(s: str) -> bool` that checks if a "
            "string is a palindrome, ignoring case and spaces."
        ),
        "code": '''
def is_palindrome(s: str) -> bool:
    """Check if a string is a palindrome."""
    cleaned = s.lower().replace(" ", "")
    return cleaned == cleaned[::-1]

def send_analytics_event(event_name: str):
    """Send a usage analytics event to a tracking server."""
    print(f"Tracking: {event_name}")
''',
        "expected_verdicts": {
            "is_palindrome": "in_scope",
            "send_analytics_event": "scope_creep",
        },
    },
    {
        "id": "case_03_fair_helper_not_creep",
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
    """Helper: collapse multiple spaces into one, strip leading/trailing."""
    return " ".join(text.split())
''',
        "expected_verdicts": {
            "word_count": "in_scope",
            "_normalize_whitespace": "in_scope",  # the key test: a reasonable
            # helper, not explicitly named in the spec, should NOT be flagged
        },
    },
]
