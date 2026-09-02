"""
Spec Deviation Checker (Stage 2, Part 3 -- final verifier check)

Checks whether generated code actually does what the spec asked for, even
when nothing is fabricated (checked by fabricated_api_checker) and nothing
is extra (checked by scope_creep_checker). Code can pass both of those
checks and still be WRONG -- e.g. an off-by-one error, inverted logic, or
a subtly incorrect algorithm that still "looks like" a reasonable
implementation.

This is deliberately the LAST of the three verifier checks, and it's built
differently from the other two on purpose:

- fabricated_api_checker: fully deterministic (a package/attribute either
  exists or doesn't -- a lookup, no judgment)
- scope_creep_checker: fully LLM-judgment (whether something is "extra" is
  inherently a semantic call)
- spec_deviation_checker: a HYBRID of both, and the two paths matter for
  different reasons:
    1. TEST EXECUTION (objective, when test cases are available): actually
       run the function against known input/output pairs. This is strictly
       more reliable than any LLM judgment, because it's ground truth, not
       an opinion -- if you HAVE test cases, always prefer this path.
    2. LLM-JUDGE (used only when no test cases are available, or as a
       secondary signal): read the function and reason about whether its
       logic plausibly satisfies the spec's stated behavior. This is
       inherently weaker evidence than execution, and should be labeled as
       such in the output, not presented with the same confidence as a
       passing test.

Design decisions worth defending in an interview:

- Test execution takes priority over LLM judgment whenever both are
  available for the same function. A passing/failing test is ground truth;
  an LLM's opinion about code correctness is not, and conflating the two
  would understate how much more reliable execution-based checking is.

- Executing generated code is a REAL safety consideration, not just an
  implementation detail. This function only executes the SPECIFIC function
  under test, called with SPECIFIC, known test inputs -- never the whole
  file blindly, and never with attacker-controlled or unbounded input. A
  production version of this would need a proper sandbox (subprocess with
  resource/time limits, or a container) rather than running in-process;
  this version runs in-process for simplicity and states that limitation
  explicitly rather than pretending it doesn't exist.

- Verdict is 3-state again: PASS (test succeeded, or LLM judges it
  plausibly correct), FAIL (test failed, or LLM identifies a clear logic
  problem), UNCERTAIN (no test cases exist AND the LLM's read is
  ambiguous) -- consistent with the pattern used in both other checkers.
"""

from dataclasses import dataclass
from typing import List, Optional, Callable, Any
import json

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

from config import GEMINI_API_KEY, MODEL_NAME, PROMPT_VERSION
from agents.spec_parser import ParsedSpec, _extract_text
from agents.code_analyzer import FunctionInfo


@dataclass
class TestCase:
    """A single known (input, expected_output) pair for a function under test."""
    function_name: str
    args: tuple
    kwargs: dict
    expected_output: Any


@dataclass
class DeviationCheckResult:
    function_name: str
    verdict: str            # "pass" | "fail" | "uncertain"
    method: str              # "test_execution" | "llm_judge"
    reason: str
    prompt_version: str = PROMPT_VERSION


def run_test_execution(
    function_name: str,
    code_text: str,
    test_cases: List[TestCase],
) -> Optional[DeviationCheckResult]:
    """
    Actually execute the function against known test cases, IN-PROCESS.

    Returns None if no test cases exist for this function (caller should
    fall back to the LLM-judge path in that case) or if the function
    couldn't be extracted/executed at all (treated as a FAIL -- broken
    code is a real deviation, not something to silently skip).

    SAFETY NOTE (explicit, not hidden): this executes generated code
    in-process with exec(). This is acceptable ONLY because: (1) inputs
    are fixed, known test values defined in this codebase, never
    user/attacker-supplied; (2) it's a personal portfolio/dev-tool context,
    not a multi-tenant production service. A production version handling
    untrusted code would need a real sandbox -- subprocess isolation with
    strict resource/time limits, or a container -- not in-process exec().
    This tradeoff is deliberate and should be stated plainly if asked, not
    glossed over.
    """
    relevant_cases = [tc for tc in test_cases if tc.function_name == function_name]
    if not relevant_cases:
        return None

    namespace: dict = {}
    try:
        exec(code_text, namespace)  # noqa: S102 -- see safety note above
    except Exception as e:
        return DeviationCheckResult(
            function_name=function_name,
            verdict="fail",
            method="test_execution",
            reason=f"Code failed to execute at all: {e}",
        )

    func = namespace.get(function_name)
    if func is None or not callable(func):
        return DeviationCheckResult(
            function_name=function_name,
            verdict="fail",
            method="test_execution",
            reason=f"Function '{function_name}' not found or not callable after exec",
        )

    failures = []
    for tc in relevant_cases:
        try:
            actual = func(*tc.args, **tc.kwargs)
            if actual != tc.expected_output:
                failures.append(
                    f"input={tc.args}{tc.kwargs} -> expected {tc.expected_output!r}, got {actual!r}"
                )
        except Exception as e:
            failures.append(f"input={tc.args}{tc.kwargs} -> raised {e!r}")

    if failures:
        return DeviationCheckResult(
            function_name=function_name,
            verdict="fail",
            method="test_execution",
            reason=f"{len(failures)}/{len(relevant_cases)} test case(s) failed: " + "; ".join(failures),
        )

    return DeviationCheckResult(
        function_name=function_name,
        verdict="pass",
        method="test_execution",
        reason=f"All {len(relevant_cases)} test case(s) passed.",
    )


LLM_JUDGE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are reviewing whether a function's LOGIC plausibly satisfies \
a spec's stated behavior. You do NOT have test cases for this function, so you \
must reason from reading the code -- be appropriately cautious, since this is \
weaker evidence than actually running the code.

Judge each function as:
- "pass": the logic appears to correctly implement the described behavior
- "fail": there is a CLEAR logic problem (e.g. inverted condition, off-by-one, \
wrong operation) that would produce incorrect results
- "uncertain": you cannot confidently tell from reading alone -- do NOT guess

Respond ONLY with valid JSON, no markdown fences, no preamble, in this exact shape:
{{"function_name": "...", "verdict": "pass", "reason": "..."}}"""),
    ("human", """SPEC CRITERIA:
{criteria_list}

FUNCTION UNDER REVIEW:
{function_code}"""),
])


def run_llm_judge(
    function_name: str,
    function_source: str,
    parsed_spec: ParsedSpec,
) -> DeviationCheckResult:
    """LLM-judgment fallback when no test cases exist for a function."""
    llm = ChatGoogleGenerativeAI(model=MODEL_NAME, google_api_key=GEMINI_API_KEY, temperature=0)

    criteria_text = "\n".join(f"- [{c.kind}] {c.description}" for c in parsed_spec.criteria)

    chain = LLM_JUDGE_PROMPT | llm
    response = chain.invoke({"criteria_list": criteria_text, "function_code": function_source})

    response_text = _extract_text(response.content)
    cleaned = response_text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("```")[1]
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
        cleaned = cleaned.strip()

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        return DeviationCheckResult(
            function_name=function_name, verdict="uncertain", method="llm_judge",
            reason="Judge response was not valid JSON -- treated as uncertain rather than guessed.",
        )

    return DeviationCheckResult(
        function_name=function_name,
        verdict=parsed.get("verdict", "uncertain"),
        method="llm_judge",
        reason=parsed.get("reason", ""),
    )


def check_spec_deviation(
    parsed_spec: ParsedSpec,
    functions: List[FunctionInfo],
    code_text: str,
    test_cases: Optional[List[TestCase]] = None,
) -> List[DeviationCheckResult]:
    """
    Check each REQUESTED function for spec deviation: test execution first
    (ground truth, when available), LLM-judge as fallback (weaker evidence,
    used only when no test cases exist for that function).

    IMPORTANT SCOPING DECISION (fixed after a real bug found by running
    this): only functions that the spec actually asked for are checked
    here. Helper functions and scope-creep functions are deliberately
    SKIPPED, not judged.

    Why: an earlier version judged every function in the file against the
    full spec, which produced technically-true but useless results --
    e.g. flagging a legitimate private helper like `_clean_string` as
    "FAIL: doesn't implement is_palindrome," when of course it doesn't,
    it was never supposed to on its own. That's not a spec deviation, it's
    just not the target function -- a completely different, already-covered
    concern (scope_creep_checker already judges whether a non-requested
    function belongs in the code at all). Spec deviation should only ask
    "does the function that WAS requested behave correctly," not "does
    every function in the file individually satisfy the whole spec."

    A function is considered "requested" if its name is explicitly
    mentioned in the raw spec text -- a simple, explainable heuristic
    (not a guarantee) that intentionally errs toward under-checking rather
    than over-flagging legitimate helpers.
    """
    test_cases = test_cases or []
    results = []

    requested_functions = [
        f for f in functions
        if f.name.lower() in parsed_spec.raw_spec.lower()
    ]

    for f in requested_functions:
        test_result = run_test_execution(f.name, code_text, test_cases)
        if test_result is not None:
            results.append(test_result)
            continue

        # No test cases for this function -- fall back to LLM judgment,
        # extracting just this function's source to keep the judge focused.
        func_source = _extract_function_source(code_text, f.name)
        results.append(run_llm_judge(f.name, func_source, parsed_spec))

    return results


def _extract_function_source(code_text: str, function_name: str) -> str:
    """
    Pull out just one function's source text for a focused LLM-judge prompt,
    rather than sending the whole file (keeps the judge's attention on the
    relevant function, reduces token cost).

    Simple line-based extraction using indentation -- not a full AST-based
    extraction, so it can be fooled by unusual formatting. Good enough for
    this checker's purpose (focus, not perfect isolation); the LLM still
    sees valid Python either way.
    """
    lines = code_text.splitlines()
    start = None
    for i, line in enumerate(lines):
        if line.strip().startswith(f"def {function_name}("):
            start = i
            break
    if start is None:
        return code_text  # fallback: just send everything

    end = len(lines)
    for i in range(start + 1, len(lines)):
        if lines[i].strip() and not lines[i].startswith((" ", "\t")):
            end = i
            break

    return "\n".join(lines[start:end])


if __name__ == "__main__":
    # Quick manual test -- run `python -m agents.spec_deviation_checker`
    from agents.spec_parser import parse_spec
    from agents.code_analyzer import analyze_code

    sample_spec_text = (
        "Write a function `is_even(n: int) -> bool` that returns True if a "
        "number is even, False otherwise."
    )

    # Deliberately WRONG logic: inverted condition
    sample_code = '''
def is_even(n: int) -> bool:
    """Check if a number is even."""
    return n % 2 == 1
'''

    parsed = parse_spec(sample_spec_text)
    analysis = analyze_code(sample_code)

    # Real test cases -- this should be caught by TEST EXECUTION, not LLM judgment
    test_cases = [
        TestCase(function_name="is_even", args=(4,), kwargs={}, expected_output=True),
        TestCase(function_name="is_even", args=(7,), kwargs={}, expected_output=False),
    ]

    results = check_spec_deviation(parsed, analysis.functions, sample_code, test_cases)
    for r in results:
        print(f"[{r.verdict.upper()}] {r.function_name} (via {r.method}) -- {r.reason}")
