"""
Scope Creep Checker (Stage 2, Part 2)

Given the spec's parsed criteria (from spec_parser.py) and the actual
functions found in generated code (from code_analyzer.py), asks an LLM to
judge: does each function serve a criterion that was actually requested,
or is it unrequested "scope creep"?

This is deliberately the SECOND verifier check built, after the fabricated
API checker, because it's a fundamentally different kind of check:

- fabricated_api_checker = deterministic (a package either exists or not,
  checkable by a lookup — no judgment call, no LLM needed)
- scope_creep_checker = semantic (a function might be a REASONABLE helper
  that wasn't explicitly named in the spec but clearly serves it, vs. one
  that's genuinely unrequested — this requires real reasoning, not a lookup)

Design decisions worth being able to defend in an interview:

- We give the LLM BOTH the original spec text AND the parsed criteria, not
  just the criteria. Parsing loses nuance (a spec_parser bug or oversimplified
  criterion shouldn't cause a false scope-creep flag on a function that
  clearly relates to something in the original spec but wasn't cleanly
  extracted as a discrete criterion).

- Verdict per function is 3-state, matching the fabricated_api_checker's
  pattern: IN_SCOPE, SCOPE_CREEP, or UNCERTAIN (not a boolean) — for the
  same reason as before: forcing a binary judgment on a genuinely
  ambiguous case (e.g. a helper function that supports a requested feature
  indirectly) produces confidently wrong answers instead of honest doubt.

- We ask for a brief justification per verdict, not just the label. This
  is what makes the tool's output actually useful to a developer deciding
  whether to accept or reject the flag, not just a black-box judgment.

- Uses temperature=0 for consistency, same as spec_parser -- this is a
  judgment task where you want the same input to produce the same output
  reliably, not creative variation.
"""

from dataclasses import dataclass
from typing import List
import json

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

from config import GEMINI_API_KEY, MODEL_NAME, PROMPT_VERSION
from agents.spec_parser import ParsedSpec, _extract_text
from agents.code_analyzer import FunctionInfo


@dataclass
class ScopeCheckResult:
    function_name: str
    verdict: str          # "in_scope" | "scope_creep" | "uncertain"
    justification: str
    prompt_version: str = PROMPT_VERSION


def _normalize_function_name(name: str) -> str:
    """
    Strip any trailing '(...)' the model might include, e.g. turning
    'is_palindrome(s)' into 'is_palindrome'. Observed in practice: the LLM
    sometimes returns the function name WITH its argument list attached,
    despite the prompt asking for just the name -- normalizing here avoids
    a false "this function was never returned" mismatch downstream.
    """
    paren_index = name.find("(")
    return name[:paren_index] if paren_index != -1 else name


SCOPE_CHECK_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a strict but fair code reviewer checking for SCOPE CREEP \
-- functions in generated code that were NOT requested by the original spec.

You will be given:
1. The ORIGINAL spec text (for full context/nuance)
2. The PARSED CRITERIA extracted from that spec (may be imperfect/incomplete)
3. A list of FUNCTIONS actually found in the generated code

For each function, decide:
- "in_scope": the function directly implements a criterion, OR is a clear, \
reasonable helper that supports implementing a requested criterion (e.g. a \
small private helper used only by a requested function is fine)
- "scope_creep": the function does something genuinely unrelated to anything \
requested -- extra features, unrelated utilities, or speculative additions \
nobody asked for
- "uncertain": you genuinely cannot tell from the available information -- \
do NOT guess; use this when evidence is ambiguous

Be fair: implementation necessities (e.g. a helper function, input validation \
that isn't explicitly stated but is clearly implied) should usually be \
"in_scope". Only flag genuinely unrequested additions as "scope_creep".

CRITICAL: You MUST return a verdict for EVERY SINGLE function listed under \
FUNCTIONS FOUND IN CODE below -- including ones that are obviously and \
trivially in_scope. Do not omit a function just because its verdict seems \
"too obvious to mention." If there are 5 functions listed, your results \
array must contain exactly 5 entries, one per function, no exceptions.

Respond ONLY with valid JSON, no markdown fences, no preamble, in this exact shape:
{{"results": [{{"function_name": "...", "verdict": "in_scope", "justification": "..."}}, ...]}}"""),
    ("human", """ORIGINAL SPEC:
{raw_spec}

PARSED CRITERIA:
{criteria_list}

FUNCTIONS FOUND IN CODE:
{function_list}"""),
])


def check_scope_creep(parsed_spec: ParsedSpec, functions: List[FunctionInfo]) -> List[ScopeCheckResult]:
    """
    Judge each function against the spec's criteria to detect scope creep.

    Takes the REAL dataclasses from spec_parser.py and code_analyzer.py --
    this is genuine agent-to-agent wiring, not a hardcoded example.
    """
    if not functions:
        return []

    llm = ChatGoogleGenerativeAI(
        model=MODEL_NAME,
        google_api_key=GEMINI_API_KEY,
        temperature=0,  # deterministic judgment, not creative variation
    )

    criteria_text = "\n".join(f"- [{c.kind}] {c.description}" for c in parsed_spec.criteria)
    functions_text = "\n".join(
        f"- {f.name}({', '.join(f.args)})" + (f" -- docstring: {f.docstring}" if f.docstring else "")
        for f in functions
    )

    chain = SCOPE_CHECK_PROMPT | llm
    response = chain.invoke({
        "raw_spec": parsed_spec.raw_spec,
        "criteria_list": criteria_text,
        "function_list": functions_text,
    })

    try:
        response_text = _extract_text(response.content)
        cleaned = response_text.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("```")[1]
            if cleaned.startswith("json"):
                cleaned = cleaned[4:]
            cleaned = cleaned.strip()
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Scope checker returned non-JSON output, likely needs prompt tuning. "
            f"Raw output: {response_text[:200]}"
        ) from e

    result_objects = [
        ScopeCheckResult(
            function_name=_normalize_function_name(r["function_name"]),
            verdict=r["verdict"],
            justification=r["justification"],
        )
        for r in parsed["results"]
    ]

    # Safeguard: even with an explicit completeness instruction, an LLM can
    # still omit a function from its response. Never let that function
    # silently vanish from the output -- surface it as "uncertain" with a
    # clear, honest reason, so a scorer/caller sees a gap instead of nothing.
    #
    # Names are normalized (parentheses/args stripped) before comparing --
    # observed in practice: the model sometimes returns "is_palindrome(s)"
    # instead of bare "is_palindrome", which would otherwise cause every
    # function to look "missing" even when it was actually returned,
    # producing spurious duplicate UNCERTAIN entries alongside the real ones.
    returned_names = {r.function_name for r in result_objects}
    for f in functions:
        if f.name not in returned_names:
            result_objects.append(
                ScopeCheckResult(
                    function_name=f.name,
                    verdict="uncertain",
                    justification="Checker's LLM response omitted this function -- "
                                   "no verdict was returned, treated as uncertain "
                                   "rather than silently dropped.",
                )
            )

    return result_objects


if __name__ == "__main__":
    # Quick manual test -- run `python agents/scope_creep_checker.py`
    # Wires together REAL spec_parser + code_analyzer output, not hardcoded data
    from agents.spec_parser import parse_spec
    from agents.code_analyzer import analyze_code

    sample_spec_text = (
        "Write a function `is_palindrome(s: str) -> bool` that checks if a "
        "string is a palindrome, ignoring case and spaces."
    )

    sample_code = '''
def is_palindrome(s: str) -> bool:
    """Check if a string is a palindrome."""
    cleaned = _clean_string(s)
    return cleaned == cleaned[::-1]

def _clean_string(s: str) -> str:
    """Helper: lowercase and strip spaces."""
    return s.lower().replace(" ", "")

def send_analytics_event(event_name: str):
    """Send a usage analytics event to a tracking server."""
    print(f"Tracking: {event_name}")
'''

    parsed = parse_spec(sample_spec_text)
    analysis = analyze_code(sample_code)
    results = check_scope_creep(parsed, analysis.functions)

    for r in results:
        print(f"[{r.verdict.upper()}] {r.function_name} -- {r.justification}")
