"""
CLI entry point for HalluCheck.

Usage:
    python cli.py --spec spec.txt --code generated.py
    python cli.py --spec spec.txt --code generated.py --test-cases tests.json
    python cli.py --spec spec.txt --code generated.py --json

This is the single command a user (or a CI pipeline, or an interviewer
watching a demo) runs to get a full HalluCheck report on a (spec, code)
pair -- wraps agents.orchestrator.run_pipeline() with real file I/O,
argument parsing, and both human-readable and machine-readable output.

Design decisions worth explaining if asked:

- Test cases are OPTIONAL and loaded from a separate JSON file, not
  embedded in the spec or code files. This keeps the CLI's core inputs
  (spec, code) minimal and matches how a real user would actually have
  these artifacts -- a spec and generated code are natural, single-purpose
  files; test cases are a separate concern that not every check-run will
  have available, and forcing users to always provide them would make the
  simple case (no tests, LLM-judge fallback for spec-deviation) needlessly
  complicated to invoke.

- --json output mode exists specifically so this CLI is usable inside
  CI: a shell script or GitHub Actions step can parse machine-readable
  output to decide pass/fail, rather than needing to scrape human-readable
  text. This is the same output the eventual CI regression gate will
  consume.

- Exit code reflects whether any FABRICATED or FAIL verdict was found
  (non-zero = problems detected), following the standard CLI convention
  that a non-zero exit code means "something is wrong" -- this is what
  lets a CI step do `python cli.py ... || exit 1` style gating without
  needing to parse output at all for the simple pass/fail case.
"""

import argparse
import json
import sys
from pathlib import Path

from agents.orchestrator import run_pipeline, print_report
from agents.spec_deviation_checker import TestCase


def load_test_cases(path: str) -> list:
    """
    Load test cases from a JSON file, expected shape:
    [
        {"function_name": "is_palindrome", "args": ["racecar"], "kwargs": {}, "expected_output": true},
        ...
    ]
    """
    with open(path) as f:
        raw = json.load(f)

    return [
        TestCase(
            function_name=tc["function_name"],
            args=tuple(tc.get("args", [])),
            kwargs=tc.get("kwargs", {}),
            expected_output=tc["expected_output"],
        )
        for tc in raw
    ]


def state_to_json(state: dict) -> dict:
    """
    Convert the pipeline's final state (dataclasses and enums) into a
    plain JSON-serializable dict, for --json output / CI consumption.
    """
    def result_to_dict(r):
        d = vars(r).copy()
        # Verdict fields may be enums (fabricated_api_checker) or plain
        # strings (scope_creep_checker, spec_deviation_checker) -- handle both.
        if "verdict" in d and hasattr(d["verdict"], "value"):
            d["verdict"] = d["verdict"].value
        return d

    return {
        "criteria_count": len(state["parsed_spec"].criteria) if state.get("parsed_spec") else 0,
        "functions_found": len(state["code_analysis"].functions) if state.get("code_analysis") else 0,
        "fabricated_api_check": {
            "error": state.get("fabricated_api_error"),
            "imports": [result_to_dict(r) for r in (state.get("import_results") or [])],
            "calls": [result_to_dict(r) for r in (state.get("call_results") or [])],
        },
        "scope_creep_check": {
            "error": state.get("scope_creep_error"),
            "results": [result_to_dict(r) for r in (state.get("scope_results") or [])],
        },
        "spec_deviation_check": {
            "error": state.get("spec_deviation_error"),
            "results": [result_to_dict(r) for r in (state.get("deviation_results") or [])],
        },
    }


def has_problems(state: dict) -> bool:
    """
    True if any check found a genuine problem (FABRICATED or FAIL) --
    used to set the CLI's exit code for CI-style pass/fail gating.
    UNVERIFIED/UNCERTAIN are deliberately NOT treated as problems here --
    they mean "couldn't determine," not "found something wrong."
    """
    for r in (state.get("call_results") or []) + (state.get("import_results") or []):
        if hasattr(r.verdict, "value") and r.verdict.value == "fabricated":
            return True
        if r.verdict == "fabricated":
            return True
    for r in (state.get("deviation_results") or []):
        if r.verdict == "fail":
            return True
    return False


def main():
    parser = argparse.ArgumentParser(
        description="HalluCheck: catch AI code hallucinations before they ship."
    )
    parser.add_argument("--spec", required=True, help="Path to a plain-English spec text file")
    parser.add_argument("--code", required=True, help="Path to the AI-generated Python code file")
    parser.add_argument("--test-cases", help="Optional path to a JSON file of test cases")
    parser.add_argument("--json", action="store_true", help="Output machine-readable JSON instead of a human-readable report")

    args = parser.parse_args()

    spec_path = Path(args.spec)
    code_path = Path(args.code)

    if not spec_path.exists():
        print(f"Error: spec file not found: {spec_path}", file=sys.stderr)
        sys.exit(2)
    if not code_path.exists():
        print(f"Error: code file not found: {code_path}", file=sys.stderr)
        sys.exit(2)

    spec_text = spec_path.read_text()
    code_text = code_path.read_text()

    test_cases = load_test_cases(args.test_cases) if args.test_cases else None

    state = run_pipeline(spec_text, code_text, test_cases=test_cases)

    if args.json:
        print(json.dumps(state_to_json(state), indent=2))
    else:
        print_report(state)

    sys.exit(1 if has_problems(state) else 0)


if __name__ == "__main__":
    main()
