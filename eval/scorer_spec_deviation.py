"""
Scorer for the Spec Deviation Checker (Stage 3, third slice)

Runs the REAL spec_deviation_checker against the golden set and measures
agreement with hand-labeled expected verdicts.

Two things this scorer checks, deliberately different from the other two
scorers:

1. Standard accuracy: does the checker's verdict match the expected one,
   for each REQUESTED function.

2. Scoping boundary check (regression test for entry #17's fix): for
   case_04, the golden set intentionally does NOT include an expected
   verdict for the helper function `_normalize_whitespace`. This scorer
   explicitly asserts that the checker's results contain ONLY the
   requested function and nothing else -- if a future prompt or logic
   change reintroduces the old bug (judging every function instead of
   just requested ones), this scorer will catch it as a scoping violation,
   not just silently ignore the extra unexpected entry.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.spec_parser import parse_spec
from agents.code_analyzer import analyze_code
from agents.spec_deviation_checker import check_spec_deviation
from eval.golden_set_spec_deviation import GOLDEN_SET


def run_scorer():
    total = 0
    correct = 0
    scoping_violations = []  # cases where an unexpected function appeared
    mismatches = []
    method_counts = {"test_execution": 0, "llm_judge": 0}

    for case in GOLDEN_SET:
        parsed_spec = parse_spec(case["spec"])
        analysis = analyze_code(case["code"])
        results = check_spec_deviation(
            parsed_spec, analysis.functions, case["code"], case["test_cases"]
        )

        actual_by_name = {r.function_name: r for r in results}

        # Scoping boundary check: results should contain EXACTLY the
        # expected functions, no more, no less.
        actual_names = set(actual_by_name.keys())
        expected_names = set(case["expected_verdicts"].keys())
        unexpected = actual_names - expected_names
        if unexpected:
            scoping_violations.append({
                "case": case["id"],
                "unexpected_functions": list(unexpected),
            })

        for func_name, expected_verdict in case["expected_verdicts"].items():
            total += 1
            result = actual_by_name.get(func_name)

            if result is None:
                mismatches.append({
                    "case": case["id"], "function": func_name,
                    "expected": expected_verdict, "actual": "MISSING",
                })
                continue

            method_counts[result.method] = method_counts.get(result.method, 0) + 1

            if result.verdict == expected_verdict:
                correct += 1
            else:
                mismatches.append({
                    "case": case["id"], "function": func_name,
                    "expected": expected_verdict, "actual": result.verdict,
                })

    accuracy = correct / total if total > 0 else 0

    return {
        "total_checked": total,
        "correct": correct,
        "accuracy": accuracy,
        "method_counts": method_counts,
        "scoping_violations": scoping_violations,
        "mismatches": mismatches,
    }


def log_to_mlflow():
    """Log real spec-deviation scorer results to MLflow."""
    from eval.tracker import log_run

    results = run_scorer()
    log_run(
        prompt_version="spec_deviation_v1",
        model_name="gemini-3.6-flash",
        metrics={
            "accuracy": results["accuracy"],
            "scoping_violations": len(results["scoping_violations"]),
            "test_execution_checks": results["method_counts"].get("test_execution", 0),
            "llm_judge_checks": results["method_counts"].get("llm_judge", 0),
        },
    )
    print("\nLogged spec-deviation scorer results to MLflow.")


if __name__ == "__main__":
    results = run_scorer()

    print(f"Total checked: {results['total_checked']}")
    print(f"Correct: {results['correct']}")
    print(f"Accuracy: {results['accuracy']:.2%}")
    print(f"Method breakdown: {results['method_counts']}")

    if results["scoping_violations"]:
        print("\nSCOPING VIOLATIONS (unexpected functions in results):")
        for v in results["scoping_violations"]:
            print(f"  [{v['case']}] unexpected: {v['unexpected_functions']}")
    else:
        print("\nNo scoping violations -- checker correctly only judges requested functions.")

    if results["mismatches"]:
        print("\nMismatches:")
        for m in results["mismatches"]:
            print(f"  [{m['case']}] {m['function']}: expected={m['expected']}, actual={m['actual']}")
    else:
        print("\nNo mismatches -- checker matches all golden set expectations.")
