"""
Scorer for the Fabricated API Checker (Stage 3)

Runs the REAL checker (agents/fabricated_api_checker.py) against the golden
set, compares against hand-labeled expected verdicts, and computes accuracy
per verdict type.

Why accuracy-per-verdict instead of one blended precision/recall number:
this is a 3-class problem (REAL / FABRICATED / UNVERIFIED), not binary.
Blending them into one precision/recall pair would hide whether the
checker is specifically good/bad at catching FABRICATED cases (the whole
point of the tool) versus just being generally accurate. We report overall
accuracy AND fabricated-specific recall separately, since missing a real
fabrication is the costliest error type -- same reasoning as the
recall-over-accuracy argument for Fracture X.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.code_analyzer import analyze_code
from agents.fabricated_api_checker import check_fabricated_apis, Verdict
from eval.golden_set_fabricated_api import GOLDEN_SET


def run_scorer():
    total_calls = 0
    correct_calls = 0

    fabricated_total = 0       # how many calls are TRULY fabricated in the golden set
    fabricated_caught = 0      # how many of those the checker actually flagged as fabricated

    mismatches = []

    for case in GOLDEN_SET:
        analysis = analyze_code(case["code"])
        _, call_results = check_fabricated_apis(analysis.imports, analysis.calls)

        actual_by_name = {r.call_name: r.verdict.value for r in call_results}

        for call_name, expected_verdict in case["expected_calls"].items():
            total_calls += 1
            actual_verdict = actual_by_name.get(call_name, "MISSING")

            if expected_verdict == "fabricated":
                fabricated_total += 1

            if actual_verdict == expected_verdict:
                correct_calls += 1
                if expected_verdict == "fabricated":
                    fabricated_caught += 1
            else:
                mismatches.append({
                    "case": case["id"],
                    "call": call_name,
                    "expected": expected_verdict,
                    "actual": actual_verdict,
                })

    accuracy = correct_calls / total_calls if total_calls > 0 else 0
    fabricated_recall = fabricated_caught / fabricated_total if fabricated_total > 0 else None

    return {
        "total_calls_checked": total_calls,
        "correct_calls": correct_calls,
        "accuracy": accuracy,
        "fabricated_total": fabricated_total,
        "fabricated_caught": fabricated_caught,
        "fabricated_recall": fabricated_recall,
        "mismatches": mismatches,
    }


if __name__ == "__main__":
    results = run_scorer()

    print(f"Total calls checked: {results['total_calls_checked']}")
    print(f"Correct: {results['correct_calls']}")
    print(f"Accuracy: {results['accuracy']:.2%}")
    print(f"Fabricated-recall: {results['fabricated_recall']:.2%} "
          f"({results['fabricated_caught']}/{results['fabricated_total']} caught)")

    if results["mismatches"]:
        print("\nMismatches:")
        for m in results["mismatches"]:
            print(f"  [{m['case']}] {m['call']}: expected={m['expected']}, actual={m['actual']}")
    else:
        print("\nNo mismatches -- checker matches all golden set expectations.")


def log_to_mlflow():
    """Run the scorer and log real results to MLflow, replacing the earlier dummy test values."""
    from eval.tracker import log_run

    results = run_scorer()
    log_run(
        prompt_version="fabricated_api_v1",
        model_name="n/a_deterministic_check",  # this checker has no LLM -- worth noting explicitly
        metrics={
            "accuracy": results["accuracy"],
            "fabricated_recall": results["fabricated_recall"] or 0.0,
        },
    )
    print("\nLogged real scorer results to MLflow.")
