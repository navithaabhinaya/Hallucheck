"""
Scorer for the Scope Creep Checker (Stage 3, second slice)

Runs the REAL scope_creep_checker against the golden set and measures
agreement with hand-labeled expected verdicts.

Important framing difference from the fabricated-API scorer: this checker
is LLM-judgment-based, not deterministic. Even at temperature=0, don't
expect literally 100% agreement across model versions/providers the way we
saw with the fully-deterministic fabricated-API check. The metric here is
"how well does the judge agree with a reasonable human reviewer" -- which
is genuinely closer to how real LLM-as-judge evaluation is scored in
practice (see: the MLflow GenAI guide's Correctness/RelevanceToQuery
built-in judges -- same underlying idea).

We report:
- Overall agreement (accuracy vs hand-labeled expected verdicts)
- Scope-creep-specific RECALL: of the truly-scope-creep functions in the
  golden set, how many did the checker actually catch? This is the metric
  that matters most for the tool's actual purpose -- missing real scope
  creep is a worse failure than being slightly too generous on an edge case.
- False-flag rate on legitimate helpers (case_03 specifically) -- since a
  checker that's too aggressive and flags every helper function is not
  useful in practice, even if it never misses real scope creep.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.spec_parser import parse_spec
from agents.code_analyzer import analyze_code
from agents.scope_creep_checker import check_scope_creep
from eval.golden_set_scope_creep import GOLDEN_SET


def run_scorer():
    total = 0
    correct = 0

    creep_total = 0
    creep_caught = 0

    false_flags_on_helpers = 0  # legitimate in_scope helpers wrongly flagged

    mismatches = []

    for case in GOLDEN_SET:
        parsed_spec = parse_spec(case["spec"])
        analysis = analyze_code(case["code"])
        results = check_scope_creep(parsed_spec, analysis.functions)

        actual_by_name = {r.function_name: r.verdict for r in results}

        for func_name, expected_verdict in case["expected_verdicts"].items():
            total += 1
            actual_verdict = actual_by_name.get(func_name, "MISSING")

            if expected_verdict == "scope_creep":
                creep_total += 1

            if actual_verdict == expected_verdict:
                correct += 1
                if expected_verdict == "scope_creep":
                    creep_caught += 1
            else:
                mismatches.append({
                    "case": case["id"],
                    "function": func_name,
                    "expected": expected_verdict,
                    "actual": actual_verdict,
                })
                # A legitimate in_scope function wrongly flagged as scope_creep
                # is specifically the "too aggressive, flags every helper" failure
                if expected_verdict == "in_scope" and actual_verdict == "scope_creep":
                    false_flags_on_helpers += 1

    accuracy = correct / total if total > 0 else 0
    creep_recall = creep_caught / creep_total if creep_total > 0 else None

    return {
        "total_checked": total,
        "correct": correct,
        "accuracy": accuracy,
        "creep_total": creep_total,
        "creep_caught": creep_caught,
        "creep_recall": creep_recall,
        "false_flags_on_helpers": false_flags_on_helpers,
        "mismatches": mismatches,
    }


def log_to_mlflow():
    """Log real scope-creep scorer results to MLflow."""
    from eval.tracker import log_run

    results = run_scorer()
    log_run(
        prompt_version="scope_creep_v1",
        model_name="gemini-3.6-flash",
        metrics={
            "accuracy": results["accuracy"],
            "creep_recall": results["creep_recall"] or 0.0,
            "false_flags_on_helpers": results["false_flags_on_helpers"],
        },
    )
    print("\nLogged scope-creep scorer results to MLflow.")


if __name__ == "__main__":
    results = run_scorer()

    print(f"Total checked: {results['total_checked']}")
    print(f"Correct: {results['correct']}")
    print(f"Accuracy: {results['accuracy']:.2%}")
    print(f"Scope-creep recall: {results['creep_recall']:.2%} "
          f"({results['creep_caught']}/{results['creep_total']} caught)")
    print(f"False flags on legitimate helpers: {results['false_flags_on_helpers']}")

    if results["mismatches"]:
        print("\nMismatches:")
        for m in results["mismatches"]:
            print(f"  [{m['case']}] {m['function']}: expected={m['expected']}, actual={m['actual']}")
    else:
        print("\nNo mismatches -- checker matches all golden set expectations.")
