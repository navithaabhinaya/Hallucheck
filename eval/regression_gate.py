"""
Regression Gate (Stage 4 -- the final MLOps piece)

Runs all three golden-set scorers and checks their results against
minimum acceptable thresholds. Exits non-zero if ANY score falls below
its threshold -- this is what a CI pipeline uses to block a merge.

This is the actual, literal implementation of "CI gate that blocks
regressions in hallucination rate" -- not a metaphor, the real mechanism.

Design decisions worth defending in an interview:

- Thresholds are hardcoded constants in this file, not computed dynamically
  (e.g. "must not drop more than X% from last run"). Why: a fixed minimum
  bar is simpler, more predictable, and avoids a subtle failure mode where
  a SLOW decline across many small commits never individually triggers a
  "regression from last run" check, but the cumulative drift still matters.
  A fixed floor catches that; a delta-only check would not. (A delta check
  is a reasonable enhancement once there's a longer MLflow history to
  compare against -- noted as a real next step, not implemented yet.)

- The deterministic check (fabricated_api) and the LLM-judgment checks
  (scope_creep, spec_deviation) have DIFFERENT thresholds. The deterministic
  check should essentially always be at or near 100% -- any drop is a real
  regression, not noise. The LLM-judgment checks have documented run-to-run
  variance (see TROUBLESHOOTING.md #15), so their thresholds are set lower
  and more conservatively, to avoid the gate flapping (failing/passing
  unpredictably) purely due to LLM sampling variance rather than an actual
  code regression. This distinction is stated explicitly, not just baked
  into a number without explanation.

- A REAL, CONSCIOUS TRADEOFF: this script makes live Gemini API calls
  (scope_creep and spec_deviation scorers), which cost real quota (see
  TROUBLESHOOTING.md -- the free tier is 20 requests/day). Running this
  gate on EVERY push would burn quota fast and could make CI flaky simply
  from hitting rate limits, not from real regressions. The accompanying
  GitHub Actions workflow is deliberately scoped to run the fast,
  deterministic check automatically on every push, and the full LLM-based
  gate only on manual trigger or a scheduled cadence -- a real, disclosed
  engineering tradeoff between thoroughness and cost/quota, not an
  oversight.
"""

import sys

# Thresholds: minimum acceptable score to pass. See module docstring for
# why the deterministic and LLM-judgment checks use different bars.
THRESHOLDS = {
    "fabricated_api_accuracy": 0.95,   # deterministic -- should be ~100%, any real drop matters
    "scope_creep_accuracy": 0.60,      # LLM-judgment -- conservative floor given documented variance
    "spec_deviation_accuracy": 0.75,   # mostly test-execution (ground truth) + some LLM-judge fallback
}


def run_gate(include_llm_checks: bool = True) -> bool:
    """
    Run the regression gate. Returns True if all checks pass their
    thresholds, False if any fail.

    include_llm_checks=False skips the two LLM-dependent scorers entirely
    (scope_creep, spec_deviation) -- used for the fast, quota-free,
    every-push CI path. The fabricated_api check always runs, since it's
    fully deterministic and costs no API quota.
    """
    results = {}
    passed = True

    print("=" * 60)
    print("HALLUCHECK REGRESSION GATE")
    print("=" * 60)

    # Fabricated API check -- always runs, deterministic, no API cost
    from eval.scorer import run_scorer as run_fabricated_api_scorer
    fab_results = run_fabricated_api_scorer()
    fab_score = fab_results["accuracy"]
    results["fabricated_api_accuracy"] = fab_score
    fab_pass = fab_score >= THRESHOLDS["fabricated_api_accuracy"]
    passed = passed and fab_pass
    print(f"\nFabricated API Check: {fab_score:.2%} "
          f"(threshold: {THRESHOLDS['fabricated_api_accuracy']:.2%}) "
          f"-- {'PASS' if fab_pass else 'FAIL'}")

    if include_llm_checks:
        from eval.scorer_scope_creep import run_scorer as run_scope_creep_scorer
        scope_results = run_scope_creep_scorer()
        scope_score = scope_results["accuracy"]
        results["scope_creep_accuracy"] = scope_score
        scope_pass = scope_score >= THRESHOLDS["scope_creep_accuracy"]
        passed = passed and scope_pass
        print(f"Scope Creep Check: {scope_score:.2%} "
              f"(threshold: {THRESHOLDS['scope_creep_accuracy']:.2%}) "
              f"-- {'PASS' if scope_pass else 'FAIL'}")

        from eval.scorer_spec_deviation import run_scorer as run_spec_deviation_scorer
        dev_results = run_spec_deviation_scorer()
        dev_score = dev_results["accuracy"]
        results["spec_deviation_accuracy"] = dev_score
        dev_pass = dev_score >= THRESHOLDS["spec_deviation_accuracy"]
        passed = passed and dev_pass
        print(f"Spec Deviation Check: {dev_score:.2%} "
              f"(threshold: {THRESHOLDS['spec_deviation_accuracy']:.2%}) "
              f"-- {'PASS' if dev_pass else 'FAIL'}")

        if dev_results.get("scoping_violations"):
            print(f"\nWARNING: {len(dev_results['scoping_violations'])} scoping "
                  f"violation(s) detected (see TROUBLESHOOTING.md #17)")
            passed = False
    else:
        print("\n(Skipping LLM-based checks -- fast/quota-free mode. "
              "Run with --full for the complete gate.)")

    print("\n" + "=" * 60)
    print(f"GATE RESULT: {'PASS' if passed else 'FAIL'}")
    print("=" * 60)

    return passed


if __name__ == "__main__":
    fast_mode = "--full" not in sys.argv
    success = run_gate(include_llm_checks=not fast_mode)
    sys.exit(0 if success else 1)
