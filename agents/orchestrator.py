"""
Orchestrator (Stage 2, final piece)

Wires spec_parser -> code_analyzer -> [fabricated_api_checker, scope_creep_checker]
into one unified LangGraph pipeline, producing a single combined verdict
instead of running each script separately by hand.

Design decisions worth defending in an interview:

- Uses LangGraph's StateGraph, not a plain function chain, even though the
  current flow is linear. Why: (1) it makes the pipeline's structure
  explicit and inspectable -- each node is a named step with clear
  input/output, rather than implicit call-order buried in a function body;
  (2) it sets up the project to add branching/looping later (e.g. a
  code-fixer node that loops back after the verifier flags something)
  without restructuring everything; (3) it's the same orchestration
  library used by RuleCraft AI's underlying pattern, so this is a
  deliberate, consistent architectural choice across the portfolio, not
  a different tool per project.

- The two verifier checks (fabricated_api_checker, scope_creep_checker)
  run as SEPARATE nodes, not merged into one step. This matters because
  they are fundamentally different kinds of checks -- deterministic vs.
  LLM-judgment -- and keeping them as distinct nodes means a future
  reader (or interviewer) can see that distinction directly in the graph
  structure, not just in a code comment.

- State is a single TypedDict threaded through every node, matching the
  real dataclasses each agent already produces (ParsedSpec, CodeAnalysis,
  ImportCheckResult/CallCheckResult, ScopeCheckResult) -- not a re-invented
  schema. This is genuine agent-to-agent wiring using the actual existing
  code, not a simplified restatement of it.

- Errors from either verifier node are captured into the state rather than
  allowed to crash the whole graph -- a failure in the scope-creep checker
  (e.g. an API outage) shouldn't prevent the fabricated-API results, which
  are independent, from being reported. Partial results are more useful
  than a total failure.
"""

from typing import TypedDict, List, Optional
from langgraph.graph import StateGraph, END

from agents.spec_parser import parse_spec, ParsedSpec
from agents.code_analyzer import analyze_code, CodeAnalysis
from agents.fabricated_api_checker import check_fabricated_apis, ImportCheckResult, CallCheckResult
from agents.scope_creep_checker import check_scope_creep, ScopeCheckResult
from agents.spec_deviation_checker import check_spec_deviation, DeviationCheckResult, TestCase


class PipelineState(TypedDict):
    spec_text: str
    code_text: str
    test_cases: Optional[List[TestCase]]  # optional -- deviation checker falls back to LLM-judge without these

    parsed_spec: Optional[ParsedSpec]
    code_analysis: Optional[CodeAnalysis]

    import_results: Optional[List[ImportCheckResult]]
    call_results: Optional[List[CallCheckResult]]
    fabricated_api_error: Optional[str]

    scope_results: Optional[List[ScopeCheckResult]]
    scope_creep_error: Optional[str]

    deviation_results: Optional[List[DeviationCheckResult]]
    spec_deviation_error: Optional[str]


def node_parse_spec(state: PipelineState) -> dict:
    """Node 1: parse the plain-English spec into structured criteria."""
    parsed = parse_spec(state["spec_text"])
    return {"parsed_spec": parsed}


def node_analyze_code(state: PipelineState) -> dict:
    """Node 2: extract functions/imports/calls from the generated code (no LLM)."""
    analysis = analyze_code(state["code_text"])
    return {"code_analysis": analysis}


def node_check_fabricated_apis(state: PipelineState) -> dict:
    """Node 3a: deterministic check -- do the imports/calls actually exist?"""
    try:
        import_results, call_results = check_fabricated_apis(
            state["code_analysis"].imports, state["code_analysis"].calls
        )
        return {"import_results": import_results, "call_results": call_results}
    except Exception as e:
        # Don't let one node's failure crash the whole pipeline -- capture
        # the error and let the scope-creep node's results still come through.
        return {"fabricated_api_error": str(e)}


def node_check_scope_creep(state: PipelineState) -> dict:
    """Node 3b: LLM-judgment check -- is anything unrequested scope creep?"""
    try:
        results = check_scope_creep(state["parsed_spec"], state["code_analysis"].functions)
        return {"scope_results": results}
    except Exception as e:
        return {"scope_creep_error": str(e)}


def node_check_spec_deviation(state: PipelineState) -> dict:
    """Node 3c: does the code actually do what was asked? Test execution
    first (ground truth, when test_cases are provided), LLM-judge fallback
    otherwise."""
    try:
        results = check_spec_deviation(
            state["parsed_spec"],
            state["code_analysis"].functions,
            state["code_text"],
            state.get("test_cases") or [],
        )
        return {"deviation_results": results}
    except Exception as e:
        return {"spec_deviation_error": str(e)}


def build_graph():
    """
    Build and compile the pipeline graph.

    Flow: parse_spec -> analyze_code -> [fabricated_api_checker, scope_creep_checker] -> END

    The two checker nodes both depend on analyze_code's output but not on
    each other, so they're modeled as two separate edges from the same
    upstream node rather than a forced sequential order between them --
    accurately reflecting that neither check needs the other's result.
    """
    graph = StateGraph(PipelineState)

    graph.add_node("parse_spec", node_parse_spec)
    graph.add_node("analyze_code", node_analyze_code)
    graph.add_node("check_fabricated_apis", node_check_fabricated_apis)
    graph.add_node("check_scope_creep", node_check_scope_creep)
    graph.add_node("check_spec_deviation", node_check_spec_deviation)

    graph.set_entry_point("parse_spec")
    graph.add_edge("parse_spec", "analyze_code")
    graph.add_edge("analyze_code", "check_fabricated_apis")
    graph.add_edge("analyze_code", "check_scope_creep")
    graph.add_edge("analyze_code", "check_spec_deviation")
    graph.add_edge("check_fabricated_apis", END)
    graph.add_edge("check_scope_creep", END)
    graph.add_edge("check_spec_deviation", END)

    return graph.compile()


def run_pipeline(spec_text: str, code_text: str, test_cases: Optional[List[TestCase]] = None) -> PipelineState:
    """
    Run the full HalluCheck pipeline on a (spec, code) pair and return the
    combined final state -- this is the single entry point cli.py will use.

    test_cases is optional: pass known (input, expected_output) pairs to get
    ground-truth spec-deviation checking via test execution; without them,
    that check falls back entirely to LLM-judgment for every function.
    """
    app = build_graph()
    initial_state: PipelineState = {
        "spec_text": spec_text,
        "code_text": code_text,
        "test_cases": test_cases,
        "parsed_spec": None,
        "code_analysis": None,
        "import_results": None,
        "call_results": None,
        "fabricated_api_error": None,
        "scope_results": None,
        "scope_creep_error": None,
        "deviation_results": None,
        "spec_deviation_error": None,
    }
    final_state = app.invoke(initial_state)
    return final_state


def print_report(state: PipelineState) -> None:
    """Human-readable summary of a pipeline run -- used by the CLI and manual testing."""
    print("=" * 60)
    print("HALLUCHECK REPORT")
    print("=" * 60)

    print(f"\nSpec criteria parsed: {len(state['parsed_spec'].criteria)}")
    print(f"Functions found in code: {len(state['code_analysis'].functions)}")

    print("\n--- Fabricated API Check ---")
    if state.get("fabricated_api_error"):
        print(f"  ERROR: {state['fabricated_api_error']}")
    else:
        for r in state["import_results"]:
            print(f"  [{r.verdict.value.upper()}] import {r.module} (source: {r.source})")
        for r in state["call_results"]:
            print(f"  [{r.verdict.value.upper()}] {r.call_name} -- {r.reason}")

    print("\n--- Scope Creep Check ---")
    if state.get("scope_creep_error"):
        print(f"  ERROR: {state['scope_creep_error']}")
    else:
        for r in state["scope_results"]:
            print(f"  [{r.verdict.upper()}] {r.function_name} -- {r.justification}")

    print("\n--- Spec Deviation Check ---")
    if state.get("spec_deviation_error"):
        print(f"  ERROR: {state['spec_deviation_error']}")
    else:
        for r in state["deviation_results"]:
            print(f"  [{r.verdict.upper()}] {r.function_name} (via {r.method}) -- {r.reason}")

    print("=" * 60)


if __name__ == "__main__":
    # Quick manual test -- run `python -m agents.orchestrator`
    sample_spec = (
        "Write a function `is_palindrome(s: str) -> bool` that checks if a "
        "string is a palindrome, ignoring case and spaces."
    )
    sample_code = '''
import os

def is_palindrome(s: str) -> bool:
    """Check if a string is a palindrome."""
    cleaned = _clean_string(s)
    return cleaned == cleaned[::-1]

def _clean_string(s: str) -> str:
    """Helper: lowercase and strip spaces."""
    return s.lower().replace(" ", "")

def send_analytics_event(event_name: str):
    """Send a usage analytics event."""
    print(f"Tracking: {event_name}")

def fetch_url(url):
    return requests.fetch_url(url)  # requests not even imported -- and fetch_url doesn't exist
'''

    result = run_pipeline(
        sample_spec,
        sample_code,
        test_cases=[
            TestCase(function_name="is_palindrome", args=("racecar",), kwargs={}, expected_output=True),
            TestCase(function_name="is_palindrome", args=("hello",), kwargs={}, expected_output=False),
        ],
    )
    print_report(result)
