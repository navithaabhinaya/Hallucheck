# HalluCheck

**Catching AI code hallucinations before they ship.**

AI coding assistants are powerful but unreliable in specific, dangerous ways:
they invent APIs that don't exist, quietly add unrequested functionality, and
sometimes produce code that runs fine while deviating from what was actually
asked for. Most teams have no systematic way to catch this before it merges.

CodeCheck is a multi-agent verification pipeline that checks AI-generated
Python code against its original spec and flags:

- 🚫 **Scope creep** — unrequested functions/logic added beyond the spec
- 🔗 **Fabricated APIs** — calls to libraries or methods that don't exist
- ⚠️ **Spec deviation** — code that runs but doesn't match stated intent

## Status: 
Core system complete: all three verification checks built and tested,
wired into one pipeline, with a full MLOps evaluation loop and CI
regression gate. See `TROUBLESHOOTING.md` for 17 real issues resolved
during development.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env       # then add your GEMINI_API_KEY
```

Get a free Gemini API key at https://aistudio.google.com/apikey

**Note on model choice**: `config.py` centralizes the model name. Gemini API
models are deprecated on an ongoing schedule -- if you hit a 404 mentioning a
model, check https://ai.google.dev/gemini-api/docs/deprecations and update
`MODEL_NAME` there; nowhere else in the codebase needs to change.

## Try it end to end

```bash
python cli.py --spec data/sample_docs/sample_spec.txt --code data/sample_docs/sample_code.py --test-cases data/sample_docs/sample_test_cases.json
```

Or run the orchestrator directly against its built-in example:

```bash
python -m agents.orchestrator
```

To test individual pieces:

```bash
python -m agents.spec_parser            # spec parsing only
python agents/code_analyzer.py           # AST extraction only (no LLM)
python -m agents.fabricated_api_checker  # fabricated-API check only
python -m agents.scope_creep_checker     # scope-creep check only
python -m agents.spec_deviation_checker  # spec-deviation check only
python -m eval.scorer                    # fabricated-API golden set scoring
python -m eval.scorer_scope_creep        # scope-creep golden set scoring
python -m eval.scorer_spec_deviation     # spec-deviation golden set scoring
python eval/regression_gate.py           # fast (deterministic-only) CI gate
python eval/regression_gate.py --full    # full gate, including LLM-based checks
```

## How it works

1. **Spec Parser** -- breaks the requirement into checkable criteria
2. **Code Analyzer** -- parses generated code via AST, extracts functions,
   imports, and calls (fully deterministic, no LLM)
3. **Fabricated API Checker** -- deterministic check: do imports/calls
   actually exist? Checks both the package level (PyPI/stdlib) and the
   attribute level (does `requests.fetch_url` actually exist on `requests`?)
4. **Scope Creep Checker** -- LLM-judgment check: does each function serve a
   requested criterion, or is it unrequested? Distinguishes legitimate
   private helpers from genuine scope creep.
5. **Spec Deviation Checker** -- does the requested function actually behave
   correctly? Runs real test cases when available (ground truth); falls
   back to LLM-judgment otherwise. Only judges functions the spec actually
   asked for -- helpers and creep functions are correctly left to the
   scope-creep checker.
6. **Orchestrator** -- wires all of the above into one LangGraph pipeline,
   producing a single combined report
7. **MLOps loop** -- golden datasets + scorers for all three checks, logged
   to MLflow, with a CI regression gate that blocks a merge if detection
   accuracy drops below threshold

## Stack

Python, LangChain/LangGraph, Gemini API, `ast` module, MLflow, GitHub Actions

## Why this matters

Research shows LLMs hallucinate package names in roughly 5-22% of code
suggestions depending on model and language -- some of those fake package
names have been registered by attackers (a supply-chain attack technique
called "slopsquatting"). Most teams have no systematic way to catch this,
or the subtler failures (scope creep, spec deviation), before code merges.
CodeCheck is the verification layer for exactly that gap.

## Documentation

See `TROUBLESHOOTING.md` for a detailed log of 17 real issues hit during
development and how each was diagnosed and fixed -- including LLM
non-determinism despite `temperature=0`, a multi-agent scoping bug only
visible when checkers ran together (not in isolation), and an accidental
secret committed to a template file, caught by GitHub's push protection
before it ever went public.
