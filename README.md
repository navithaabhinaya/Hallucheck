# CodeCheck (HalluCheck)

**Catching AI code hallucinations before they ship.**

AI coding assistants are powerful but unreliable in specific, dangerous ways:
they invent APIs that don't exist, quietly add unrequested functionality, and
sometimes produce code that runs fine while deviating from what was actually
asked for. Most teams have no systematic way to catch this before it merges.

CodeCheck is a verification pipeline that checks AI-generated Python code
against its original spec and flags:

- 🚫 **Scope creep** — unrequested functions/logic added beyond the spec
- 🔗 **Fabricated APIs** — calls to libraries or methods that don't exist
- ⚠️ **Spec deviation** — code that runs but doesn't match stated intent

## Status: 🚧 Stage 1 in progress

- [x] Stage 1 — Spec Parser + Code Analyzer (foundation)
- [ ] Stage 2 — Verifier (scope/API/spec checks)
- [ ] Stage 3 — Golden Set + Scorer
- [ ] Stage 4 — MLOps Loop (MLflow + CI gate)
- [ ] Stage 5 — Polish + demo

## Setup

```bash
python3 -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env       # then add your GEMINI_API_KEY
```

Get a free Gemini API key at https://aistudio.google.com/apikey

## Try Stage 1 right now

```bash
python agents/spec_parser.py
python agents/code_analyzer.py
```

The first parses a sample spec into structured criteria (LLM-powered).
The second parses sample code into functions/imports/calls (pure AST, no LLM —
deterministic extraction doesn't need an LLM's judgment).

## How it works

1. **Spec Parser** — breaks the requirement into checkable criteria
2. **Code Analyzer** — parses generated code via AST, extracts functions,
   imports, and calls
3. **Verifier** *(Stage 2)* — checks scope/API/spec alignment using a mix of
   deterministic checks (AST parsing, PyPI lookups, test execution) and an
   LLM-judge for semantic checks
4. **MLOps loop** *(Stage 4)* — a golden dataset of labeled (spec, code,
   expected flags) cases is scored automatically on every prompt/model
   change, logged to MLflow, with a CI gate that blocks regressions

## Stack

Python, LangChain/LangGraph, Gemini API, `ast` module, MLflow, GitHub Actions

## Why this matters

Research shows LLMs hallucinate package names in roughly 5–22% of code
suggestions depending on model and language — some of those fake package
names have been registered by attackers (a supply-chain attack technique
called "slopsquatting"). Most teams have no systematic way to catch this,
or the subtler failures (scope creep, spec deviation), before code merges.
CodeCheck is the verification layer for exactly that gap.
