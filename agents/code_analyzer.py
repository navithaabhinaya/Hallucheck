"""
Code Analyzer Agent

Parses AI-generated Python code using the `ast` module — no LLM call needed
here, which is deliberate. Extraction (what functions/imports exist) is a
deterministic fact, not a judgment call, so we don't want an LLM guessing at
it. Save the LLM for Stage 2, where actual judgment (is this scope creep?)
is genuinely required.
"""

import ast
from dataclasses import dataclass, field
from typing import List


@dataclass
class FunctionInfo:
    name: str
    args: List[str]
    line_number: int
    docstring: str | None = None


@dataclass
class ImportInfo:
    module: str          # e.g. "os" or "numpy"
    names: List[str]      # e.g. ["path"] for `from os import path`
    line_number: int


@dataclass
class CallInfo:
    name: str             # e.g. "requests.get" or "print"
    line_number: int


@dataclass
class CodeAnalysis:
    functions: List[FunctionInfo] = field(default_factory=list)
    imports: List[ImportInfo] = field(default_factory=list)
    calls: List[CallInfo] = field(default_factory=list)
    syntax_valid: bool = True
    syntax_error: str | None = None


def analyze_code(code_str: str) -> CodeAnalysis:
    """
    Parse Python source code and extract its structural elements.

    This is intentionally "dumb" (no interpretation of intent) — it just
    tells you WHAT is in the code. The verifier agent in Stage 2 is what
    decides whether what's there is a problem.
    """
    try:
        tree = ast.parse(code_str)
    except SyntaxError as e:
        return CodeAnalysis(syntax_valid=False, syntax_error=str(e))

    analysis = CodeAnalysis()

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            docstring = ast.get_docstring(node)
            args = [a.arg for a in node.args.args]
            analysis.functions.append(
                FunctionInfo(name=node.name, args=args, line_number=node.lineno, docstring=docstring)
            )

        elif isinstance(node, ast.Import):
            for alias in node.names:
                analysis.imports.append(
                    ImportInfo(module=alias.name, names=[], line_number=node.lineno)
                )

        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            names = [alias.name for alias in node.names]
            analysis.imports.append(
                ImportInfo(module=module, names=names, line_number=node.lineno)
            )

        elif isinstance(node, ast.Call):
            call_name = _resolve_call_name(node.func)
            if call_name:
                analysis.calls.append(CallInfo(name=call_name, line_number=node.lineno))

    return analysis


def _resolve_call_name(func_node: ast.expr) -> str | None:
    """Turn a Call node's func into a readable string, e.g. 'requests.get'."""
    if isinstance(func_node, ast.Name):
        return func_node.id
    if isinstance(func_node, ast.Attribute):
        base = _resolve_call_name(func_node.value)
        return f"{base}.{func_node.attr}" if base else func_node.attr
    return None


if __name__ == "__main__":
    # Quick manual test — run `python agents/code_analyzer.py`
    sample_code = '''
import os
from requests import get as http_get

def is_palindrome(s: str) -> bool:
    """Check if a string is a palindrome."""
    cleaned = s.lower().replace(" ", "")
    return cleaned == cleaned[::-1]

def fetch_and_log(url):
    response = http_get(url)
    os.system("echo done")
    return response
'''
    result = analyze_code(sample_code)
    print(f"Functions: {[f.name for f in result.functions]}")
    print(f"Imports: {[(i.module, i.names) for i in result.imports]}")
    print(f"Calls: {[c.name for c in result.calls]}")
