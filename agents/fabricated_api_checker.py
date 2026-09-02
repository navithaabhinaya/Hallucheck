"""
Fabricated API Checker (Stage 2, Part 1 -- extended, bug-fixed)

Checks two things:
1. Import-level: does the imported package actually exist? (stdlib or PyPI)
2. Attribute-level: for a call like `requests.get`, does `get` actually
   exist on the real `requests` module?

Design decisions worth knowing cold for an interview:

- Verdict is a 3-state enum (REAL / FABRICATED / UNVERIFIED), not a boolean.
  Guessing wrong when we genuinely don't know (package not installed,
  network down, rate-limited) is worse than admitting uncertainty.

- BUG FIX (found by actually running this, not just reading it): the PyPI
  check originally treated ANY non-200 response as "package doesn't exist".
  That's wrong -- a 403 (rate limit / blocked network), 500, or other
  transient failure is not proof the package is fake. Only a genuine 404
  means "PyPI confirms this name doesn't exist." Everything else (timeouts,
  403s, 5xx, connection errors) now correctly falls back to UNVERIFIED.

- BUG FIX: attribute resolution now walks MULTI-level attribute chains
  (e.g. `os.path.join`), not just one level. The original version split on
  the first dot only, so `os.path.join` became module="os", attr="path.join",
  which isn't resolvable in one hasattr() call. Fixed by walking each
  dotted segment via getattr() in sequence.

- If an import is already FABRICATED, we don't separately flag every call
  on it -- avoids double-counting one root cause as many separate errors.

- KNOWN LIMITATION (by design, not oversight): this is AST-based and does
  not track variable types across statements. `resp = requests.get(...);
  resp.json()` cannot be validated, since knowing `resp`'s type requires
  flow-sensitive type inference or actually executing the code -- out of
  scope here. This checker resolves dotted attribute chains rooted at an
  imported MODULE name, not at arbitrary variables.
"""

import importlib
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional
import requests

from config import STDLIB_MODULES, PYPI_LOOKUP_TIMEOUT_SECONDS
from agents.code_analyzer import ImportInfo, CallInfo


class Verdict(Enum):
    REAL = "real"
    FABRICATED = "fabricated"
    UNVERIFIED = "unverified"


@dataclass
class ImportCheckResult:
    module: str
    verdict: Verdict
    source: str  # "stdlib", "pypi", "not_found", "unverified"


@dataclass
class CallCheckResult:
    call_name: str
    verdict: Verdict
    reason: str


# ---------------------------------------------------------------------------
# Import-level checking (package existence)
# ---------------------------------------------------------------------------

def check_stdlib(module_name: str) -> bool:
    top_level = module_name.split(".")[0]
    return top_level in STDLIB_MODULES


def check_pypi(module_name: str):
    """
    Returns:
        True  -> confirmed real (200 response)
        False -> confirmed fabricated (genuine 404)
        None  -> could not determine (timeout, network error, 403, 5xx, etc.)
    """
    top_level = module_name.split(".")[0]
    try:
        response = requests.get(
            f"https://pypi.org/pypi/{top_level}/json",
            timeout=PYPI_LOOKUP_TIMEOUT_SECONDS,
        )
        if response.status_code == 200:
            return True
        elif response.status_code == 404:
            return False
        else:
            # 403 (rate-limited/blocked), 5xx, etc. -- NOT proof it's fake
            return None
    except requests.RequestException:
        return None


def check_import(module_name: str) -> ImportCheckResult:
    if check_stdlib(module_name):
        return ImportCheckResult(module=module_name, verdict=Verdict.REAL, source="stdlib")

    pypi_result = check_pypi(module_name)
    if pypi_result is True:
        return ImportCheckResult(module=module_name, verdict=Verdict.REAL, source="pypi")
    elif pypi_result is False:
        return ImportCheckResult(module=module_name, verdict=Verdict.FABRICATED, source="not_found")
    else:
        return ImportCheckResult(module=module_name, verdict=Verdict.UNVERIFIED, source="unverified")


# ---------------------------------------------------------------------------
# Attribute-level checking (does the method/attribute actually exist?)
# ---------------------------------------------------------------------------

def resolve_attribute_chain(module_part: str, attr_chain: str):
    """
    Walk a dotted attribute chain (e.g. 'path.join') starting from an
    imported module, using getattr() at each step.

    Returns (True, "") if fully resolved, (False, reason) otherwise.
    This is what makes `os.path.join` work correctly, not just single-level
    attributes like `requests.get`.
    """
    try:
        current = importlib.import_module(module_part)
    except ImportError:
        return None, f"Module '{module_part}' not installed locally -- cannot introspect"

    segments = attr_chain.split(".")
    for i, segment in enumerate(segments):
        if hasattr(current, segment):
            current = getattr(current, segment)
        else:
            resolved_so_far = ".".join([module_part] + segments[:i])
            return False, f"'{segment}' not found on '{resolved_so_far}' -- likely hallucinated"

    return True, f"Full chain '{module_part}.{attr_chain}' resolved successfully"


def check_call(call_name: str, fabricated_modules: set) -> CallCheckResult:
    if "." not in call_name:
        return CallCheckResult(call_name=call_name, verdict=Verdict.UNVERIFIED,
                                reason="Bare call, not a module attribute -- out of scope for this check")

    module_part, attr_chain = call_name.split(".", 1)

    if module_part in fabricated_modules:
        return CallCheckResult(
            call_name=call_name, verdict=Verdict.UNVERIFIED,
            reason=f"Skipped -- root import '{module_part}' is already fabricated"
        )

    result, reason = resolve_attribute_chain(module_part, attr_chain)

    if result is True:
        return CallCheckResult(call_name=call_name, verdict=Verdict.REAL, reason=reason)
    elif result is False:
        return CallCheckResult(call_name=call_name, verdict=Verdict.FABRICATED, reason=reason)
    else:
        return CallCheckResult(call_name=call_name, verdict=Verdict.UNVERIFIED, reason=reason)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def check_fabricated_apis(imports: List[ImportInfo], calls: List[CallInfo]):
    unique_import_names = list({imp.module for imp in imports})
    import_results = [check_import(m) for m in unique_import_names]

    fabricated_modules = {
        r.module for r in import_results if r.verdict == Verdict.FABRICATED
    }

    unique_call_names = list({c.name for c in calls})
    call_results = [check_call(c, fabricated_modules) for c in unique_call_names]

    return import_results, call_results


if __name__ == "__main__":
    from agents.code_analyzer import analyze_code

    sample_code = '''
import os
import requests
import this_package_definitely_does_not_exist_xyz

def fetch_data(url):
    resp = requests.get(url)
    os.system("echo done")
    os.path.join("a", "b")
    requests.fetch_url(url)  # fabricated method on a real package
    return resp
'''

    analysis = analyze_code(sample_code)
    import_results, call_results = check_fabricated_apis(analysis.imports, analysis.calls)

    print("=== Import checks ===")
    for r in import_results:
        print(f"  [{r.verdict.value.upper()}] {r.module} (source: {r.source})")

    print("\n=== Call checks ===")
    for r in call_results:
        print(f"  [{r.verdict.value.upper()}] {r.call_name} -- {r.reason}")
