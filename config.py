"""
Central config for CodeCheck.
"""

import os
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MODEL_NAME = "gemini-3.6-flash"  # gemini-1.5-flash, gemini-2.0-flash, and gemini-2.5-flash are all now unavailable to new users as of 2026; check https://ai.google.dev/gemini-api/docs/deprecations if this breaks again

PROMPT_VERSION = "v1"

# Timeout for live PyPI lookups (fabricated-API checking) — keep short so a
# slow/unreachable network doesn't stall the whole verification pipeline.
PYPI_LOOKUP_TIMEOUT_SECONDS = 5

STDLIB_MODULES = {
    "os", "sys", "re", "json", "math", "random", "datetime", "time",
    "collections", "itertools", "functools", "typing", "pathlib",
    "subprocess", "logging", "unittest", "argparse", "csv", "sqlite3",
}
