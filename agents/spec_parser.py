"""
Spec Parser Agent

Takes a plain-English requirement/spec and breaks it into a structured,
checkable list of criteria: what functions/behavior are actually asked for.

This is the "ground truth" the verifier will later compare generated code
against -- so the quality of this parsing directly determines how good your
scope-creep and spec-deviation checks can be.
"""

from dataclasses import dataclass, field
from typing import List
import json

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

from config import GEMINI_API_KEY, MODEL_NAME, PROMPT_VERSION


@dataclass
class SpecCriterion:
    """A single, atomic requirement extracted from the spec."""
    id: str
    description: str
    kind: str  # "function" | "behavior" | "constraint"


@dataclass
class ParsedSpec:
    raw_spec: str
    criteria: List[SpecCriterion] = field(default_factory=list)
    prompt_version: str = PROMPT_VERSION


SPEC_PARSE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a precise requirements analyst. Given a plain-English \
coding spec, extract the ATOMIC list of things being asked for. Do not infer \
anything beyond what's explicitly stated or clearly implied.

For each item, classify it as one of:
- "function": a specific function/method that should exist
- "behavior": something the code should do (e.g. "handle empty input")
- "constraint": a limit or rule (e.g. "must not use external libraries")

Respond ONLY with valid JSON, no markdown fences, no preamble, in this exact shape:
{{"criteria": [{{"id": "c1", "description": "...", "kind": "function"}}, ...]}}"""),
    ("human", "{spec}"),
])


def _extract_text(response_content) -> str:
    """
    Normalize an LLM response's .content into a plain string.

    Newer versions of langchain-google-genai (and other providers) sometimes
    return .content as a LIST of content blocks (e.g. [{"type": "text",
    "text": "..."}]) instead of a plain string -- this changed under us
    between SDK versions and broke a plain json.loads(response.content) call.

    Rather than assume one shape, handle both explicitly: if it's already a
    string, use it as-is; if it's a list, concatenate the text from any
    text-type blocks. This is defensive, not speculative -- it's exactly the
    shape mismatch that caused the TypeError in practice.
    """
    if isinstance(response_content, str):
        return response_content

    if isinstance(response_content, list):
        parts = []
        for block in response_content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and "text" in block:
                parts.append(block["text"])
        return "".join(parts)

    # Unexpected shape -- fail loudly with a clear message rather than a
    # confusing downstream error, so this is easy to diagnose if the SDK
    # changes shape again in the future.
    raise TypeError(
        f"Unexpected response.content type: {type(response_content)}. "
        f"Expected str or list. Value: {response_content!r}"
    )


def parse_spec(spec_text: str) -> ParsedSpec:
    """
    Parse a plain-English spec into structured criteria.

    Example:
        >>> result = parse_spec("Write a function that reverses a string.")
        >>> result.criteria[0].description
        'A function that reverses a string'
    """
    llm = ChatGoogleGenerativeAI(
        model=MODEL_NAME,
        google_api_key=GEMINI_API_KEY,
        temperature=0,  # deterministic -- consistent parsing, not creativity
    )

    chain = SPEC_PARSE_PROMPT | llm
    response = chain.invoke({"spec": spec_text})

    response_text = _extract_text(response.content)

    # Some models wrap JSON in markdown fences despite instructions -- strip
    # defensively rather than assuming perfect instruction-following.
    cleaned = response_text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("```")[1]
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
        cleaned = cleaned.strip()

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Spec parser returned non-JSON output, likely needs prompt tuning. "
            f"Raw output: {response_text[:200]}"
        ) from e

    criteria = [
        SpecCriterion(id=c["id"], description=c["description"], kind=c["kind"])
        for c in parsed["criteria"]
    ]
    return ParsedSpec(raw_spec=spec_text, criteria=criteria)


if __name__ == "__main__":
    # Quick manual test -- run `python -m agents.spec_parser`
    sample_spec = (
        "Write a function `is_palindrome(s: str) -> bool` that checks if a "
        "string is a palindrome, ignoring case and spaces. Do not use any "
        "external libraries."
    )
    result = parse_spec(sample_spec)
    print(f"Parsed {len(result.criteria)} criteria:")
    for c in result.criteria:
        print(f"  [{c.kind}] {c.description}")
