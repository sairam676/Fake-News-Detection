"""
ingestion.py
------------
First step in the pipeline.
Cleans raw input text and detects whether it is
a short WhatsApp forward or a long pasted article.

No URL fetching — forwards are either short claims
or pasted article text.
"""

import re


SHORT_TEXT_THRESHOLD = 280


def clean_text(text: str) -> str:
    """
    Basic text cleaning.
    - Strip leading/trailing whitespace
    - Collapse multiple spaces and newlines
    - Remove non-printable characters
    """
    text = text.strip()
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text


def ingest(raw_input: str) -> dict:
    """
    Clean and classify raw input text.

    Args:
        raw_input: Raw text from WhatsApp message

    Returns:
        {
            "text"       : cleaned text ready for pipeline,
            "input_type" : "short_forward" or "long_article",
            "success"    : True/False,
            "error"      : error message if failed, else None
        }
    """
    if not raw_input or not raw_input.strip():
        return {
            "text"       : "",
            "input_type" : "unknown",
            "success"    : False,
            "error"      : "Empty input received."
        }

    text = clean_text(raw_input)
    input_type = "short_forward" if len(text) <= SHORT_TEXT_THRESHOLD else "long_article"

    print(f"[ingestion] Type: {input_type} ({len(text)} chars)")

    return {
        "text"       : text,
        "input_type" : input_type,
        "success"    : True,
        "error"      : None
    }


if __name__ == "__main__":
    tests = [
        "AIIMS doctor says lemon juice cures diabetes!! Share fast!!",
        """The Indian government announced a new policy on agricultural subsidies today.
        The minister stated that farmers will receive direct benefit transfers starting
        next quarter. The policy was approved in a cabinet meeting held on Monday and
        is expected to benefit over 10 million farmers across rural India."""
    ]

    for t in tests:
        result = ingest(t)
        print(f"Type    : {result['input_type']}")
        print(f"Length  : {len(result['text'])} chars")
        print(f"Preview : {result['text'][:80]}...")
        print()