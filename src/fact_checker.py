"""
fact_checker.py
---------------
Sends a short structured prompt to Groq's free API
to fact-check a claim against extracted sources.

Groq is free — sign up at console.groq.com
Uses Llama 3 model — fast and accurate enough for fact checking.

Returns a structured verdict — REAL, FAKE, or UNVERIFIABLE.
"""

import json
import os
from groq import Groq
from dotenv import load_dotenv
load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────

MODEL      = "llama-3.3-70b-versatile"   # free on Groq, fast, good accuracy
MAX_TOKENS = 200                 # only need JSON back


# ── Fact Check ────────────────────────────────────────────────────────────────

def fact_check(text: str, sources: list[str]) -> dict:
    """
    Call Groq LLM to fact-check a claim.

    Args:
        text    : The claim or article text
        sources : List of org/person names extracted by NER

    Returns:
        {
            "label"      : "REAL", "FAKE", or "UNVERIFIABLE",
            "score"      : float 0-1 (probability of being real),
            "confidence" : float 0-1,
            "reason"     : one sentence explanation
        }
    """
    source_str = ", ".join(sources) if sources else "none found"

    prompt = f"""You are a fact-checker. Analyze the claim below and return ONLY a JSON object. No explanation outside the JSON.

Claim: "{text[:500]}"
Sources or entities mentioned: {source_str}

Return exactly this JSON:
{{
  "verdict"    : "REAL" or "FAKE" or "UNVERIFIABLE",
  "reason"     : "one sentence max",
  "confidence" : integer between 0 and 100
}}"""

    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

    try:
        response = client.chat.completions.create(
            model      = MODEL,
            max_tokens = MAX_TOKENS,
            messages   = [
                {
                    "role"   : "system",
                    "content": "You are a fact-checker. Always respond with valid JSON only. No extra text."
                },
                {
                    "role"   : "user",
                    "content": prompt
                }
            ]
        )

        raw = response.choices[0].message.content.strip()

        # Strip markdown fences if model adds them
        if "```" in raw:
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]

        parsed     = json.loads(raw.strip())
        verdict    = parsed.get("verdict", "UNVERIFIABLE").upper()
        confidence = parsed.get("confidence", 50) / 100.0
        reason     = parsed.get("reason", "Could not determine.")

        # Convert verdict to score (1.0 = real, 0.0 = fake)
        if verdict == "REAL":
            score = confidence
        elif verdict == "FAKE":
            score = round(1.0 - confidence, 4)
        else:
            score = 0.5   # unverifiable = neutral

        return {
            "label"      : verdict,
            "score"      : round(score, 4),
            "confidence" : round(confidence, 4),
            "reason"     : reason
        }

    except json.JSONDecodeError:
        print("[fact_checker] JSON parse failed — marking as UNVERIFIABLE")
        return {
            "label"      : "UNVERIFIABLE",
            "score"      : 0.5,
            "confidence" : 0.0,
            "reason"     : "Could not parse fact-check response."
        }

    except Exception as e:
        print(f"[fact_checker] API error: {e}")
        return {
            "label"      : "UNVERIFIABLE",
            "score"      : 0.5,
            "confidence" : 0.0,
            "reason"     : "Fact-check service unavailable."
        }


# ── Quick test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_cases = [
        {
            "text"    : "AIIMS doctor says lemon juice cures diabetes!!",
            "sources" : ["AIIMS"]
        },
        {
            "text"    : "India GDP grew by 7.2 percent last quarter.",
            "sources" : []
        },
        {
            "text"    : "WHO confirms that 5G towers spread coronavirus.",
            "sources" : ["WHO"]
        },
    ]

    for case in test_cases:
        print(f"Claim   : {case['text']}")
        result = fact_check(case["text"], case["sources"])
        print(f"Verdict : {result['label']}")
        print(f"Score   : {result['score']}")
        print(f"Reason  : {result['reason']}")
        print()