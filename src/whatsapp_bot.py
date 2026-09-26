"""
whatsapp_bot.py
---------------
Twilio WhatsApp sandbox setup guide + bot logic.

How it works:
    1. User saves your Twilio number on WhatsApp
    2. User sends "join <sandbox-keyword>" once
    3. User forwards any message to the bot
    4. Twilio POSTs to your /webhook endpoint
    5. main.py processes it and replies

This file handles the message formatting and
can be run standalone to test formatting only.
"""

from twilio.rest import Client
import os
from dotenv import load_dotenv

load_dotenv()

ACCOUNT_SID = os.environ.get("TWILIO_ACCOUNT_SID")
AUTH_TOKEN  = os.environ.get("TWILIO_AUTH_TOKEN")
FROM_NUMBER = os.environ.get("TWILIO_WHATSAPP_NUMBER")  # whatsapp:+14155238886


def send_message(to_number: str, message: str):
    """
    Send a WhatsApp message via Twilio.
    Used for proactive messaging (optional).

    Args:
        to_number : recipient in format whatsapp:+91XXXXXXXXXX
        message   : text to send
    """
    client = Client(ACCOUNT_SID, AUTH_TOKEN)

    msg = client.messages.create(
        from_ = FROM_NUMBER,
        to    = to_number,
        body  = message
    )

    print(f"[whatsapp_bot] Message sent: {msg.sid}")
    return msg.sid


def format_reply(result: dict) -> str:
    """
    Format pipeline result into a clean WhatsApp message.
    No LLM — ML ensemble only. No llm_reason field.
    """
    if not result["success"]:
        return f"Sorry, could not process your message.\nError: {result['error']}"

    verdict     = result["verdict"]
    confidence  = result["confidence"]
    risk        = result["risk"]
    explanation = result["explanation"]
    sources     = result["sources"]

    emoji = {
        "FAKE"        : "🔴",
        "LIKELY FAKE" : "🟠",
        "MISLEADING"  : "🟠",
        "UNCERTAIN"   : "🟡",
        "REAL"        : "🟢",
        "OUT OF SCOPE": "⚪"
    }.get(verdict, "⚪")

    lines = [
        f"{emoji} *VERDICT: {verdict}*",
        f"Confidence: {confidence}",
        f"Risk: {risk}",
        f"",
        f"_{explanation}_",
    ]

    if sources:
        lines.append(f"")
        lines.append(f"*Sources found:* {', '.join(sources)}")

    if "word_highlights" in result:
        fake_words = [w for w, _ in result["word_highlights"]["fake_words"]]
        real_words = [w for w, _ in result["word_highlights"]["real_words"]]
        if fake_words:
            lines.append(f"*Suspicious words:* {', '.join(fake_words)}")
        if real_words:
            lines.append(f"*Credible words:* {', '.join(real_words)}")

    lines.append("")
    lines.append("_Fake News Detector — NIT Patna Minor Project_")

    return "\n".join(lines)


# ── Test formatting standalone ────────────────────────────────────────────────

if __name__ == "__main__":
    dummy_result = {
        "success"     : True,
        "input_type"  : "short_forward",
        "verdict"     : "FAKE",
        "confidence"  : "87%",
        "risk"        : "HIGH",
        "explanation" : "Multiple fake news signals detected. Sensational language confirmed.",
        "sources"     : ["AIIMS"],
        "model_scores": {
            "ml_ensemble": {"label": "FAKE", "score": 0.12}
        }
    }
    print(format_reply(dummy_result))