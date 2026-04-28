"""
main.py
-------
FastAPI server — two entry points:

    GET  /          → health check
    POST /predict   → direct API call (for testing)
    POST /webhook   → Twilio WhatsApp webhook
"""

import os
import sys
from fastapi import FastAPI, Request, Form
from fastapi.responses import Response
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from pipeline import run
from twilio.twiml.messaging_response import MessagingResponse
from twilio.rest import Client


app = FastAPI()

ACCOUNT_SID = os.environ.get("TWILIO_ACCOUNT_SID")
AUTH_TOKEN  = os.environ.get("TWILIO_AUTH_TOKEN")
FROM_NUMBER = os.environ.get("TWILIO_WHATSAPP_NUMBER")


# ── Health Check ──────────────────────────────────────────────────────────────

@app.get("/")
def health_check():
    return {"status": "running", "service": "Fake News Detector"}


# ── Direct API (for testing without WhatsApp) ─────────────────────────────────

@app.post("/predict")
async def predict(request: Request):
    body = await request.json()
    text = body.get("text", "")
    if not text:
        return {"error": "No text provided"}
    result = run(text)
    return result


# ── WhatsApp Webhook (Twilio) ─────────────────────────────────────────────────

@app.post("/webhook")
async def webhook(Body: str = Form(...), From: str = Form(...)):
    """
    Twilio sends POST to this endpoint when user sends WhatsApp message.
    We reply using Twilio REST API directly — avoids XML showing in chat.
    """
    print(f"[webhook] Message from {From}: {Body[:60]}...")

    result = run(Body)
    reply  = format_whatsapp_reply(result)

    # Send reply via Twilio REST API directly
    # This avoids the XML response showing up in WhatsApp
    try:
        client = Client(ACCOUNT_SID, AUTH_TOKEN)
        client.messages.create(
            from_ = FROM_NUMBER,
            to    = From,
            body  = reply
        )
        print("[webhook] Reply sent successfully.")
    except Exception as e:
        print(f"[webhook] Failed to send reply: {e}")

    # Return empty TwiML — Twilio needs a 200 response
    twiml = MessagingResponse()
    return Response(content=str(twiml), media_type="application/xml")


# ── Format Reply For WhatsApp ─────────────────────────────────────────────────

def format_whatsapp_reply(result: dict) -> str:
    if not result["success"]:
        return f"Sorry, could not process your message.\nError: {result['error']}"

    verdict     = result["verdict"]
    confidence  = result["confidence"]
    risk        = result["risk"]
    explanation = result["explanation"]
    llm_reason  = result["llm_reason"]
    sources     = result["sources"]

    emoji = {
        "FAKE"        : "🔴",
        "REAL"        : "🟢",
        "UNCERTAIN"   : "🟡",
        "LIKELY FAKE" : "🟠"
    }.get(verdict, "⚪")

    lines = [
        f"{emoji} *VERDICT: {verdict}*",
        f"Confidence: {confidence}",
        f"Risk: {risk}",
        f"",
        f"_{explanation}_",
        f"",
        f"*Fact-check:* {llm_reason}",
    ]

    if sources:
        lines.append(f"*Sources found:* {', '.join(sources)}")

    if "word_highlights" in result:
        fake_words = [w for w, _ in result["word_highlights"]["fake_words"]]
        real_words = [w for w, _ in result["word_highlights"]["real_words"]]
        if fake_words:
            lines.append(f"*Suspicious words:* {', '.join(fake_words)}")
        if real_words:
            lines.append(f"*Credible words:* {', '.join(real_words)}")

    lines.append(f"")
    lines.append(f"_Fake News Detector — NIT Patna Minor Project_")

    return "\n".join(lines)


# ── Run Server ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)