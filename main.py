"""
main.py  — FIXED VERSION
-------------------------
FastAPI server — entry points:

    GET  /          → health check
    POST /predict   → direct API call (for testing)
    POST /webhook   → Twilio WhatsApp webhook

Fixes applied:
  1. on_startup: auto-fetches ngrok public URL and prints it clearly
     so you know exactly what URL to paste in Twilio console.
  2. Twilio webhook reply: added fallback TwiML response in case
     REST API send fails (ensures 200 always returned to Twilio).
  3. Added /status endpoint so you can check if webhook is live.
  4. Better error messages that tell you exactly what env var is missing.
"""

import os
import sys
import httpx
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, Form
from fastapi.responses import Response, JSONResponse
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from pipeline import run
from twilio.twiml.messaging_response import MessagingResponse
from twilio.rest import Client


ACCOUNT_SID = os.environ.get("TWILIO_ACCOUNT_SID")
AUTH_TOKEN  = os.environ.get("TWILIO_AUTH_TOKEN")
FROM_NUMBER = os.environ.get("TWILIO_WHATSAPP_NUMBER")

# Stored at startup — printed so you can copy-paste into Twilio console
_ngrok_public_url = None


# ── Startup: fetch ngrok URL automatically ────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Runs once when the server starts.
    Fetches the ngrok public URL from the local ngrok API and prints it.
    This is the URL you paste into Twilio console → no more guessing.
    """
    global _ngrok_public_url

    # Check env vars on startup so you know immediately if something is missing
    missing = []
    if not ACCOUNT_SID: missing.append("TWILIO_ACCOUNT_SID")
    if not AUTH_TOKEN:  missing.append("TWILIO_AUTH_TOKEN")
    if not FROM_NUMBER: missing.append("TWILIO_WHATSAPP_NUMBER")
    if missing:
        print(f"\n[startup] ⚠️  Missing .env vars: {', '.join(missing)}")
        print(f"[startup]    Twilio will not work until these are set in .env\n")
    else:
        print(f"[startup] ✅ Twilio credentials loaded.")

    # Try to get ngrok URL (only works if ngrok is running)
    try:
        async with httpx.AsyncClient() as client:
            r = await client.get("http://127.0.0.1:4040/api/tunnels", timeout=3)
            tunnels = r.json().get("tunnels", [])
            https_tunnels = [t for t in tunnels if t["proto"] == "https"]
            if https_tunnels:
                _ngrok_public_url = https_tunnels[0]["public_url"]
                print(f"\n[startup] ✅ ngrok tunnel detected!")
                print(f"[startup]    Public URL : {_ngrok_public_url}")
                print(f"[startup]    Webhook URL: {_ngrok_public_url}/webhook")
                print(f"[startup]    Paste this into Twilio console:")
                print(f"[startup]    https://console.twilio.com → Messaging → Sandbox Settings\n")
            else:
                print(f"[startup] ⚠️  ngrok running but no HTTPS tunnel found.")
    except Exception:
        print(f"[startup] ℹ️  ngrok not detected. Start ngrok with: ngrok http 8000")
        print(f"[startup]    Or double-click start.bat to launch everything together.\n")

    yield  # Server runs here


app = FastAPI(lifespan=lifespan)


# ── Health check ──────────────────────────────────────────────────────────────

@app.get("/")
def health_check():
    return {"status": "running", "service": "Fake News Detector — NIT Patna"}


# ── Status — tells you if webhook is reachable ────────────────────────────────

@app.get("/status")
def status():
    return {
        "fastapi"         : "running",
        "ngrok_url"       : _ngrok_public_url or "not detected",
        "webhook_url"     : f"{_ngrok_public_url}/webhook" if _ngrok_public_url else "not available",
        "twilio_sid_set"  : bool(ACCOUNT_SID),
        "twilio_token_set": bool(AUTH_TOKEN),
        "twilio_from_set" : bool(FROM_NUMBER),
    }


# ── Direct API (for testing without WhatsApp) ─────────────────────────────────

@app.post("/predict")
async def predict(request: Request):
    body = await request.json()
    text = body.get("text", "")
    if not text:
        return JSONResponse({"error": "No text provided"}, status_code=400)
    result = run(text)
    return result


# ── WhatsApp Webhook (Twilio) ─────────────────────────────────────────────────

@app.post("/webhook")
async def webhook(Body: str = Form(...), From: str = Form(...)):
    """
    Twilio sends POST here when user sends a WhatsApp message.

    Strategy:
      1. Try to reply via Twilio REST API (cleaner, no XML in chat).
      2. If REST API fails (e.g. bad credentials), fall back to TwiML response.
      3. Always return HTTP 200 — if we return non-200, Twilio retries endlessly.
    """
    print(f"\n[webhook] ← Message from {From}")
    print(f"[webhook]   Body: {Body[:80]}{'...' if len(Body) > 80 else ''}")

    # Run the fake news pipeline
    try:
        result = run(Body)
        reply  = _format_whatsapp_reply(result)
    except Exception as e:
        print(f"[webhook] Pipeline error: {e}")
        reply = "Sorry, something went wrong processing your message. Please try again."

    print(f"[webhook] → Verdict: {result.get('verdict', 'ERROR')} ({result.get('confidence', '?')})")

    # Strategy 1: Send via Twilio REST API
    rest_ok = False
    if ACCOUNT_SID and AUTH_TOKEN and FROM_NUMBER:
        try:
            client = Client(ACCOUNT_SID, AUTH_TOKEN)
            client.messages.create(
                from_ = FROM_NUMBER,
                to    = From,
                body  = reply
            )
            print("[webhook] ✅ Reply sent via REST API.")
            rest_ok = True
        except Exception as e:
            print(f"[webhook] ⚠️  REST API failed: {e}")
            print(f"[webhook]    Falling back to TwiML response...")
    else:
        print("[webhook] ⚠️  Twilio credentials not set — using TwiML fallback.")

    # Strategy 2 (fallback): Reply via TwiML
    twiml = MessagingResponse()
    if not rest_ok:
        twiml.message(reply)

    # Always return 200 — Twilio will retry if it gets anything else
    return Response(content=str(twiml), media_type="application/xml")


# ── Format reply for WhatsApp ─────────────────────────────────────────────────

def _format_whatsapp_reply(result: dict) -> str:
    if not result.get("success"):
        return f"❌ Could not process your message.\nError: {result.get('error', 'Unknown error')}"

    verdict     = result["verdict"]
    confidence  = result["confidence"]
    risk        = result["risk"]
    explanation = result["explanation"]
    sources     = result.get("sources", [])

    emoji = {
        "FAKE"        : "🔴",
        "REAL"        : "🟢",
        "UNCERTAIN"   : "🟡",
        "LIKELY FAKE" : "🟠",
        "MISLEADING"  : "🟠",
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
        lines += ["", f"*Sources found:* {', '.join(sources)}"]

    if "word_highlights" in result:
        fake_words = [w for w, _ in result["word_highlights"].get("fake_words", [])]
        real_words = [w for w, _ in result["word_highlights"].get("real_words", [])]
        if fake_words:
            lines.append(f"*Suspicious words:* {', '.join(fake_words)}")
        if real_words:
            lines.append(f"*Credible words:* {', '.join(real_words)}")

    lines += ["", "_Fake News Detector — NIT Patna Minor Project_"]

    return "\n".join(lines)


# ── Run server ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)