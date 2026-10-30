"""
whatsapp_bot.py
---------------
Twilio WhatsApp sandbox setup guide + bot logic.

How it works:
    1. User saves your Twilio number on WhatsApp
    2. User sends "join <sandbox-keyword>" once
    3. User forwards any message or image to the bot
    4. Twilio POSTs to your /webhook endpoint
    5. main.py processes it and replies
"""

from twilio.rest import Client
import os
from dotenv import load_dotenv

load_dotenv()

ACCOUNT_SID = os.environ.get("TWILIO_ACCOUNT_SID")
AUTH_TOKEN  = os.environ.get("TWILIO_AUTH_TOKEN")
FROM_NUMBER = os.environ.get("TWILIO_WHATSAPP_NUMBER")


def send_message(to_number: str, message: str):
    """Send a WhatsApp message via Twilio."""
    client = Client(ACCOUNT_SID, AUTH_TOKEN)
    msg = client.messages.create(
        from_ = FROM_NUMBER,
        to    = to_number,
        body  = message
    )
    print(f"[whatsapp_bot] Message sent: {msg.sid}")
    return msg.sid


def format_reply(result: dict) -> str:
    """Format text pipeline result into a clean WhatsApp message."""
    if not result.get("success"):
        return f"Sorry, could not process your message.\nError: {result.get('error', 'Unknown error')}"

    verdict     = result["verdict"]
    confidence  = result["confidence"]
    risk        = result["risk"]
    explanation = result["explanation"]
    sources     = result.get("sources", [])

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
        lines.append("")
        lines.append(f"*Sources found:* {', '.join(sources)}")

    if "word_highlights" in result:
        fake_words = [w for w, _ in result["word_highlights"].get("fake_words", [])]
        real_words = [w for w, _ in result["word_highlights"].get("real_words", [])]
        if fake_words:
            lines.append(f"*Suspicious words:* {', '.join(fake_words)}")
        if real_words:
            lines.append(f"*Credible words:* {', '.join(real_words)}")

    lines.append("")
    lines.append("_Fake News Detector — NIT Patna Minor Project_")

    return "\n".join(lines)


def format_image_reply(result: dict) -> str:
    """Format image forensics FASC prediction into a clean WhatsApp message."""
    if not result.get("success"):
        return f"❌ Could not process your image.\nError: {result.get('error', 'Unknown error')}"

    verdict     = result.get("verdict", "UNKNOWN")
    generator   = result.get("predicted_generator", "Unknown")
    confidence  = result.get("confidence", "N/A")
    reliability = result.get("reliability_score", confidence)
    decision    = result.get("decision", "ACCEPT")
    risk        = result.get("risk", "UNKNOWN")
    explanation = result.get("explanation", "")
    breakdown   = result.get("forensic_breakdown", {})
    branch_sens = result.get("branch_sensitivities", {})

    emoji = {
        "AI GENERATED": "🔴",
        "REAL PHOTO": "🟢",
        "UNKNOWN / UNCERTAIN GENERATOR": "🟡",
    }.get(verdict, "⚪")

    lines = [
        "🔍 *IMAGE FORENSICS REPORT*",
        "━━━━━━━━━━━━━━━━━━━━━",
        f"{emoji} *VERDICT: {verdict}*",
        f"🤖 *Origin:* {generator}",
        f"🛡️ *Reliability Score (FASC):* {reliability}",
        f"📊 *Raw Confidence:* {confidence}",
        f"⚖️ *Decision:* {decision}",
        f"⚠️ *Risk Level:* {risk}",
        "",
        f"_{explanation}_",
    ]

    if branch_sens:
        lines.append("")
        lines.append("*🔬 Branch Sensitivity Breakdown:*")
        for b_name, s_val in branch_sens.items():
            lines.append(f"  • {b_name}: diff={s_val}")

    if breakdown:
        lines.append("")
        lines.append("*📡 Forensic Signal Indicators:*")
        if "high_freq_energy_score" in breakdown:
            lines.append(f"  • Spectral Energy: {breakdown['high_freq_energy_score']}")
        if "low_bit_noise_density" in breakdown:
            lines.append(f"  • Low-Bit Noise Density: {breakdown['low_bit_noise_density']}")
        if "style_texture_variance" in breakdown:
            lines.append(f"  • Style Texture Variance: {breakdown['style_texture_variance']}")

    lines.extend([
        "━━━━━━━━━━━━━━━━━━━━━",
        "_Fake News Detector — NIT Patna Minor Project_"
    ])
    return "\n".join(lines)
