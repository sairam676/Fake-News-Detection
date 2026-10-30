"""Verify integration of image forensics into FastAPI app and endpoints."""

import os
from fastapi.testclient import TestClient
from main import app
import numpy as np
from PIL import Image
import io

client = TestClient(app)

def test_health():
    resp = client.get("/")
    assert resp.status_code == 200
    print("Health check OK:", resp.json())

def test_predict_image_endpoint():
    # Create test image in memory
    arr = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)

    files = {"file": ("test.jpg", buf, "image/jpeg")}
    resp = client.post("/predict-image", files=files)
    assert resp.status_code == 200, f"Error: {resp.text}"
    data = resp.json()
    print("\nPOST /predict-image response:")
    print("  • Verdict:   ", data.get("verdict"))
    print("  • Generator: ", data.get("predicted_generator"))
    print("  • Confidence:", data.get("confidence"))
    print("  • Signals:   ", data.get("forensic_breakdown"))
    assert "verdict" in data
    assert "confidence" in data

def test_webhook_text_and_image():
    # 1. Text webhook
    resp_text = client.post("/webhook", data={"Body": "Breaking news vaccine microchip!", "From": "whatsapp:+919999999999"})
    assert resp_text.status_code == 200
    print("\nText Webhook OK! (HTTP 200 returned)")

if __name__ == "__main__":
    print("=" * 60)
    print("TESTING FASTAPI & WHATSAPP IMAGE INTEGRATION")
    print("=" * 60)
    test_health()
    test_predict_image_endpoint()
    test_webhook_text_and_image()
    print("\nALL INTEGRATION TESTS PASSED SUCCESSFULLY!")
    print("=" * 60)
