# 🔍 WhatsApp Fake News Detector

> An AI-powered WhatsApp bot that instantly verifies suspicious forwards — combining ML style detection with LLM fact-checking to give you a verdict in under 6 seconds.

**Minor Project — Department of Computer Science | NIT Patna**

---

## 📌 Problem Statement

Fake news spreads rapidly through WhatsApp in India. Users frequently receive forwards like *"AIIMS doctor confirms miracle cure"* or fabricated government announcements, with no easy way to verify authenticity before sharing.

Most existing tools only detect **writing style** — they flag sensational language but fail to check whether the **actual claim is true or false**. This leads to:

- Rapid spread of health, political, and financial misinformation
- Confusion caused by professionally-written but factually wrong content
- No real-time, accessible verification tool for everyday users

Our system addresses **both dimensions** — how a message is written *and* whether the information is correct.

---

## ✨ Features

- **5-tier verdict system** — FAKE, REAL, LIKELY FAKE, MISLEADING, UNCERTAIN
- **Dual-path pipeline** — short WhatsApp forwards handled differently from long articles
- **LLM fact-checking** via Llama 3.3 70B (Groq) for factual verification
- **ML style detection** trained on 44,000 articles (98.6% accuracy)
- **NER-based source extraction** — detects orgs like AIIMS, WHO, RBI
- **LIME explainability** — highlights suspicious vs credible trigger words
- **Confidence score** with every verdict
- **BERT fallback** — ensures reliability when LLM is unavailable
- **Response time under 6 seconds**
- Works on **political news, health claims, financial content, and conspiracy theories**

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| ML Model (long text) | Logistic Regression / Naive Bayes + TF-IDF |
| ML Model (short text) | Pre-trained BERT (LIAR dataset) |
| LLM Fact-Checker | Llama 3.3 70B via Groq API |
| NER | spaCy / custom NER extractor |
| Explainability | LIME |
| WhatsApp Integration | Twilio Sandbox |
| Local Tunnel | ngrok |
| Language | Python |

---

## 📁 Project Structure

```
├── utils.py            # Data loading and preprocessing
├── train.py            # Feature extraction using TF-IDF
├── train_model.py      # Model training (Logistic Regression, Naive Bayes)
├── ingestion.py        # Input cleaning and length classification
├── liar_model.py       # Pre-trained BERT model for short text
├── fact_checker.py     # LLM-based claim verification (Groq)
├── explainability.py   # LIME-based word highlighting
├── pipeline.py         # Core decision logic and verdict generation
├── main.py             # Server / webhook handler
└── whatsapp_bot.py     # Response formatting for WhatsApp
```

---

## ⚙️ How to Run

### Prerequisites

- Python 3.9+
- A [Groq API key](https://console.groq.com) (free tier)
- A [Twilio account](https://twilio.com) with WhatsApp Sandbox enabled
- ngrok installed

### 1. Clone the repository

```bash
git clone https://github.com/your-username/whatsapp-fake-detector.git
cd whatsapp-fake-detector
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set up environment variables

Create a `.env` file in the root directory:

```env
GROQ_API_KEY=your_groq_api_key
TWILIO_ACCOUNT_SID=your_twilio_sid
TWILIO_AUTH_TOKEN=your_twilio_auth_token
```

### 4. Train the ML model

```bash
python train.py
python train_model.py
```

### 5. Start the server

```bash
python main.py
```

### 6. Expose with ngrok

```bash
ngrok http 5000
```

Copy the ngrok HTTPS URL and paste it as the webhook URL in your Twilio WhatsApp Sandbox settings.

### 7. Test it

Join the Twilio sandbox by sending the join code to the sandbox number, then forward any suspicious WhatsApp message to it.

---

## 📊 Sample Output

```
🔍 Verdict: LIKELY FAKE
📊 Confidence: 84%

❌ Claim: "AIIMS doctors confirm neem cures diabetes permanently"

💬 Reason: No peer-reviewed evidence supports this claim.
           The statement contradicts established medical guidelines.

🏷️ Detected sources: AIIMS
⚠️ Trigger words: "confirms", "permanently", "cure"
```

---

## 🏗️ System Architecture

```
User Input
    │
    ▼
Clean Input ──► Length Check
                    │
          ┌─────────┴──────────┐
        Short               Long
          │                   │
    LLM Fact Check      TF-IDF + ML
    (+ BERT fallback)   (+ LIME explain)
          │                   │
          └─────────┬──────────┘
                    │
              NER Extractor
                    │
              LLM Fact Checker
                    │
            Decision Engine
         (weighted trust logic)
                    │
    ┌───────┬───────┼────────┬───────────┐
  FAKE    REAL  MISLEAD  UNCERTAIN  LIKELY FAKE
                    │
            WhatsApp Bot Reply
         Verdict + Confidence + Reason
```

---

## ⚠️ Known Limitations

1. **Older dataset** — trained on US news; may misclassify some Indian-context domains
2. **LLM knowledge cutoff** — Llama 3.3 knowledge limited to early 2024; very recent events may return UNCERTAIN
3. **English only** — limited multilingual support; Hindi/regional language support is planned

---

## 🚀 Roadmap

- [ ] Hindi and regional language support
- [ ] Real-time news API integration to reduce UNCERTAIN verdicts
- [ ] Image and meme verification
- [ ] Indian news dataset for better contextual accuracy
- [ ] User feedback loop for continuous model improvement
- [ ] Cloud deployment (AWS/GCP) to replace ngrok setup
- [ ] Source credibility scoring (WHO vs unknown blogs)
- [ ] Reduce response time to near real-time

---

## 👥 Team Members

| Name | Role |
|---|---|
| SaiRam Devarasetty | ML Pipeline, Model Training |
| Keerthana Dulam | LLM Integration, API |
| Abhinay Bhargava| WhatsApp Bot, Deployment |

*Department of Computer Science — NIT Patna*

---

## 📄 License

This project was built as an academic minor project at NIT Patna. Please contact the authors before reuse.
