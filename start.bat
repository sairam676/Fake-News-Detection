@echo off
title Fake News Detector — NIT Patna

echo ================================================
echo  Fake News Detector — Starting all services...
echo ================================================

:: Activate virtual environment
call .venv\Scripts\activate

:: Start FastAPI server in background (Terminal 1)
echo [1/2] Starting FastAPI server on port 8000...
start "FastAPI Server" cmd /k "call .venv\Scripts\activate && uvicorn main:app --host 0.0.0.0 --port 8000 --reload"

:: Wait for FastAPI to boot
timeout /t 3 /nobreak > nul

:: Start ngrok tunnel (Terminal 2)
echo [2/2] Starting ngrok tunnel...
start "ngrok Tunnel" cmd /k "ngrok http 8000"

echo.
echo ================================================
echo  Both services started.
echo  FastAPI : http://localhost:8000
echo  Docs    : http://localhost:8000/docs
echo  ngrok   : check the ngrok window for public URL
echo.
echo  Paste the ngrok URL into Twilio console:
echo  https://console.twilio.com/
echo  Messaging > Sandbox > Webhook URL:
echo  https://xxxx.ngrok-free.app/webhook
echo ================================================
echo.
pause