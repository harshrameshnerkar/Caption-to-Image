@echo off
echo ===================================================
echo   CaptionAI — Bidirectional Multimodal Vision Studio
echo ===================================================
echo.
echo Starting server at http://127.0.0.1:5000
echo.

if exist "venv\Scripts\python.exe" (
    venv\Scripts\python.exe app.py
) else (
    python app.py
)

pause
