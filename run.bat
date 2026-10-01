@echo off
cd /d "%~dp0"
echo Starting RAG Document QA System...
call .venv\Scripts\activate.bat
streamlit run app.py
pause
