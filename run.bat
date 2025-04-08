@echo off
echo Starting Multi-Agent Research Chat...

REM Check if venv exists, if not create it
if not exist venv (
    echo Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
call venv\Scripts\activate

REM Install requirements if needed
echo Installing requirements...
pip install -r requirements.txt

REM Run the app
echo Starting Streamlit app...
python gradio_chat.py

REM Deactivate virtual environment
call venv\Scripts\deactivate 