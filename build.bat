@echo off
setlocal
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install pyinstaller
python -m PyInstaller --noconfirm --clean --windowed --name KyZerAI app.py
echo.
echo Build complete: dist\KyZerAI\KyZerAI.exe
echo The AI model is downloaded separately on first generation (~9 MB).
pause
