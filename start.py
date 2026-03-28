"""
QtorchX  —  start.py
Run this file to launch the full stack (backend + frontend).

    python start.py

Opens http://localhost:8888/static/index.html in your browser automatically.
"""

import os
import sys
import subprocess
import threading
import time
import webbrowser

ROOT  = os.path.dirname(os.path.abspath(__file__))
VENV  = os.path.join(ROOT, "venv")
PORT  = 8888
URL   = f"http://localhost:{PORT}/static/index.html"

# Resolve the correct Python executable (venv if available, else current)
if os.path.isdir(VENV):
    PY = os.path.join(VENV, "Scripts", "python.exe") if sys.platform == "win32" \
         else os.path.join(VENV, "bin", "python")
else:
    PY = sys.executable


def open_browser():
    # 2.5 s gives uvicorn time to bind the port before the browser opens.
    time.sleep(2.5)
    webbrowser.open(URL)


if __name__ == "__main__":
    print("=" * 60)
    print("  QtorchX Quantum Simulator")
    print(f"  Backend  : http://localhost:{PORT}")
    print(f"  Frontend : {URL}")
    print("  GPU warmup cycles will run before first request.")
    print("  Press  Ctrl+C  to stop")
    print("=" * 60)

    threading.Thread(target=open_browser, daemon=True).start()

    subprocess.run(
        [PY, os.path.join(ROOT, "entry.py")],
        cwd=ROOT,
    )
