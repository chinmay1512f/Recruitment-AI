import os
import subprocess
import sys

port = os.environ.get("PORT", "8000")
cmd = ["gunicorn", "app:app", "--bind", f"0.0.0.0:{port}", "--workers", "1"]
sys.exit(subprocess.call(cmd))
