"""Scribe entry point. Run with pythonw (no console):

    .venv\\Scripts\\pythonw.exe scribe.pyw [--toggle-notetaker] [--add PATH ... [--vault NAME]] [--quit] [--status]

A second launch hands its command to the running Scribe and exits.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from scribe import app  # noqa: E402

if __name__ == "__main__":
    sys.exit(app.main())
