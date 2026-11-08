# Contributing

Thanks for helping. Scribe is a small Windows app; issues and pull requests are both welcome.

## Set up

```powershell
git clone https://github.com/wimlrx/scribe
cd scribe
powershell -ExecutionPolicy Bypass -File install.ps1
```

## Tests

```powershell
.venv\Scripts\python.exe -m unittest discover -s tests
```

Some tests need an NVIDIA GPU, the Whisper and speaker models, or a running Ollama. They skip themselves when those are missing. CI runs the rest on `windows-latest`.

## Pull requests

- One change per pull request, with a test where it makes sense.
- Keep everything local: no telemetry, no new network calls, no cloud services.
- Run `ruff check .` before you push.
- Describe what you tested on (GPU, Windows version).

## Reporting bugs

Use the bug template. Attach the relevant lines from `%USERPROFILE%\.scribe\logs\scribe.log`. The log holds ids and counts, not transcript text, but check before you paste.
