# Changelog

## 0.1.1 — 2026-11-11

- **Windows installer:** `ScribeSetup.exe` (per-user, no admin) installs Scribe, fetches Python 3.13 with the bundled uv, PyTorch and the models, and adds a Start menu entry and an uninstaller.
- Runtime data moved from `%LOCALAPPDATA%\Scribe` to `%USERPROFILE%\.scribe`, so an MCP server started by the Claude desktop app sees the same settings, key and voiceprint as the tray app. `SCRIBE_HOME` overrides it.
- `install.ps1 -Uninstall` removes only its own shortcuts, an existing transcribe skill is kept, and the large Ollama vision model is no longer pulled unasked.
- README: an NVIDIA GPU is required (there is no CPU-only mode yet).

## 0.1.0 — 2026-11-03

First public release.

- Push-to-talk dictation (`Ctrl+Win`) typed into any window, with local LLM clean-up.
- Notetaker: records microphone and system sound, transcribes, and labels speakers.
- Voice enrolment, so your name lands on your lines.
- English/Afrikaans code-switching per stretch of speech.
- Learns your corrections from edits you make after a dictation.
- Video files get notes on what was shown on screen (local vision model).
- Optional hand-off of finished transcripts to a Claude Code vault, behind a permission wall.
- MCP server so Claude Code can read recordings and live transcripts.
