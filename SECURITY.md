# Security

Scribe records audio and can type into any window, so security reports are welcome.

## Reporting

Please use GitHub's **private vulnerability reporting** (Security tab → *Report a vulnerability*). Do not open a public issue for a security problem.

## What Scribe does by design

- Speech is transcribed on your own GPU. Audio, transcripts and voiceprints stay on your PC.
- Dictation clean-up and video notes use a local Ollama model at `127.0.0.1`.
- The only thing that leaves the PC is a finished transcript you choose to send to a Claude Code vault. That run is walled: `dontAsk` permission mode, no MCP servers, no connectors, no browser, and an allow list limited to the job folder and the vault. Recorded speech is treated as data, never as instructions.
- The MCP server talks to the running app over a local pipe with a per-install key. It opens no network port.
- Password fields are never read by the edit-learning feature.

## In scope

Prompt injection through recorded speech or screen text, escapes from the Claude vault wall, leaks of audio or transcripts, and anything that types into a window the user did not target.
