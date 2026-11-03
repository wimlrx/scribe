# Benchmark — dictation clean-up model

**gemma4:e4b is the clean-up model: 47 of 47 cases pass, and a 40-word dictation is cleaned in a median 332 ms warm.** Measured 31 Oct 2026 on the RTX 5090 Laptop (24 GB), Ollama 0.40.13, with Scribe's Whisper resident (about 2.4 GB). The candidate table below is from the first run, on the first 40 cases.

**Method.** Each candidate ran the 40 cases in `tests/fixtures/cleanup_cases.json` through `cleanup.Cleaner`, including its code guards and spelling pass. A case passes only when the pasted text meets every check. Where Scribe may fall back to Whisper's text (requests and questions), the fallback also passes. The cases are in two sets. The tuning set has 25 cases and the prompt was written against it. The held-out set has 15 cases, written before any model saw them. Warm latency is the median of 5 runs on the 40-word `latency_input`, varying one name per run, after a warm-up. Every request used `think: false`, temperature 0, `num_ctx` 8192 and a bounded `num_predict`.

| Model (exact tag) | Pass | Held-out | Warm median, 40 words | 415 words | VRAM |
|---|---|---|---|---|---|
| **gemma4:e4b** | **40/40** | **15/15** | **345 ms** (334–385) | 3.3–3.5 s | ~5.1 GB |
| qwen3:4b-instruct-2507-q4_K_M | 37/40 | 12/15 | 654 ms | 7.1–7.3 s | — |
| qwen3.5:4b | 37/40 | 13/15 | 998 ms | 7.5–7.6 s | — |
| qwen3.5:9b | 38/40 | 14/15 | 1,772 ms | 13.1–14.0 s | — |

**Where the others failed.**
- Both 4B Qwen models translate code-switched speech. In an Afrikaans-led sentence they turn English into Afrikaans: "and he's happy with" became "en hy is happy met". The translation guard rejects these outputs, so Whisper's text is pasted.
- qwen3:4b-instruct missed Afrikaans "wag" as a self-correction and did not bullet an Afrikaans "die volgende" list.
- qwen3.5:4b changed "before lunch" to "before 12:00". It also invented a three-point list after "Here is what I think…", which the length guard caught.
- qwen3.5:9b wrote "nommers" for "numbers" and was over the ~1.2 s budget.

**Caveats.** The three Qwen models were measured one prompt revision earlier. That revision lacked one sentence: "Add no unit or currency they did not say." The sentence was added after gemma4:e4b wrote "R2.3 billion" for an unstated currency in a held-out case. That held-out case is therefore no longer fully blind. The three Qwen models were removed afterwards, as the brief asked, so they were not re-run. The brief's tag `qwen3:4b-instruct-2507` does not exist on ollama.com, so its q4_K_M build was measured. gemma4:e4b is 6.6 GB on disk because it ships vision and audio parts that Scribe does not use.

**Re-run after the review fixes (31 Oct 2026).** The review added stricter guards: figures, term, preface, answers. It also added a tighter sound-alike respell, a safe-pairs filter, and 7 cases (`"set": "review"`). The system prompt is unchanged. gemma4:e4b passes all 47 cases, and the warm median on 40 words is 332 ms (326–372 ms). Two review cases pass only because the guard falls back to Whisper's text:
- "we ran the project planning…": the model writes "Project Planner", and the term guard rejects it.
- "ek dink ons moet die client vandag bel scratch that I will call him tomorrow": the model returns "Ek dink ons moet die client môre bel.", and the translation guard rejects it.

A prompt sentence telling the model that a term needs all its words did not stop the first. It was not kept. The reviewer's other live failures now come out right: "working capital is tight", "with the notes", "markup", "1234" twice, and the three dictations that open with a filler.

**Cold start.** A clean-up on an unloaded model hits the 2.5 s timeout and pastes Whisper's text. It then loads the model in the background. A load takes 1.4–5.5 s from the disk cache, and 17 s on the first read after the pull. If `warm()` runs on the Ctrl+Win press, the load overlaps the speech: after 3 s of speech, the first clean-up took 2.0 s and was used. When the 17 GB vision model loads for a video job, Ollama evicts gemma4:e4b, and the next press loads it again.

**Timeout.** `timeout_s` 2.5 covers dictations up to 40 words. Each extra word adds `timeout_per_word_s` 0.01 s, so a 415-word dictation gets 6.25 s against the measured 3.5 s.

The live tests re-check all of this: `set SCRIBE_LIVE_CLEANUP=1` then `.venv\Scripts\python.exe -m unittest tests.test_cleanup`. They assert all 47 cases pass and the warm median stays under 1.2 s.
