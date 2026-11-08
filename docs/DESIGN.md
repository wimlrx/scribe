# Scribe — design

Scribe is a local dictation and meeting notetaker for Windows. Everything that hears audio runs locally on an NVIDIA GPU; only the final enhancement step uses Claude, inside the vault the user picks.

Three jobs:

1. **Dictation.** Hold `Ctrl+Win`, speak, release. The text is typed where the cursor is.
2. **Notetaker.** One click starts recording the microphone and the computer's audio (calls). It can run for 2+ hours. On stop the user picks a vault; Scribe transcribes locally with speaker labels, then hands the draft to a headless Claude run in that vault, which corrects it against the vault's context, shelves the final transcript, and writes a summary and to-do list.
3. **Drop-in.** A recording or video dropped in `Documents\Scribe\Inbox\<Vault>\` (or sent via Explorer's *Send to*) goes through the same pipeline. For video, Scribe also notes what was shown on screen.

## What Scribe must match (Wispr Flow's meeting notes)

- Meetings of 2h+, in person and on calls. No calendar attendees, no folders.
- Speech may switch language mid-sentence (for example **Afrikaans and English**). The owner is labelled by name and others as `Speaker 2`, `Speaker 3`.
- Notes: one-line summary, `###` topic headings with bullets, then *Next steps* and *Decisions*, written in the meeting's main language.
- Where Claude would read meetings through a connector, Scribe writes Markdown files in the vault instead.

## Pipeline

```
record (mic.flac + system.flac, 16 kHz mono, flushed every 5 s)
  -> mix.flac
  -> local transcription  (faster-whisper large-v3, CUDA fp16, Silero VAD chunks <= 28 s,
                           language picked per chunk from the configured set, e.g. {en, af}, word timestamps)
  -> local speaker labels (sherpa-onnx: pyannote segmentation-3.0 + WeSpeaker ResNet34;
                           the owner's cluster named by their enrolled voiceprint)
  -> [video only] screen notes (ffmpeg scene-change keyframes -> dHash dedupe -> Qwen3.6 27B vision via Ollama)
  -> draft.md in the transcribe skill's house format, speaker-labelled
  -> vault = Local: stop here, draft.md is the deliverable
     vault = any configured vault: headless `claude -p` in that vault runs the transcribe skill's
        "Gather context" -> "Correct the draft" -> "Shelve" steps on the draft, shelving the transcript
        and recording in the folder the vault's CLAUDE.md names for meetings (else Meetings\ at the
        vault root), then writes "<topic> notes <d Mon yyyy>.md" (summary, decisions, to-do) beside them.
```

The wall: the Claude run executes with `cwd` = the chosen vault only. Scribe itself never writes into a vault; only that Claude run does, under the vault's own CLAUDE.md.

## Layout

```
<install folder>\
  README.md  requirements.txt
  docs\DESIGN.md
  skill\transcribe\SKILL.md   the transcribe skill; install copies it to %USERPROFILE%\.claude\skills\transcribe\
  scribe.pyw                  entry point (pythonw, no console)
  install.ps1                 shortcuts: Startup, Desktop+Start "Notetaker", SendTo
  scribe\
    __init__.py
    config.py                 settings + paths
    winkeys.py                low-level keyboard hook: hold Ctrl+Win
    paste.py                  clipboard paste into the focused app
    audio.py                  MicCapture, MeetingRecorder, mix_tracks
    asr.py                    DictationASR, transcribe_file
    diarize.py                diarize, Voiceprint, name_speakers, assign_words
    draft.py                  render_draft
    video.py                  probe, extract_audio, keyframes, describe_frames
    enhance.py                build_prompt, run_claude
    jobs.py                   Job, JobQueue, pipeline
    ui.py                     overlay pills, vault picker, tray icon
    app.py                    single instance, IPC, wiring, inbox watcher
  tests\                      unittest (no pytest installed): python -m unittest discover -s tests
```

Runtime state lives outside the repo:

- `%USERPROFILE%\.scribe\` — `settings.json`, `models\`, `voiceprint.json`, `logs\scribe.log`, `ipc.key`
- `%USERPROFILE%\Documents\Scribe\` — `Recordings\<YYYY-MM-DD HHMM[-n]>\` (one folder per job) and `Inbox\<Vault>\` (one per configured vault, plus `Local`)

## Environment (fixed)

- Windows 11, Python 3.13 venv `.venv` in the install folder, with system site packages.
- From system: faster-whisper 1.2.1, ctranslate2 4.8.1 (CUDA works), numpy 2.4.6, pillow 12.2, torch 2.11+cu128, watchdog 6.0, av 18, requests 2.34, psutil 7.2.
- In the venv: PyAudioWPatch 0.2.12.9, soundfile 0.14.0, soxr 1.1.0, sherpa-onnx 2.0.1, pystray 0.19.5.
- **No other packages may be installed.** No pytest; tests use `unittest`.
- Tools on PATH: `ffmpeg`, `ffprobe`, `ollama` (server at http://127.0.0.1:11434). Claude CLI: `%USERPROFILE%\.local\bin\claude.exe`.
- Models: Whisper `large-v3` and `large-v3-turbo` in the HF cache (`WhisperModel("large-v3")` resolves them); sherpa models in `%USERPROFILE%\.scribe\models\`: `sherpa-onnx-pyannote-segmentation-3-0\model.onnx`, `wespeaker_en_voxceleb_resnet34_LM.onnx`. Ollama vision model `qwen3.6:27b-q4_K_M` (may still be downloading; tests that need it skip with a message if `ollama list` lacks it).

## Rules for every module

- **No windows on the user's screen during build or tests.** Every subprocess uses `creationflags=subprocess.CREATE_NO_WINDOW`. Tests never open Tk windows, tray icons or audio playback devices audibly; GUI code is tested by importing and by pure-function tests only.
- **UI shows labels, counts and symbols only** — no explanatory sentences or tooltips beyond a one-word label. Light, clean type (Segoe UI, normal weight), never heavy bold.
- **Never touch a vault** (any folder configured under `vaults`) in code paths other than `enhance.run_claude`, and never in tests. Tests write only under `tempfile` dirs.
- All ctypes calls declare `argtypes` and `restype` (64-bit handles and pointers: `c_void_p`, `wintypes.HANDLE`, `LPARAM`/`WPARAM` as `c_ssize_t`/`c_size_t`).
- Logging through `logging.getLogger("scribe.<module>")`. No `print` in library code.

## Module contracts

### config.py
```python
APP_DIR: Path            # %USERPROFILE%\.scribe
OUTPUT_ROOT: Path        # from settings, default %USERPROFILE%\Documents\Scribe
def load_settings() -> dict          # merges settings.json over DEFAULTS, writes DEFAULTS on first run
def recordings_dir() -> Path; def inbox_dir(vault: str) -> Path; def models_dir() -> Path
def new_job_folder(stamp: datetime | None = None) -> Path   # Recordings\YYYY-MM-DD HHMM, -2, -3 on clash
DEFAULTS = {
  "owner_name": "",                     # shown as "Me" when empty
  "languages": ["en"],
  "vocabulary": ["Claude"],
  "dictation": {"model": "large-v3-turbo", "strip_fillers": True, "min_seconds": 0.4, "enroll_voice": True},
  "meetings":  {"model": "large-v3", "cluster_threshold": 0.5, "owner_match": 0.45},
  "vaults": {},                         # the user adds their own, e.g. {"Work": "C:\\Vaults\\Work"}
  "default_vault": "Local",
  "picker_timeout_s": 20,
  "claude": {"exe": "%USERPROFILE%\\.local\\bin\\claude.exe", "model": "claude-opus-5-5", "timeout_s": 7200},
  "ollama": {"url": "http://127.0.0.1:11434", "vision_model": "qwen3.6:27b-q4_K_M"},
  "output_root": "%USERPROFILE%\\Documents\\Scribe",
  "test_accept_injected_keys": False,
}
```

### winkeys.py
```python
class HoldHotkey:
    def __init__(self, on_press, on_release, on_cancel, accept_injected: bool = False): ...
    def start(self) -> None   # own thread: SetWindowsHookExW(WH_KEYBOARD_LL) + GetMessageW loop
    def stop(self) -> None    # PostThreadMessageW(WM_QUIT), unhook, join
def modifiers_down() -> bool                 # any of Ctrl/Win/Shift/Alt physically down (GetAsyncKeyState)
def wait_modifiers_released(timeout: float = 2.0) -> bool
```
- `on_press` fires once when a Ctrl key (L/R) and a Win key (L/R) are both down and no other key went down since the first of the two.
- `on_release` fires when either is released after `on_press`.
- `on_cancel` fires (instead of `on_release`) if any other key goes down while held — `Ctrl+Win+D`, `Ctrl+Win+Arrow` etc. keep working; the dictation is discarded.
- Keys are never swallowed (always `CallNextHookEx`). After a press or cancel, before the Win key-up is passed on, inject the mask key `VK 0xE8` down+up with `SendInput` so the Start menu does not open.
- Injected events (`LLKHF_INJECTED`) are ignored unless `accept_injected`.
- Callbacks are handed to a worker thread through a queue; the hook procedure returns in well under 1 ms (Windows silently removes slow LL hooks).
- Keep a module-level reference to the `HOOKPROC` callback so it is not garbage collected.

### paste.py
```python
def paste_text(text: str) -> bool
```
Waits for modifiers released (`winkeys.wait_modifiers_released`), saves the clipboard's `CF_UNICODETEXT` if present, sets `text`, sends `Ctrl+V` with `SendInput`, waits 250 ms, restores the previous text only if the clipboard still holds ours. Retries `OpenClipboard` up to 10× with 20 ms sleeps. If the clipboard held non-text (an image), paste anyway and do not restore. Returns success. Pure ctypes.

### audio.py
```python
class AudioError(RuntimeError): ...
class MicCapture:
    def start(self) -> None           # default input, 16 kHz mono int16 via PyAudioWPatch callback; fast open
    def stop(self) -> np.ndarray      # float32 16 kHz mono
    def level(self) -> float          # RMS 0..1 of the last ~100 ms, for the UI
class MeetingRecorder:
    def __init__(self, folder: Path): ...
    def start(self) -> None
    def stop(self) -> dict            # {"mic": Path, "system": Path, "seconds": float, "events": [...]}
    def elapsed(self) -> float
    def levels(self) -> tuple[float, float]
def mix_tracks(folder: Path) -> Path  # mic.flac + system.flac -> mix.flac (16 kHz mono, peak -1 dBFS); tolerates a missing/empty track
def to_m4a(src: Path, dst: Path) -> Path   # ffmpeg AAC 48 kbps mono, for shelving
```
`MeetingRecorder` records the default microphone and the WASAPI loopback of the default output device, resamples each to 16 kHz mono (`soxr.ResampleStream`), and writes `mic.flac` and `system.flac` incrementally (soundfile, FLAC PCM_16), flushed at least every 5 s; RAM stays bounded for any length. WASAPI loopback delivers nothing during silence, so for the whole recording it plays silence to the default output (an output stream of zeros) and also pads `system.flac` with zeros by wall clock so both tracks stay aligned to within 100 ms. If a stream errors (device unplugged, default device changed) it reopens the current default device, pads the gap with silence, and logs an event. It writes `recording.json` (`{"status": "recording"|"stopped", "started", "last_flush", "events"}`) at every flush so an interrupted recording can be recovered.

### asr.py
```python
class DictationASR:
    def __init__(self, model_name: str, languages: list[str], vocabulary: list[str]): ...
    def load(self) -> None            # WhisperModel(model_name, device="cuda", compute_type="float16"); stays resident
    def transcribe(self, audio: np.ndarray) -> str
def detect_language(model, audio: np.ndarray, allowed: list[str]) -> tuple[str, float]
def transcribe_file(audio_path: Path, model_name: str, languages: list[str], vocabulary: list[str],
                    progress=None) -> dict
```
- `detect_language`: use faster-whisper's language probabilities for the first 30 s of `audio`; return the allowed language with the highest probability (renormalised over the allowed set).
- `DictationASR.transcribe`: energy gate (return `""` below a low RMS threshold) so silence never becomes "Thank you."; language from `detect_language`; `beam_size=5`, `condition_on_previous_text=False`, `without_timestamps=True`, `initial_prompt` from the vocabulary; strip standalone fillers (um, uh, uhm, er, ah, eh, hmm) when `strip_fillers`; collapse whitespace.
- `transcribe_file` returns the transcribe skill's `whisper.json` schema plus extras:
  `{"source", "model", "language", "language_probability", "languages": {"en": secs, "af": secs}, "duration", "integrity": {"truncated": False}, "teams_transcript": None, "segments": [{"start", "end", "text", "language", "words": [{"start", "end", "word"}]}]}`.
  Method: `faster_whisper.decode_audio` (16 kHz mono) -> Silero VAD (`faster_whisper.vad.get_speech_timestamps`, `min_silence_duration_ms=500`) -> group speech regions into chunks of at most 28 s, splitting at the longest pause -> per chunk `detect_language` over `languages` -> `model.transcribe(chunk, language=lang, beam_size=5, word_timestamps=True, condition_on_previous_text=False, vad_filter=False, initial_prompt=<vocabulary + a short mixed Afrikaans/English sample so code-switching is kept as spoken>)` -> offset times by the chunk start. Drop segments with `no_speech_prob > 0.6 and avg_logprob < -1.0`. Collapse repetition loops with the same rule as the transcribe skill (3+ back-to-back repeats of a 3–15 word phrase -> one copy plus `[repetition loop removed, N repeats to mm:ss]`). Free the model at the end (`del`, `gc.collect()`). Call `progress(fraction)` per chunk.

### diarize.py
```python
def diarize(audio_path: Path, models_dir: Path, threshold: float = 0.5, num_speakers: int | None = None) -> list[dict]
    # [{"start", "end", "speaker": int}] via sherpa_onnx.OfflineSpeakerDiarization
class Voiceprint:
    def __init__(self, path: Path, models_dir: Path): ...
    def enroll(self, audio: np.ndarray) -> bool      # >= 2 s of speech; running mean of L2-normalised embeddings
    @property
    def enrolled(self) -> bool
    def similarity(self, embedding: np.ndarray) -> float
    def embed(self, audio: np.ndarray) -> np.ndarray
def name_speakers(audio: np.ndarray, diar: list[dict], voiceprint: Voiceprint | None, owner_name: str,
                  owner_match: float = 0.45) -> dict[int, str]
def assign_words(segments: list[dict], diar: list[dict], names: dict[int, str], gap: float = 3.0) -> list[dict]
    # [{"speaker", "start", "end", "text"}]
```
- `diarize` reads the whole 16 kHz mono file (a 2 h file is ~460 MB float32; acceptable) and runs on CPU.
- `name_speakers`: mean embedding per cluster from up to 8 of its longest segments (>= 1.5 s each); the best cluster with similarity >= `owner_match` becomes `owner_name`; the rest are `Speaker 2`, `Speaker 3`, … in order of first appearance. With no owner match: `Speaker 1`, `Speaker 2`, ….
- `assign_words`: each word goes to the diarisation segment with the greatest time overlap (nearest segment if none overlaps); a new line starts on a speaker change or a pause longer than `gap`.

### draft.py
```python
def render_draft(whisper: dict, lines: list[dict] | None, visuals: list[dict] | None, meta: dict, out: Path) -> Path
```
Writes `draft.md` in the transcribe skill's house format — the same frontmatter fields (`client`, `people`, `date`, `source`, `speaker_source`, `duration_min`, `recording_status`, `transcriber`, `coverage`, `redactions`), the same title line `# <title> transcript — <date>`, the same intro HTML comment, a `## Opening` heading, and lines `- [mm:ss] **Speaker** text`. `speaker_source` reads `local diarisation (sherpa-onnx); <owner> matched by voiceprint` or `... ; owner not matched`. `coverage` notes "Afrikaans and English, language detected per chunk" with the seconds of each. With `visuals`, interleave `- [mm:ss] *(on screen)* <description>` lines at their times. `meta` carries `date`, `source`, `title`, `recording_status`. Timestamps use the skill's `fmt_ts` (`mm:ss`, or `h:mm:ss` past an hour).

### video.py
```python
def probe(path: Path) -> dict                     # {"duration", "has_video", "has_audio"} via ffprobe -of json
def extract_audio(path: Path, out_flac: Path) -> Path    # ffmpeg -vn -ac 1 -ar 16000
def keyframes(path: Path, out_dir: Path, scene: float = 0.3, max_gap: float = 120.0, max_frames: int = 200) -> list[dict]
    # [{"t": seconds, "file": Path}] — scene changes plus at least one frame every max_gap s; width 1280;
    # timestamps from ffmpeg showinfo pts_time; near-duplicates dropped with a numpy dHash (Hamming <= 5 vs last kept)
def describe_frames(frames: list[dict], ollama_url: str, model: str, progress=None) -> list[dict]
    # [{"t", "text"}]; POST /api/chat, images base64, "think": false, options {"num_ctx": 8192, "temperature": 0};
    # asks for titles, key text, numbers, table contents and chart takeaways; frames answered "NONE"
    # (camera feed, people only) are dropped; unloads the model at the end (keep_alive 0).
```

### enhance.py
```python
def build_prompt(job: dict) -> str
def run_claude(job: dict, vault_path: Path, log_path: Path, settings: dict) -> dict
    # {"returncode", "session_id", "transcript", "notes"}
```
Runs `claude.exe -p <prompt> --model <settings model> --permission-mode dontAsk --strict-mcp-config --no-chrome --disallowedTools WebFetch WebSearch --settings <allow list> --add-dir <job folder> --output-format json` with `cwd=vault_path`, `CREATE_NO_WINDOW`, the configured timeout, stdout/stderr to `log_path`, and `ENABLE_CLAUDEAI_MCP_SERVERS=false` (no claude.ai connectors). The allow list holds only reads of this job's folder and the skill's folder, edits inside the vault, and one pinned `cp` of this job's recording; dontAsk refuses everything else, so the job folder stays read-only. Session persistence stays on so the user can resume it to discuss the to-do list; `session_id` comes from the JSON output. Everything the recording contains (speech, screen text, the file's name) is framed in the prompt as untrusted data, never an instruction. The prompt tells Claude:
1. A local Scribe run already did the transcribe skill's *Run whisper* step: `draft.md` and `whisper.json` are at <job folder>; the recording is <audio path>. Speaker labels come from local diarisation, not Teams; the user is `<owner>`; keep the labels but fix clear swaps at turn boundaries from the content.
2. Follow the transcribe skill from *Gather context* through *Hand back*: the vault's CLAUDE.md for its conventions, earlier transcripts in its meetings folder, its recent notes; correct names, terms and misheard words against that context; `[unclear]` where audio and context leave it open; where speech switches language mid-sentence, keep it as spoken, never translated.
3. Shelve the transcript and the `.m4a` recording in the folder the vault's CLAUDE.md names for meetings (choosing the project from the content where it gives each its own); if it names none, in `Meetings\` at the vault root.
4. Write `<topic> notes <d Mon yyyy>.md` beside the transcript: a one-line summary, `###` topic sections with bullets, `### Decisions`, `### To-do` (owner, action, due date if one was said), in the meeting's main language.
5. Send nothing to anyone. End with one line of JSON: `{"transcript": "<vault-relative path>", "notes": "<vault-relative path>"}`.
For a video, add: the `*(on screen)*` lines are machine descriptions of keyframes; use them as context and keep them in the transcript.

### jobs.py
```python
@dataclass
class Job:  id, kind ("meeting"|"file"), source (Path|None), folder (Path), vault (str), status, error, created, updated, outputs (dict)
class JobQueue:
    def __init__(self, settings: dict, notify=None, asr_lock: threading.Lock | None = None): ...
    def start(self) -> None          # worker thread; resumes every non-terminal job found in Recordings\*\job.json
    def submit(self, job: Job) -> None
    def jobs(self) -> list[Job]
def run_pipeline(job: Job, settings: dict, progress=None) -> Job
```
`job.json` in the job folder is the single source of truth; every step is idempotent and skipped when its output exists, so a restart resumes where it stopped. Steps and statuses: `queued` -> (`meeting`: `mix_tracks`; `file`: copy nothing, `extract_audio` from the source in place) -> `transcribing` (`whisper.json`) -> `labelling` (`speakers.json`, using the voiceprint) -> `frames` (video only: `frames\`, `visuals.json`) -> `drafting` (`draft.md`, `recording.m4a`) -> `enhancing` (vault != Local: `claude.log`, `enhance.json`) -> `done` | `failed` (error kept; a failed job can be retried from its folder). A meeting folder left with `recording.json` status `recording` at startup is an interrupted recording: mark it stopped and queue it with the default vault. One job runs at a time (one GPU). The worker frees Whisper before the vision step and before Claude.

### ui.py
```python
class Overlay:   # Tk, owns the main thread
    def __init__(self, root): ...
    def show_listening(self, level_fn) / show_working() / hide() / flash_error()
    def show_recording(self, elapsed_fn, on_stop) / hide_recording()
    def pick_vault(self, vaults: list[str], default: str, timeout_s: int, on_pick) -> None
class Tray:      # pystray, run_detached
    def __init__(self, on_toggle_notetaker, on_add_file, on_open_recordings, on_open_inbox, on_quit): ...
    def set_state(self, state: str, count: int = 0)    # "idle" | "dictating" | "recording" | "working"
    def notify(self, title: str, message: str)
```
- Dictation pill: bottom centre, ~140x36, rounded, dark, a red dot plus 5 level bars while listening, `…` while transcribing. Non-activating, topmost, tool window, click-through (`WS_EX_NOACTIVATE | WS_EX_TOOLWINDOW | WS_EX_TOPMOST | WS_EX_TRANSPARENT | WS_EX_LAYERED`).
- Recording pill: top right, `● 1:23:45` and a `■` stop button. Non-activating, topmost, not click-through.
- Vault picker: replaces the recording pill on stop: one button per vault plus `Local`, default highlighted, auto-picks the default after `timeout_s` with a visible countdown number.
- Tray icon drawn with Pillow: mic glyph; grey idle, green dictating, red recording, amber working (with job count). Left click toggles the notetaker. Menu: `Start notetaker`/`Stop notetaker`, `Add file…`, `Recordings`, `Inbox`, `Quit`.
- All Tk calls happen on the Tk thread; other threads call through a thread-safe queue polled with `root.after`.

### app.py and scribe.pyw
- Single instance via named mutex `Local\ScribeSingleton`. A second launch sends its command over IPC (`multiprocessing.connection` on `127.0.0.1:47391`, authkey from `%USERPROFILE%\.scribe\ipc.key`, created on first run) and exits.
- CLI: `scribe.pyw [--toggle-notetaker] [--add PATH [--vault NAME]] [--quit] [--status]`.
- Wiring: `HoldHotkey` -> `MicCapture` -> `DictationASR` (loaded at startup, resident) -> `paste_text`; after a successful dictation of >= 3 s, `Voiceprint.enroll` on that audio (first 30 clips, then 1 in 10). The notetaker toggles `MeetingRecorder` in a new job folder; stop -> vault picker -> `JobQueue.submit`. Dictation keeps working while a job runs (separate model instance).
- Inbox watcher (`watchdog`) on `Inbox\<Vault>\`: a new media file (`.mp4 .mkv .mov .webm .m4a .mp3 .wav .flac .ogg .aac`) whose size is stable for 5 s becomes a `file` job with that vault. `--add PATH --vault NAME` does the same without moving the file.
- Rotating log at `%USERPROFILE%\.scribe\logs\scribe.log`. Unhandled exceptions in any thread are logged, never shown as dialogs.

### install.ps1
ASCII only. Creates with `WScript.Shell`: Startup `Scribe.lnk` (`.venv\Scripts\pythonw.exe scribe.pyw`), Desktop and Start-menu `Notetaker.lnk` (`... scribe.pyw --toggle-notetaker`, pinnable to the taskbar), SendTo `Scribe - <Vault>.lnk` for each configured vault and `Scribe - transcript only.lnk` (`... scribe.pyw --vault <name> --add`; SendTo appends the file path). Creates `Documents\Scribe\Inbox\<Vault>` for each vault plus `Local`. Copies `skill\transcribe\SKILL.md` to `%USERPROFILE%\.claude\skills\transcribe\SKILL.md`. Generates `scribe.ico` from the tray glyph if missing. Idempotent. `-Uninstall` removes the shortcuts only.
