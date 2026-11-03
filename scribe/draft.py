"""draft.md in the transcribe skill's house format, so the vault's Claude run corrects a Scribe draft exactly as it
would one the skill wrote. Mirrors render() in ~/.claude/skills/transcribe/transcribe.py: same frontmatter fields,
title line, intro comment and timestamps."""
from __future__ import annotations

import logging
import re
from pathlib import Path

log = logging.getLogger("scribe.draft")

# Verbatim from the transcribe skill: the Claude correction pass replaces this comment with the intro paragraph.
INTRO_NOTE = (
    "<!-- Intro: who was on the call, when, what it was for. Then the standard transcription note: "
    "machine transcription with misheard words and names corrected against the project's own terms, "
    "capitals and punctuation added; fillers and repeats kept as spoken; repetition loops removed and marked; "
    "[unclear] marks unrecovered words, a likely reading followed by [unclear] is a best guess; "
    "timestamps are minutes:seconds from the start; say where speaker labels come from, or that the recording has none. -->"
)

DIARISER = "local diarisation (sherpa-onnx)"
LANGUAGE_NAMES = {"en": "English", "af": "Afrikaans"}
GENERIC_SPEAKER = re.compile(r"^Speaker \d+$")
LIST_MARKER = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+")
MODEL_NAME = re.compile(r"(?:distil-)?(?:large-v\d(?:-turbo)?|medium|small|base|tiny)(?:\.en)?")


def fmt_ts(seconds: float) -> str:
    """The skill's timestamp: mm:ss, or h:mm:ss past an hour."""
    total = int(round(max(0, seconds)))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def _one_line(text) -> str:
    return " ".join(str(text or "").split())


def _visual_text(text) -> str:
    """A keyframe description on one list line: its own lines joined with '; ', list markers dropped."""
    parts = (_one_line(LIST_MARKER.sub("", line)) for line in str(text or "").splitlines())
    return "; ".join(p for p in parts if p)


def _date_from(source: str) -> str:
    m = re.search(r"(20\d{2})-?(\d{2})-?(\d{2})", source)
    return f"{m.group(1)}-{m.group(2)}-{m.group(3)}" if m else "YYYY-MM-DD"


def _title_from(source: str) -> str:
    title = re.sub(r"[-_]?\d{8}_\d{6}[-_]?", " ", Path(source).stem)
    return re.sub(r"\s*(Meeting Recording|Recording)\s*$", "", title, flags=re.I).strip() or "Recording"


def _status_from(integrity: dict) -> str:
    if integrity.get("truncated"):
        declared = integrity.get("declared_seconds")
        return (f"partial — truncated download holding {integrity.get('present_fraction', 0):.0%} of a "
                f"{fmt_ts(declared) if declared else 'longer'} recording")
    return "complete"


def _where(model: str) -> str:
    """Where Whisper ran, from whisper.json's "model" (asr names the device there): GPU, CPU, or GPU then CPU
    when the GPU ran out of memory part-way."""
    text = model.lower()
    if "cpu" not in text:
        return "GPU"
    return "GPU then CPU" if "cuda" in text else "CPU"


def _language_note(languages: dict) -> str:
    """'Afrikaans and English, language detected per chunk (Afrikaans 312 s, English 2841 s)'."""
    named = sorted(((LANGUAGE_NAMES.get(k, k), float(v or 0)) for k, v in languages.items()), key=lambda x: x[0])
    names = [n for n, _ in named]
    head = names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]
    return f"{head}, language detected per chunk ({', '.join(f'{n} {int(round(s))} s' for n, s in named)})"


def render_draft(whisper: dict, lines: list[dict] | None, visuals: list[dict] | None, meta: dict, out: Path) -> Path:
    """Write draft.md and return its path. `out` is the draft's path when it ends in .md, else the job folder.
    `lines` from diarize.assign_words (None: no speaker labels, one line per whisper segment); `visuals` from
    video.describe_frames, interleaved at their times; `meta` carries date, source, title, recording_status
    (optional: owner or owner_name, owner_matched, client). Without an owner in meta, a speaker label that is not
    'Speaker N' is the voiceprint-matched owner."""
    path = Path(out)
    if path.suffix.lower() != ".md":
        path = path / "draft.md"
    source = str(meta.get("source") or whisper.get("source") or "")
    date = str(meta.get("date") or _date_from(source))
    title = str(meta.get("title") or _title_from(source))
    duration = float(whisper.get("duration") or 0)
    integrity = whisper.get("integrity") or {}
    truncated = bool(integrity.get("truncated"))
    status = str(meta.get("recording_status") or _status_from(integrity))
    model = MODEL_NAME.search(str(whisper.get("model") or ""))

    coverage = f"first {fmt_ts(duration)} only" if truncated else "whole recording"
    if whisper.get("languages"):
        coverage += "; " + _language_note(whisper["languages"])

    labelled = bool(lines) and any(_one_line(line.get("speaker")) for line in lines)
    speech: list[tuple[float, str]] = []
    if labelled:
        speakers = [_one_line(line.get("speaker")) for line in lines]
        people = sorted({s for s in speakers if s})
        owner = (meta.get("owner") or meta.get("owner_name")
                 or next((s for s in speakers if s and not GENERIC_SPEAKER.match(s)), None))
        if owner and owner in people and meta.get("owner_matched", True):
            speaker_source = f"{DIARISER}; {owner} matched by voiceprint"
        else:
            speaker_source = f"{DIARISER}; owner not matched"
        coverage += "; speakers from local diarisation, matched word by word on time"
        for line in lines:
            text, name = _one_line(line.get("text")), _one_line(line.get("speaker"))
            if text:
                label = f"**{name}** " if name else ""
                speech.append((float(line["start"]), f"- [{fmt_ts(float(line['start']))}] {label}{text}"))
    else:
        people, speaker_source = [], "none"
        coverage += "; no speaker labels (no local diarisation)"
        for seg in whisper.get("segments") or []:
            text = _one_line(seg.get("text"))
            if text:
                speech.append((float(seg["start"]), f"- [{fmt_ts(float(seg['start']))}] {text}"))

    shown = sorted(
        ((float(v["t"]), f"- [{fmt_ts(float(v['t']))}] *(on screen)* {_visual_text(v.get('text'))}")
         for v in (visuals or []) if _visual_text(v.get("text"))),
        key=lambda x: x[0],
    )
    if shown:
        coverage += f"; on-screen notes from {len(shown)} keyframes"

    # Spoken lines keep their order; each on-screen note goes in before the first line starting at or after it.
    body: list[str] = []
    vi = 0
    for t, row in speech:
        while vi < len(shown) and shown[vi][0] <= t:
            body.append(shown[vi][1])
            vi += 1
        body.append(row)
    body.extend(row for _, row in shown[vi:])

    md = [
        "---",
        f"client: {meta.get('client') or ''}",
        f"people: [{', '.join(people)}]",
        f"date: {date}",
        f"source: {source}",
        f"speaker_source: {speaker_source}",
        f"duration_min: {round(duration / 60, 1)}",
        f"recording_status: {status}",
        f"transcriber: faster-whisper {model.group(0) if model else 'large-v3'} "
        f"(local, {_where(str(whisper.get('model') or ''))}), hand-corrected",
        f"coverage: {coverage}",
        "redactions: none",
        "---",
        "",
        f"# {title} transcript — {date}",
        "",
        INTRO_NOTE,
        "",
        "## Opening",
        "",
        *body,
    ]
    if truncated:
        md.append(f"- [{fmt_ts(duration)}] [file ends here; the rest of the recording is not in the download]")
    md.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(md), encoding="utf-8")
    log.info("draft: %s (%d spoken lines, %d on-screen notes, %s)", path, len(speech), len(shown), speaker_source)
    return path
