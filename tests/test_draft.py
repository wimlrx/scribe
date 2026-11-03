"""draft: render_draft output parsed back and checked against the transcribe skill's own render().
Writes only under tempfile dirs; the skill's transcribe.py is imported read-only when present."""
from __future__ import annotations

import importlib.util
import json
import re
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scribe import draft  # noqa: E402

SKILL = Path.home() / ".claude" / "skills" / "transcribe" / "transcribe.py"
SKILL_KEYS = ["client", "people", "date", "source", "speaker_source", "duration_min", "recording_status",
              "transcriber", "coverage", "redactions"]
LINE = re.compile(r"^- \[(?P<ts>\d{2}:\d{2}|\d+:\d{2}:\d{2})\] (?:\*\*(?P<name>[^*]+)\*\* )?(?P<text>\S.*)$")
OWNER = "Alex Morgan"


def load_skill():
    if not SKILL.is_file():
        return None
    spec = importlib.util.spec_from_file_location("transcribe_skill", SKILL)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def parse(md: str) -> tuple[list[tuple[str, str]], list[str]]:
    """(frontmatter key/value pairs in order, body lines)."""
    lines = md.split("\n")
    assert lines[0] == "---", lines[0]
    end = lines.index("---", 1)
    front = [tuple(l.split(": ", 1)) if ": " in l else (l.rstrip(":"), "") for l in lines[1:end]]
    return front, lines[end + 1:]


def whisper(duration: float = 75.0, **extra) -> dict:
    return {
        "source": "mix.flac",
        "model": "faster-whisper large-v3 (cuda, float16)",
        "language": "en",
        "language_probability": 0.9,
        "languages": {"en": 61.4, "af": 12.6},
        "duration": duration,
        "integrity": {"truncated": False},
        "teams_transcript": None,
        "segments": [
            {"start": 0.4, "end": 4.0, "text": " Goeie more, let's start.", "language": "af", "words": []},
            {"start": 20.0, "end": 24.0, "text": "The second point.", "language": "en", "words": []},
        ],
        **extra,
    }


META = {"date": "2026-10-30", "source": "recording.m4a", "title": "BlueSky kickoff", "recording_status": "complete"}
LINES = [
    {"speaker": OWNER, "start": 0.4, "end": 4.0, "text": "Goeie more, let's start."},
    {"speaker": "Speaker 2", "start": 5.0, "end": 9.0, "text": "Sure,  the   numbers first."},
    {"speaker": OWNER, "start": 20.0, "end": 24.0, "text": "The second point."},
]


class RenderDraft(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def render(self, lines=LINES, visuals=None, meta=META, data=None, out=None) -> tuple[Path, str]:
        path = draft.render_draft(data or whisper(), lines, visuals, meta, out or self.dir)
        return path, path.read_text(encoding="utf-8")

    def test_house_format(self):
        path, md = self.render()
        self.assertEqual(path, self.dir / "draft.md")
        front, body = parse(md)
        self.assertEqual([k for k, _ in front], SKILL_KEYS)
        fm = dict(front)
        self.assertEqual(fm["people"], f"[{OWNER}, Speaker 2]")   # sorted, as the skill sorts its people
        self.assertEqual(fm["date"], "2026-10-30")
        self.assertEqual(fm["source"], "recording.m4a")
        self.assertEqual(fm["speaker_source"], f"local diarisation (sherpa-onnx); {OWNER} matched by voiceprint")
        self.assertEqual(fm["duration_min"], "1.2")
        self.assertEqual(fm["recording_status"], "complete")
        self.assertEqual(fm["transcriber"], "faster-whisper large-v3 (local, GPU), hand-corrected")
        self.assertIn("Afrikaans and English, language detected per chunk (Afrikaans 13 s, English 61 s)", fm["coverage"])
        self.assertTrue(fm["coverage"].startswith("whole recording"))
        self.assertEqual(fm["redactions"], "none")
        self.assertIn("client: \n", md, "blank client keeps the skill's trailing space")
        self.assertEqual(body[:6], ["", "# BlueSky kickoff transcript — 2026-10-30", "", draft.INTRO_NOTE, "", "## Opening"])
        rows = [l for l in body[7:] if l]
        self.assertEqual(rows, [
            f"- [00:00] **{OWNER}** Goeie more, let's start.",
            "- [00:05] **Speaker 2** Sure, the numbers first.",
            f"- [00:20] **{OWNER}** The second point.",
        ])
        for row in rows:
            self.assertRegex(row, LINE)
        self.assertTrue(md.endswith("\n"))

    def test_owner_not_matched(self):
        lines = [dict(l, speaker="Speaker 1" if l["speaker"] == OWNER else l["speaker"]) for l in LINES]
        _, md = self.render(lines=lines)
        fm = dict(parse(md)[0])
        self.assertEqual(fm["speaker_source"], "local diarisation (sherpa-onnx); owner not matched")
        self.assertEqual(fm["people"], "[Speaker 1, Speaker 2]")
        _, md = self.render(meta=dict(META, owner="Someone Else"))
        self.assertEqual(dict(parse(md)[0])["speaker_source"], "local diarisation (sherpa-onnx); owner not matched")
        # jobs.py's meta keys
        _, md = self.render(meta=dict(META, owner_name=OWNER, owner_matched=True))
        self.assertEqual(dict(parse(md)[0])["speaker_source"], f"local diarisation (sherpa-onnx); {OWNER} matched by voiceprint")
        _, md = self.render(lines=lines, meta=dict(META, owner_name=OWNER, owner_matched=False))
        self.assertEqual(dict(parse(md)[0])["speaker_source"], "local diarisation (sherpa-onnx); owner not matched")

    def test_hours_format(self):
        lines = [{"speaker": OWNER, "start": 59.6, "end": 61.0, "text": "a"},
                 {"speaker": "Speaker 2", "start": 3599.4, "end": 3601.0, "text": "b"},
                 {"speaker": OWNER, "start": 3725.2, "end": 3730.0, "text": "c"},
                 {"speaker": "Speaker 2", "start": 36005.0, "end": 36010.0, "text": "d"}]
        _, md = self.render(lines=lines, data=whisper(duration=36010.0))
        rows = [l for l in parse(md)[1] if l.startswith("- [")]
        self.assertEqual([LINE.match(r)["ts"] for r in rows], ["01:00", "59:59", "1:02:05", "10:00:05"])
        self.assertEqual(dict(parse(md)[0])["duration_min"], "600.2")

    def test_visuals_interleaved_by_time(self):
        visuals = [{"t": 30.0, "text": "Closing slide"}, {"t": 0.0, "text": "Title: Q3 review\n- Revenue R12m\n- Margin 31%"},
                   {"t": 5.0, "text": "Agenda"}, {"t": 12.0, "text": "   "}]
        _, md = self.render(visuals=visuals)
        rows = [l for l in parse(md)[1] if l.startswith("- [")]
        self.assertEqual(rows, [
            "- [00:00] *(on screen)* Title: Q3 review; Revenue R12m; Margin 31%",
            f"- [00:00] **{OWNER}** Goeie more, let's start.",
            "- [00:05] *(on screen)* Agenda",
            "- [00:05] **Speaker 2** Sure, the numbers first.",
            f"- [00:20] **{OWNER}** The second point.",
            "- [00:30] *(on screen)* Closing slide",
        ])
        self.assertIn("on-screen notes from 3 keyframes", dict(parse(md)[0])["coverage"])
        times = [int(LINE.match(r)["ts"].replace(":", "")) for r in rows]
        self.assertEqual(times, sorted(times))

    def test_no_speaker_labels(self):
        for lines in (None, [], [{"speaker": "", "start": 0.4, "end": 4.0, "text": "x"}]):
            _, md = self.render(lines=lines)
            front, body = parse(md)
            fm = dict(front)
            self.assertEqual(fm["speaker_source"], "none")
            self.assertEqual(fm["people"], "[]")
            self.assertIn("no speaker labels", fm["coverage"])
            self.assertEqual([l for l in body if l.startswith("- [")],
                             ["- [00:00] Goeie more, let's start.", "- [00:20] The second point."])

    def test_out_path_and_fallbacks(self):
        target = self.dir / "sub" / "my draft.md"
        path, md = self.render(meta={}, out=target,
                               data=whisper(source="Team sync-20261002_123532-Meeting Recording.mp4",
                                            model="faster-whisper large-v3-turbo (cuda, float16)"))
        self.assertEqual(path, target)
        fm = dict(parse(md)[0])
        self.assertEqual(fm["date"], "2026-10-02")
        self.assertEqual(fm["recording_status"], "complete")
        self.assertEqual(fm["transcriber"], "faster-whisper large-v3-turbo (local, GPU), hand-corrected")
        self.assertIn("# Team sync transcript — 2026-10-02", md)

    def test_transcriber_names_the_cpu_after_the_gpu_ran_out_of_memory(self):
        for model, where in (("faster-whisper large-v3 (cpu, int8)", "CPU"),
                             ("faster-whisper large-v3 (cuda, float16; cpu, int8 from 12:30)", "GPU then CPU")):
            _, md = self.render(data=whisper(model=model))
            self.assertEqual(dict(parse(md)[0])["transcriber"],
                             f"faster-whisper large-v3 (local, {where}), hand-corrected")

    def test_truncated(self):
        data = whisper(integrity={"truncated": True, "declared_seconds": 3600.0, "present_fraction": 0.5})
        _, md = self.render(meta={k: v for k, v in META.items() if k != "recording_status"}, data=data)
        fm = dict(parse(md)[0])
        self.assertEqual(fm["recording_status"], "partial — truncated download holding 50% of a 1:00:00 recording")
        self.assertTrue(fm["coverage"].startswith("first 01:15 only"))
        self.assertEqual([l for l in parse(md)[1] if l][-1],
                         "- [01:15] [file ends here; the rest of the recording is not in the download]")

    def test_without_language_split(self):
        data = whisper()
        del data["languages"]
        _, md = self.render(data=data)
        self.assertNotIn("language detected", dict(parse(md)[0])["coverage"])


@unittest.skipUnless(SKILL.is_file(), f"transcribe skill not at {SKILL}")
class MatchesSkill(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.skill = load_skill()

    def test_intro_note_identical(self):
        self.assertEqual(draft.INTRO_NOTE, self.skill.INTRO_NOTE)

    def test_fmt_ts_identical(self):
        for s in [0, 0.4, 0.5, 1.5, 59.49, 59.5, 61, 599.6, 3599.4, 3599.5, 3600, 3725.2, 36005, 86399.9, -3]:
            self.assertEqual(draft.fmt_ts(s), self.skill.fmt_ts(s), s)

    def test_frontmatter_and_title_match_skill_render(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            data = {"source": "Weekly-20261030_091500-Meeting Recording.mp4", "model": "faster-whisper large-v3 (cuda, float16)",
                    "language": "en", "language_probability": 0.9, "duration": 125.0,
                    "integrity": {"truncated": False}, "teams_transcript": None,
                    "segments": [{"start": 1.0, "end": 3.0, "text": "Hello.", "words": []},
                                 {"start": 70.0, "end": 72.0, "text": "Next item.", "words": []}]}
            (tmp / "whisper.json").write_text(json.dumps(data), encoding="utf-8")
            self.skill.render(tmp / "whisper.json", tmp / "skill.md")
            ours = draft.render_draft(data, None, None, {}, tmp / "ours.md")
            skill_md = (tmp / "skill.md").read_text(encoding="utf-8")
            our_md = ours.read_text(encoding="utf-8")
        skill_front, skill_body = parse(skill_md)
        our_front, our_body = parse(our_md)
        self.assertEqual([k for k, _ in our_front], [k for k, _ in skill_front])
        same = {"client", "people", "date", "source", "speaker_source", "duration_min", "recording_status",
                "transcriber", "redactions"}
        self.assertEqual({k: v for k, v in our_front if k in same}, {k: v for k, v in skill_front if k in same})
        # Without diarisation the body is line for line what the skill writes.
        self.assertEqual(our_body, skill_body)


if __name__ == "__main__":
    unittest.main()
