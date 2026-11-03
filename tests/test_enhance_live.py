"""Live: the enhancing step with a real headless Claude run, in a throwaway vault, on the end-to-end draft.

Skipped unless SCRIBE_LIVE_CLAUDE=1, because it spends a real Claude run (the configured model, up to
SCRIBE_LIVE_TIMEOUT seconds, default 1800). It first runs the Local pipeline on two_speakers.wav exactly as
test_end_to_end does, then re-queues that job for a vault named "Test" so jobs.run_pipeline calls
enhance.run_claude for real with cwd = a temp vault holding a minimal CLAUDE.md and
Clients\\Test\\Meetings\\. Nothing outside the temp dir is written by Scribe; the Claude CLI keeps its own
session record. SCRIBE_LIVE_KEEP=1 keeps the temp dir for inspection (its path is logged).

    set SCRIBE_LIVE_CLAUDE=1
    .venv\\Scripts\\python.exe -m unittest tests.test_enhance_live -v
"""
from __future__ import annotations

import hashlib
import logging
import os
import re
import sys
import unittest
from datetime import date
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for p in (ROOT, HERE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from scribe import config, enhance, jobs  # noqa: E402

import test_end_to_end as e2e  # noqa: E402

log = logging.getLogger("scribe.tests.enhance_live")

LIVE = os.environ.get("SCRIBE_LIVE_CLAUDE") == "1"
KEEP = os.environ.get("SCRIBE_LIVE_KEEP") == "1"
TIMEOUT_S = int(os.environ.get("SCRIBE_LIVE_TIMEOUT", "1800"))
OWNER = e2e.OWNER
VAULT = "Test"

CLAUDE_MD = """# Test vault

A throwaway vault for one Scribe test run. Use only this file and the shelves below as context.

- `Clients\\Test\\` is the only project: the Test client's bakery website. Its meetings go in
  `Clients\\Test\\Meetings\\`.

Write nothing outside this folder. Send nothing to anyone.
"""

CLIENT_README = """# Test client

The Test client is a small bakery. Its photos come from the sourdough menu and its own price list, both slow to
update by hand. The project is a new website with online orders and a dashboard that pulls the orders and the stock
together. Sam checks the old site.
"""


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_vault(root: Path) -> Path:
    vault = root / "vault"
    (vault / "Clients" / "Test" / "Meetings").mkdir(parents=True)
    (vault / "CLAUDE.md").write_text(CLAUDE_MD, encoding="utf-8")
    (vault / "Clients" / "Test" / "README.md").write_text(CLIENT_README, encoding="utf-8")
    return vault


@unittest.skipUnless(LIVE, "live Claude run: set SCRIBE_LIVE_CLAUDE=1")
class EnhanceLive(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        reason = e2e.missing_requirement()
        if reason:
            raise unittest.SkipTest(reason)
        exe = Path(config.expand(config.DEFAULTS["claude"]["exe"]))
        if not exe.is_file():
            raise unittest.SkipTest(f"claude not found: {exe}")
        cls.sb = e2e.Sandbox("scribe_live_")
        cls.addClassCleanup(cls.sb.close, KEEP)
        if KEEP:
            log.warning("SCRIBE_LIVE_KEEP: temp dir kept at %s", cls.sb.root)

        cls.sb.enroll_owner()
        source = cls.sb.drop(e2e.FIXTURES / "two_speakers.wav")
        job, _statuses, _calls = e2e.run_local_job(cls.sb, source)
        if job.status != "done":
            raise AssertionError(f"local pipeline failed: {job.error}")
        cls.folder = job.folder
        cls.draft_sha = _sha(cls.folder / "draft.md")

        cls.vault = make_vault(cls.sb.root)
        cls.sb.settings["vaults"] = {VAULT: str(cls.vault)}
        cls.sb.settings["claude"]["timeout_s"] = TIMEOUT_S
        cls.calls: list = []
        real = enhance.run_claude

        def spy(job_dict, vault_path, log_path, settings, **kw):
            result = real(job_dict, vault_path, log_path, settings, **kw)
            cls.calls.append({"vault_path": Path(vault_path), "log_path": Path(log_path), "result": result})
            return result

        job.vault, job.status = VAULT, "queued"
        cls.statuses: list[str] = []
        with mock.patch.object(enhance, "run_claude", spy):
            cls.job = jobs.run_pipeline(
                job, cls.sb.settings,
                progress=lambda j, f: cls.statuses.append(j.status) if f is None else None)

    def _vault_file(self, key: str) -> Path:
        rel = self.job.outputs.get(key)
        self.assertTrue(rel, f"no {key} path in job outputs")
        path = (self.vault / rel).resolve()
        self.assertTrue(path.is_relative_to(self.vault.resolve()), path)
        self.assertTrue(path.is_file(), path)
        return path

    def test_job_done_after_one_claude_run(self):
        self.assertEqual(self.job.status, "done", self.job.error)
        self.assertEqual(self.statuses, ["queued", "transcribing", "labelling", "drafting", "enhancing", "done"])
        self.assertEqual(len(self.calls), 1)
        call = self.calls[0]
        self.assertEqual(call["vault_path"], self.vault)
        self.assertEqual(call["result"]["returncode"], 0)
        self.assertTrue(call["result"]["session_id"])
        self.assertEqual(self.job.outputs.get("session_id"), call["result"]["session_id"])

    def test_log_and_enhance_json(self):
        text = (self.folder / "claude.log").read_text(encoding="utf-8", errors="replace")
        self.assertIn("--- prompt ---", text)
        self.assertIn(str((self.folder / "draft.md").resolve()), text)  # the prompt carries long paths, never 8.3
        self.assertTrue((self.folder / "enhance.json").is_file())

    def test_transcript_shelved_with_recording(self):
        transcript = self._vault_file("transcript")
        shelf = transcript.parent
        allowed = {(self.vault / "Clients" / "Test" / "Meetings").resolve(), (self.vault / "Meetings").resolve()}
        self.assertTrue(any(shelf == a or shelf.is_relative_to(a) for a in allowed), shelf)
        self.assertTrue(list(shelf.glob("*.m4a")), f"no .m4a beside {transcript.name}")
        text = transcript.read_text(encoding="utf-8")
        for word in (OWNER, "Speaker 2", "sourdough"):
            self.assertIn(word, text)

    def test_notes(self):
        notes = self._vault_file("notes")
        when = enhance.human_date(date.fromisoformat(self.job.outputs["date"]))
        self.assertRegex(notes.name, rf" notes {re.escape(when)}\.md$")
        self.assertEqual(notes.parent, self._vault_file("transcript").parent)
        text = notes.read_text(encoding="utf-8")
        self.assertRegex(text, r"(?mi)^### Decisions")
        self.assertRegex(text, r"(?mi)^### To-?do")

    def test_job_folder_left_as_it_was(self):
        self.assertEqual(_sha(self.folder / "draft.md"), self.draft_sha)
        for name in ("whisper.json", "speakers.json", "recording.m4a", "audio.flac"):
            self.assertTrue((self.folder / name).is_file(), name)


if __name__ == "__main__":
    unittest.main()
