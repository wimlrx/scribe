"""Scribe's local stage from the command line, for the transcribe skill in a Claude session.

    python -m scribe.run "<recording or video>"

Runs the pipeline synchronously with vault "Local" (no headless Claude run: the calling session does the
correction) and ends with "DONE <job folder>", or "FAILED <reason>" and exit code 2. It takes the same job-folder
lock as the running app (jobs.run_pipeline), so the two never run one folder at once. It shares the GPU with the
app rather than waiting for it.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import datetime
from pathlib import Path


def _progress(job, fraction: float | None) -> None:
    """jobs.run_pipeline calls progress(job, fraction): the status on each change, then 0..1 inside long steps."""
    print(job.status if fraction is None else f"{job.status} {fraction:.2f}", flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="scribe.run")
    parser.add_argument("source", type=Path)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s", stream=sys.stdout)

    os.environ["HF_HUB_OFFLINE"] = "1"   # models from the local cache only
    from . import config, jobs

    jobs.kill_children_with_me()          # ffmpeg and the labelling child end with this run
    source = args.source.resolve()
    if not source.is_file():
        print(f"FAILED no such file: {source}", flush=True)
        return 2
    settings = config.load_settings()
    folder = config.new_job_folder(settings=settings)
    now = datetime.now().isoformat(timespec="seconds")
    job = jobs.Job(id=folder.name, kind="file", source=source, folder=folder, vault="Local",
                   status="queued", error=None, created=now, updated=now, outputs={})
    job = jobs.run_pipeline(job, settings, progress=_progress)
    if job.status != "done":
        print(f"FAILED {job.error or job.status} ({folder})", flush=True)
        return 2
    print(f"DONE {folder}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
