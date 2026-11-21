"""One run of the command line, followed through its progress events (progress.py).

The run is a child process in a process group of its own: the memory cap, the GPU and a crash
stay there, and Cancel can stop all of it (the pipeline, a separation, ffmpeg) at once.
One job at a time.
"""

from __future__ import annotations

import collections
import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

from .. import progress
from .projects import DEFAULT, UserError

LOG_LINES = 2000


def command(cfg_path: Path, rooms: list[str], profiles: list[str], ab: bool, separate_only: bool,
            again: bool = False, separate_again: bool = False) -> list[str]:
    """The command line for a job: the takes by one of their files, the profiles as presets.
    again: also the takes that are already mastered with these very settings (left alone otherwise).
    separate_again: their tracks are separated again too, the kept stems not used (and so mastered again)."""
    cmd = [sys.executable, "-m", "mooz.cli", *rooms, "-c", str(cfg_path)]
    if separate_only:
        return cmd + ["--separate-only"] + (["--separate-again"] if separate_again else [])
    for p in profiles:
        cmd += ["--preset", p]
    return cmd + (["--ab"] if ab else []) + (["--separate-again"] if separate_again else ["--again"] if again else [])


class Job:
    def __init__(self, project: dict, takes: list[dict], profiles: list[str], separate_only: bool):
        self.project = {"id": project["id"], "name": project["name"]}
        self.separate_only = separate_only
        self.audio = {t["name"]: t.get("seconds") or 0.0 for t in takes}  # each take's length, for the estimate
        # the plan as asked for; the run's own "plan" event replaces it (same order)
        self.units = [self._unit(t["name"], p) for p in ([""] if separate_only else profiles) for t in takes]
        self.state = "running"  # running, done, failed, cancelled
        self.message = ""
        self.started = time.time()
        self.ended: float | None = None
        self.current: int | None = None
        self.ahead: dict | None = None        # separating the takes with close tracks before the first
        self.separation: dict | None = None   # a separation in flight
        self.separated: dict[str, dict] = {}  # every track the run has separated or found separated, by its cache's name
        self.log: collections.deque[str] = collections.deque(maxlen=LOG_LINES)
        self.lines = 0                        # log lines so far, the dropped ones included
        self.saw_end = False
        self.stopped: dict | None = None      # the run said it was stopped from outside, and why (out of memory)
        self.cancelled = False
        self.proc: subprocess.Popen | None = None
        self.lock = threading.Lock()

    @staticmethod
    def _unit(take: str, preset: str) -> dict:
        return {"take": take, "profile": preset if preset not in ("", DEFAULT) else DEFAULT, "state": "pending", "frac": 0.0,
                "stage": "", "key": "", "detail": "", "started": None, "seconds": None, "summary": None, "error": None,
                "separated": False, "kept": False, "part": 0.0, "chains": {}}

    # -- the run's events ------------------------------------------------------------------

    def event(self, e: dict) -> None:
        ev = e.get("ev")
        if ev == "plan":
            self.units = [self._unit(u["take"], u["preset"]) for u in e["units"]]
        elif ev == "prepare":
            self.ahead = {"take": e["take"], "i": e["i"], "n": e["n"]}
        elif ev == "separation":
            made_here = self.separated.get(e["track"], {}).get("state") == "done"  # by this job: its stems found again
            if not (made_here and e["state"] == "cached"):                         # at the take's turn are not "kept from an earlier run"
                self.separated[e["track"]] = {"state": e["state"], "i": e.get("i", 0), "n": e.get("n", 0), "model": e.get("model", "")}
            if e["state"] in ("cached", "done"):
                self.separation = None
            else:
                self.separation = {"track": e["track"], "i": e.get("i", 0), "n": e.get("n", 0), "model": e.get("model", "")}
                if self.current is not None:
                    self.units[self.current]["separated"] = True
        elif ev == "unit" and e["i"] < len(self.units):
            self.current, self.ahead = e["i"], None
            self.units[e["i"]] |= {"state": "running", "started": time.time(), "chains": {}, "part": 0.0}
        elif ev == "stage" and self.current is not None:
            self.units[self.current] |= {"frac": e["frac"], "stage": e["label"], "key": e.get("key", ""),
                                         "detail": e.get("detail", ""), "part": e.get("part", 0.0)}
        elif ev == "chain" and self.current is not None:  # one stem's chain, from whichever process makes it
            self.units[self.current]["chains"] = self.units[self.current]["chains"] | {
                e["stem"]: {"state": e["state"], "frac": float(e.get("frac", 0.0))}}
        elif ev == "unit_done" and e["i"] < len(self.units):
            self.units[e["i"]] |= {"state": "done" if e["ok"] else "failed", "frac": 1.0, "seconds": e.get("seconds"),
                                   "summary": e.get("summary"), "error": e.get("error"), "stage": "", "key": "", "detail": "",
                                   "kept": bool(e.get("kept"))}  # kept: already mastered with these settings, not made again
            self.current = None
        elif ev == "stopped":
            self.stopped = {"why": e.get("why"), "limit": e.get("limit")}
        elif ev == "end":
            self.saw_end = True

    def line(self, text: str) -> None:
        with self.lock:
            if text.startswith(progress.PREFIX):
                try:
                    self.event(json.loads(text[len(progress.PREFIX):]))
                    return
                except (json.JSONDecodeError, KeyError, TypeError):
                    pass  # not ours after all: a line of the log
            self.log.append(text)
            self.lines += 1

    def finish(self, returncode: int) -> None:
        with self.lock:
            self.ended = time.time()
            self.separation = self.ahead = None
            failed = [u for u in self.units if u["state"] == "failed"]
            why = None  # stopped from outside: a kill leaves no message of the run's own
            if self.stopped and self.stopped.get("why") == "memory":
                why = f"it needed more memory than the {self.stopped.get('limit') or 'limit'} it may use"
            elif returncode in (-9, 137):
                why = "it was killed, most likely for lack of memory"
            for u in self.units:
                if u["state"] == "running":  # the run stopped under it
                    u["state"] = "cancelled" if self.cancelled else "failed"
                    u["error"] = None if self.cancelled else (f"stopped here: {why}" if why else "the run stopped before this take was finished")
            if self.cancelled:
                self.state, self.message = "cancelled", "Cancelled."
            elif returncode == 0 and self.saw_end:
                self.state = "done"
            elif self.saw_end and failed:
                self.state = "failed"
                self.message = f"{len(failed)} of {len(self.units)} failed."
            elif why:
                self.state = "failed"
                self.message = (f"The system stopped the run: {why}. The takes already finished are kept: start it again and "
                                "only the others are made.")
            else:  # it never got to the end: the last lines say why
                self.state = "failed"
                tail = [x.strip() for x in self.log if x.strip()]
                self.message = tail[-1] if tail else f"The run stopped (exit code {returncode})."
            self.current = None

    # -- for the page ----------------------------------------------------------------------

    def snapshot(self, since: int = 0) -> dict:
        with self.lock:
            # what is left, from the takes finished so far: their seconds of work per second of audio
            # (a take's time goes with its length), or their mean time where a length is not known
            done = [u for u in self.units if u["state"] == "done" and u["seconds"]]
            left = [u for u in self.units if u["state"] in ("pending", "running")]
            eta = None
            if self.state == "running" and done and left:
                mean = sum(u["seconds"] for u in done) / len(done)
                audio = sum(self.audio.get(u["take"], 0.0) for u in done)
                rate = sum(u["seconds"] for u in done) / audio if audio and all(self.audio.get(u["take"]) for u in done) else None
                cost = lambda u: rate * self.audio[u["take"]] if rate and self.audio.get(u["take"]) else mean  # noqa: E731
                eta = max(0.0, sum(cost(u) * (1 - u["frac"]) for u in left))
            first = self.lines - len(self.log)  # the number of the oldest line still held
            return {
                "project": self.project, "state": self.state, "message": self.message, "separate_only": self.separate_only,
                "started": self.started, "elapsed": (self.ended or time.time()) - self.started, "eta": eta,
                "frac": sum(u["frac"] for u in self.units) / max(1, len(self.units)),
                "current": self.current, "ahead": self.ahead, "separation": self.separation,
                "separated": dict(self.separated), "units": [dict(u) for u in self.units],
                "log": list(self.log)[max(0, since - first):], "lines": self.lines,
            }


class Runner:
    """Starts jobs and holds the last one. make_command: what to run (replaced in the tests)."""

    def __init__(self, make_command=command):
        self.make_command = make_command
        self.job: Job | None = None
        self.lock = threading.Lock()

    def running(self) -> bool:
        return self.job is not None and self.job.state == "running"

    def start(self, project: dict, takes: list[dict], profiles: list[str], ab: bool = False,
              separate_only: bool = False, again: bool = False, separate_again: bool = False) -> Job:
        """takes: the project's takes to run (name and room file); profiles: by name. again: also
        what is already mastered with these settings. separate_again: and their stems made again."""
        with self.lock:
            if self.running():
                raise UserError("A job is already running. Wait for it, or cancel it.")
            if not takes:
                raise UserError("Choose at least one take.")
            if not profiles and not separate_only:
                raise UserError("Choose at least one profile.")
            cfg_path = Path(project["config"])
            job = Job(project, takes, profiles, separate_only)
            env = os.environ | {"REHEARSAL_PROGRESS": "1", "PYTHONUNBUFFERED": "1"}
            cmd = self.make_command(cfg_path, [t["room"] for t in takes], profiles, ab, separate_only, again, separate_again)
            job.log.append("$ " + " ".join(cmd[1:]))
            job.lines = 1
            try:
                job.proc = subprocess.Popen(cmd, cwd=cfg_path.parent, env=env, stdout=subprocess.PIPE,
                                            stderr=subprocess.STDOUT, text=True, errors="replace", start_new_session=True)
            except OSError as e:
                raise UserError(f"Could not start the run: {e}") from None
            self.job = job
            threading.Thread(target=self._follow, args=(job,), daemon=True).start()
            return job

    @staticmethod
    def _follow(job: Job) -> None:
        for text in job.proc.stdout:
            job.line(text.rstrip("\n"))
        job.finish(job.proc.wait())

    def cancel(self, wait_s: float = 8.0) -> None:
        """Stop the running job: its whole process group, gently first."""
        job = self.job
        if job is None or job.state != "running" or job.proc is None:
            return
        job.cancelled = True
        for sig in (signal.SIGTERM, signal.SIGKILL):
            try:
                os.killpg(job.proc.pid, sig)
            except ProcessLookupError:
                return
            try:
                job.proc.wait(timeout=wait_s)
                return
            except subprocess.TimeoutExpired:
                continue
