"""The web app's server: `mooz-gui [--port N] [--no-browser]` (docs/GUI.md).

The standard library's HTTP server, JSON in and out, for the one page in static/. It listens
on this machine only and does no audio work itself: a job is a run of the command line
(jobs.py). Requests from anywhere but this address, and writes sent by another site's page,
are refused.
"""

from __future__ import annotations

import argparse
import json
import signal
import sys
import threading
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlsplit

from . import jobs, listening, live, projects, stagemap
from .projects import UserError

STATIC = Path(__file__).with_name("static")
STATIC_TYPES = {".html": "text/html; charset=utf-8", ".css": "text/css; charset=utf-8",
                ".js": "text/javascript; charset=utf-8", ".svg": "image/svg+xml"}
LOCAL = {"127.0.0.1", "localhost", "[::1]"}
PORT = 8377
MAX_BODY = 2_000_000


class App:
    """What the routes work on: the list of projects and the job runner."""

    def __init__(self, registry: projects.Registry | None = None, runner: jobs.Runner | None = None):
        self.registry = registry or projects.Registry()
        self.runner = runner or jobs.Runner()
        self._roots: dict[str, list[str]] = {}  # a project's folders, for serving its files
        self._lock = threading.Lock()

    def detail(self, pid: str) -> dict:
        d = projects.detail(self.registry.get(pid))
        with self._lock:
            self._roots[pid] = d["roots"]
        return d

    def roots(self, pid: str) -> list[str]:
        with self._lock:
            known = self._roots.get(pid)
        return known if known is not None else self.detail(pid)["roots"]

    def cfg_path(self, pid: str) -> Path:
        path = Path(self.registry.get(pid)["config"])
        if not path.exists():
            raise UserError(f"{path} is gone.")
        return path

    # -- routes: (method, first parts of the path) -> a function of (rest of the path, query, body) --

    def state(self, query: dict) -> dict:
        job = self.runner.job
        since = int(query["since"]) if query.get("since", "").isdigit() else 0
        snap = job.snapshot(since) if job else None
        if snap:
            snap["live"] = self._live(job, snap)
        return {"projects": self.registry.list(), "job": snap, "home": str(Path.home())}

    def _live(self, job: jobs.Job, snap: dict, graph: bool = False) -> dict | None:
        """The take the job is at, on its graph (live.py): which take and profile, and where each box
        is; with graph, the graph itself (asked for once a take: it does not change while it is made)."""
        units = snap["units"]
        unit = units[snap["current"]] if snap["current"] is not None else None
        if unit is None and snap["state"] != "running" and units:  # a job that is over: where its last take got to
            unit = next((u for u in reversed(units) if u["state"] != "pending"), None)
        if unit is not None:
            take, profile = unit["take"], unit["profile"]
        elif snap["state"] == "running" and snap["ahead"] and units:  # the stems of every take first: this one's
            take, profile = snap["ahead"]["take"], units[0]["profile"]
        else:
            return None
        key = f"{take}\n{profile}"
        plans = job.__dict__.setdefault("plans", {})
        if key not in plans:
            try:
                plans[key] = live.plan(self.cfg_path(job.project["id"]), take, profile, snap["separate_only"])
            except (UserError, OSError, KeyError, ValueError):
                plans[key] = None  # a project that is gone, settings that do not load: no graph, the rest of the page as ever
        if plans[key] is None:
            return None
        out = {"key": key, "take": take, "profile": profile, "states": live.states(plans[key], take, unit, snap["separated"])}
        return out | ({"graph": plans[key]} if graph else {})

    def start_job(self, body: dict) -> dict:
        pid = body.get("project", "")
        d = self.detail(pid)
        if d.get("error"):
            raise UserError(d["error"])
        wanted = body.get("takes") or []
        chosen = [t for t in d["takes"] if t["name"] in wanted]
        if len(chosen) != len(set(wanted)):
            raise UserError("Some of the chosen takes are not in the project any more. Reload the page.")
        names = body.get("profiles") or []
        separate_only = bool(body.get("separate_only"))
        if not separate_only:
            known = {p["name"]: p for p in d["profiles"]}
            dirs: dict[str, str] = {}
            for n in names:
                if n not in known:
                    raise UserError(f"There is no profile '{n}'.")
                if "error" in known[n]:
                    raise UserError(f"Profile {n}: {known[n]['error']}")
                other = dirs.setdefault(known[n]["output_dir"], n)
                if other != n:
                    raise UserError(f"{other} and {n} both write to {known[n]['output_dir']}: the second would overwrite the "
                                    "first. Run them one at a time, or give one its own [output] dir.")
        job = self.runner.start(d, chosen, names, bool(body.get("ab")), separate_only, bool(body.get("again")),
                                bool(body.get("separate_again")))
        return {"job": job.snapshot()}


class Handler(BaseHTTPRequestHandler):
    app: App  # set by serve()
    protocol_version = "HTTP/1.1"
    server_version = "mooz-gui"

    def log_message(self, fmt, *args):  # quiet: the page polls every second
        pass

    def handle(self) -> None:
        """A connection the browser drops is how it ends one, not an error: a player that changes
        file resets the connection it was reading the old file on, and the next read here fails."""
        try:
            super().handle()
        except (ConnectionResetError, ConnectionAbortedError, BrokenPipeError):
            self.close_connection = True

    # -- plumbing --------------------------------------------------------------------------

    def _send(self, status: int, body: bytes, ctype: str, extra: dict | None = None) -> None:
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _json(self, data, status: int = 200) -> None:
        self._send(status, json.dumps(data, default=str).encode(), "application/json")

    def _error(self, status: int, message: str) -> None:
        self._json({"error": message}, status)

    def _local(self) -> bool:
        """Only this machine's own pages: the Host must be a local name (a DNS-rebinding page has
        its own), and a write must come from our own page, as JSON."""
        host = self.headers.get("Host") or ""
        name = host.rsplit("]", 1)[0] + "]" if host.startswith("[") else host.rsplit(":", 1)[0]
        if name not in LOCAL:
            return False
        if self.command in ("GET", "HEAD"):
            return True
        origin = self.headers.get("Origin")
        if origin and (urlsplit(origin).hostname or "") not in {"127.0.0.1", "localhost", "::1"}:
            return False
        return self.command == "DELETE" or (self.headers.get("Content-Type") or "").startswith("application/json")

    def _body(self) -> dict:
        n = int(self.headers.get("Content-Length") or 0)
        if n > MAX_BODY:
            raise UserError("Too much to send at once.")
        try:
            data = json.loads(self.rfile.read(n) or b"{}")
        except json.JSONDecodeError:
            raise UserError("Not JSON.") from None
        if not isinstance(data, dict):
            raise UserError("A JSON object is needed.")
        return data

    def _handle(self) -> None:
        if not self._local():
            self.close_connection = True
            return self._error(HTTPStatus.FORBIDDEN, "This app answers only to this machine.")
        url = urlsplit(self.path)
        parts = [unquote(p) for p in url.path.split("/") if p]
        query = {k: v[0] for k, v in parse_qs(url.query).items()}
        try:
            if self.command in ("GET", "HEAD"):
                self._get(parts, query)
            else:
                self._write(parts, self._body() if self.command == "POST" else {})
        except UserError as e:
            self._error(HTTPStatus.BAD_REQUEST, str(e))
        except (BrokenPipeError, ConnectionResetError):
            self.close_connection = True
        except Exception as e:  # ours to fix, but the page should say so
            self._error(HTTPStatus.INTERNAL_SERVER_ERROR, f"{type(e).__name__}: {e}")

    do_GET = do_HEAD = do_POST = do_DELETE = _handle

    # -- routes ----------------------------------------------------------------------------

    def _get(self, parts: list[str], query: dict) -> None:
        app = self.app
        match parts:
            case []:
                self._static("index.html")
            case ["static", name]:
                self._static(name)
            case ["files", pid, root, *rel] if rel and root.isdigit():
                self._file(projects.served_file(app.roots(pid), int(root), "/".join(rel)))
            case ["api", "state"]:
                self._json(app.state(query))
            case ["api", "live"]:  # the take the job is at, with its graph
                job = app.runner.job
                self._json({"live": app._live(job, job.snapshot(10 ** 9), graph=True) if job else None})
            case ["api", "browse"]:
                self._json(projects.browse(query.get("path")))
            case ["api", "preview"]:
                self._json(projects.preview(query.get("folder", "")))
            case ["api", "projects", pid]:
                self._json(app.detail(pid))
            case ["api", "projects", pid, "settings"]:
                self._json(projects.settings(app.cfg_path(pid)))
            case ["api", "projects", pid, "profiles", name]:
                cfg_path = app.cfg_path(pid)
                if "new" in query:  # where a new profile starts (not saved yet)
                    self._json(projects.new_profile(cfg_path, name, query.get("copy")))
                else:
                    self._json(projects.read_profile(cfg_path, name))
            case ["api", "projects", pid, "takes", take, "stages"]:  # the take's files along the pipeline
                app.roots(pid)
                self._json(stagemap.build(app.cfg_path(pid), take, query.get("profile") or projects.DEFAULT))
            case _:
                self._error(HTTPStatus.NOT_FOUND, "Nothing here.")

    def _write(self, parts: list[str], body: dict) -> None:
        app = self.app
        match [self.command, *parts]:
            case ["POST", "api", "projects"]:
                p = projects.create(app.registry, str(body.get("name", "")), str(body.get("folder", "")),
                                    list(body.get("profiles") or []))
                self._json(p)
            case ["DELETE", "api", "projects", pid]:
                job = app.runner.job
                if app.runner.running() and job.project["id"] == pid:
                    raise UserError("This project has a job running. Cancel it first.")
                app.registry.remove(pid)
                self._json({})
            case ["POST", "api", "projects", pid, "settings"]:
                projects.save_settings(app.cfg_path(pid), body.get("values"), body.get("run"), body.get("toml"))
                self._json(projects.settings(app.cfg_path(pid)))
            case ["POST", "api", "projects", pid, "profiles", name]:
                projects.save_profile(app.cfg_path(pid), name, str(body.get("text", "")), body.get("values"))
                self._json(projects.read_profile(app.cfg_path(pid), name))
            case ["DELETE", "api", "projects", pid, "profiles", name]:
                projects.delete_profile(app.cfg_path(pid), name)
                self._json({})
            case ["POST", "api", "projects", pid, "verdicts"]:  # a blind comparison's choice (listening.py)
                listening.add(app.cfg_path(pid), body, app.detail(pid)["takes"])
                self._json(app.detail(pid)["listening"])
            case ["DELETE", "api", "projects", pid, "verdicts", vid]:
                listening.remove(app.cfg_path(pid), vid)
                self._json(app.detail(pid)["listening"])
            case ["POST", "api", "jobs"]:
                self._json(app.start_job(body))
            case ["POST", "api", "job", "cancel"]:
                app.runner.cancel()
                job = app.runner.job
                self._json({"job": job.snapshot() if job else None})
            case _:
                self._error(HTTPStatus.NOT_FOUND, "Nothing here.")

    def _static(self, name: str) -> None:
        f = STATIC / name
        if f.parent != STATIC or f.suffix not in STATIC_TYPES or not f.is_file():
            return self._error(HTTPStatus.NOT_FOUND, "Nothing here.")
        self._send(200, f.read_bytes(), STATIC_TYPES[f.suffix])

    def _file(self, path: Path) -> None:
        """A project's file, with byte ranges: a browser seeks in audio by asking for them."""
        size = path.stat().st_size
        start, end, status = 0, size - 1, 200
        rng = self.headers.get("Range", "")
        if rng.startswith("bytes=") and "," not in rng:
            a, _, b = rng[6:].partition("-")
            try:
                if a:
                    start, end = int(a), min(int(b), size - 1) if b else size - 1
                else:  # the last b bytes
                    start = max(0, size - int(b))
                status = 206
            except ValueError:
                start, end, status = 0, size - 1, 200
            if status == 206 and not 0 <= start <= end:
                self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                self.send_header("Content-Range", f"bytes */{size}")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
        self.send_response(status)
        self.send_header("Content-Type", projects.SERVED[path.suffix.lower()])
        self.send_header("Content-Length", str(end - start + 1))
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("X-Content-Type-Options", "nosniff")
        if status == 206:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        if self.command == "HEAD":
            return
        with open(path, "rb") as f:
            f.seek(start)
            left = end - start + 1
            while left > 0:
                chunk = f.read(min(1 << 16, left))
                if not chunk:
                    break
                self.wfile.write(chunk)
                left -= len(chunk)


def serve(app: App, port: int = PORT) -> ThreadingHTTPServer:
    """The server, bound to this machine only (port 0: any free one); the caller runs it."""
    handler = type("BoundHandler", (Handler,), {"app": app})
    server = ThreadingHTTPServer(("127.0.0.1", port), handler)
    server.daemon_threads = True
    return server


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="mooz-gui", description="The web app: projects, remastering, settings, progress.")
    ap.add_argument("--port", type=int, default=PORT, help=f"port on 127.0.0.1 (default {PORT})")
    ap.add_argument("--no-browser", action="store_true", help="do not open the browser")
    args = ap.parse_args(argv)
    app = App()
    here = app.registry.adopt(Path.cwd())  # started in a project's folder: it is in the list
    try:
        server = serve(app, args.port)
    except OSError as e:
        sys.exit(f"cannot listen on 127.0.0.1:{args.port} ({e.strerror}): is the app already running? "
                 f"Try http://127.0.0.1:{args.port} or --port")
    url = f"http://127.0.0.1:{server.server_address[1]}/" + (f"#/p/{here['id']}" if here else "")
    print(f"mooz: {url}  (Ctrl-C stops it, and a running job with it)", flush=True)
    if not args.no_browser:
        threading.Timer(0.3, webbrowser.open, [url]).start()

    def stop(signum, frame):  # a plain kill leaves like Ctrl-C: the running job is stopped too
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, stop)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        if app.runner.running():
            print("stopping the running job ...")
            app.runner.cancel()
        server.server_close()


if __name__ == "__main__":
    main()
