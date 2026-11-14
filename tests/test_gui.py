"""The web app (mooz/gui), over real HTTP on a temporary folder, and the progress
events it follows.

No audio is processed here but in the last test: a job's command is replaced by a stand-in
that prints the events a run would, so the server, the files it writes and the job's
bookkeeping are what is checked.
"""

import http.client
import json
import os
import socket
import struct
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

import test_pipeline as tp
from mooz import cli, config, progress, separate, stages
from mooz.gui import jobs, projects, server

FS = 48000
SETTINGS = """\
# The band's settings.
[loudness]
target_lufs = -14.0   # where streaming plays

[output]
mp3 = false
# artist = "Circhi"
"""


def _wav(path: Path, seconds: float = 0.5, channels: int = 2) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, np.zeros((int(seconds * FS), channels), dtype=np.float32), FS, subtype="FLOAT")


def _recordings(folder: Path) -> None:
    """Two takes: a Zoom's files of one moment in a folder of their own, and a file on its own."""
    take = folder / "261105_225442 Miramare"
    _wav(take / "261105_225442_TrMic.WAV")
    _wav(take / "261105_225442_Tr2.WAV", channels=1)
    _wav(take / "261105_225442_TrLR.WAV")
    _wav(folder / "260928_231324_TrLR Vita.WAV", seconds=1.0)


class Client:
    def __init__(self, port: int):
        self.port = port

    def call(self, method: str, path: str, body=None, headers: dict | None = None):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=20)
        h = dict(headers or {})
        data = None
        if body is not None:
            data = json.dumps(body).encode()
            h.setdefault("Content-Type", "application/json")
        conn.request(method, path, body=data, headers=h)
        res = conn.getresponse()
        raw = res.read()
        conn.close()
        ctype = res.getheader("Content-Type") or ""
        return res.status, (json.loads(raw) if ctype.startswith("application/json") and raw else raw), res

    def ok(self, method: str, path: str, body=None):
        status, data, _ = self.call(method, path, body)
        assert status == 200, data
        return data


@pytest.fixture
def app(tmp_path):
    """The app on a free port, with its list of projects in the temporary folder."""
    a = server.App(projects.Registry(tmp_path / "config" / "projects.json"))
    srv = server.serve(a, 0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield a, Client(srv.server_address[1])
    a.runner.cancel(wait_s=2)
    srv.shutdown()
    srv.server_close()


def _project(client: Client, folder: Path, name: str = "Rehearsal") -> dict:
    _recordings(folder)
    return client.ok("POST", "/api/projects", {"name": name, "folder": str(folder), "profiles": ["default", "albini"]})


def _stand_in(script: str):
    """A job's command replaced by a script that prints what a run would."""
    return lambda cfg_path, rooms, profiles, ab, separate_only: [sys.executable, "-c", script]


def _wait(client: Client, until, seconds: float = 15.0) -> dict:
    t0 = time.time()
    while time.time() - t0 < seconds:
        job = client.ok("GET", "/api/state")["job"]
        if job and until(job):
            return job
        time.sleep(0.05)
    raise AssertionError(f"the job did not get there in {seconds} s: {job}")


def test_a_project_is_made_from_a_folder_and_shows_its_takes(app, tmp_path):
    _, client = app
    rec = tmp_path / "rec"
    _recordings(rec)
    found = client.ok("GET", f"/api/preview?folder={rec}")
    assert [t["name"] for t in found["takes"]] == ["260928_231324_TrLR Vita", "261105_225442 Miramare"] and not found["has_settings"]
    assert not (rec / "rehearsal.toml").exists()  # looking writes nothing

    p = client.ok("POST", "/api/projects", {"name": "Rehearsal", "folder": str(rec), "profiles": ["default", "albini"]})
    cfg = config.load(rec / "rehearsal.toml")
    assert cfg["input"]["paths"] == ["."] and cfg["input"]["presets"] == ["", "albini"]
    assert [x["name"] for x in client.ok("GET", "/api/state")["projects"]] == ["Rehearsal"]

    d = client.ok("GET", f"/api/projects/{p['id']}")
    vita, miramare = d["takes"]
    assert (miramare["title"], miramare["date"], miramare["tracks"]) == ("Miramare", "2026-11-05", ["TrMic", "Tr2", "TrLR"])
    assert (vita["title"], vita["tracks"], vita["seconds"]) == ("Vita", ["stereo"], 1.0)
    assert miramare["room_file"] == "261105_225442 Miramare/261105_225442_TrMic.WAV"  # the recording, for listening
    status, _, r = client.call("GET", f"/files/{p['id']}/{miramare['root']}/" + miramare["room_file"].replace(" ", "%20"),
                               headers={"Range": "bytes=0-3"})
    assert status == 206 and r.getheader("Content-Type") == "audio/wav"
    assert d["run"] == ["default", "albini"] and miramare["results"] == {}
    assert {x["name"]: x["output_dir"] for x in d["profiles"]}["albini"] == "processed/albini"


def test_a_folder_with_settings_is_adopted_as_it_is(app, tmp_path):
    _, client = app
    rec = tmp_path / "rec"
    _recordings(rec)
    (rec / "rehearsal.toml").write_text(SETTINGS)
    client.ok("POST", "/api/projects", {"name": "Old", "folder": str(rec), "profiles": ["indie"]})
    assert (rec / "rehearsal.toml").read_text() == SETTINGS


def test_results_come_from_the_reports_and_their_files_are_served_with_ranges(app, tmp_path):
    _, client = app
    rec = tmp_path / "rec"
    p = _project(client, rec)
    out = rec / "processed"
    (out / "reports").mkdir(parents=True)
    (out / "reports" / "260928_231324_TrLR Vita.json").write_text(json.dumps({
        "original": {"lra_lu": 6.0}, "processed": {"integrated_lufs": -14.0, "lra_lu": 3.0, "true_peak_dbtp": -1.0},
        "trim": {"start_s": 1.5, "end_s": 9.0}, "finalize": {"limiter": {"limited_pct": 4.2}}}))
    (out / "260928_231324_TrLR Vita.mp3").write_bytes(bytes(range(200)))

    vita = client.ok("GET", f"/api/projects/{p['id']}")["takes"][0]
    res = vita["results"]["default"]
    assert (res["lufs"], res["lra_lu"], res["original_lra_lu"], res["limited_pct"]) == (-14.0, 3.0, 6.0, 4.2)
    assert res["trim_s"] == 1.5  # where the master starts in the recording: the two are played from the same moment
    url = f"/files/{p['id']}/{vita['root']}/" + res["files"]["mp3"].replace(" ", "%20")
    status, body, r = client.call("GET", url, headers={"Range": "bytes=10-19"})
    assert status == 206 and body == bytes(range(10, 20)) and r.getheader("Content-Range") == "bytes 10-19/200"
    assert client.call("GET", url)[1] == bytes(range(200))

    # nothing outside the project's folders, and nothing but audio, figures and reports
    (tmp_path / "secret.mp3").write_bytes(b"x")
    for path in ("..%2Fsecret.mp3", "rehearsal.toml", "processed/nothing.mp3"):
        assert client.call("GET", f"/files/{p['id']}/{vita['root']}/{path}")[0] == 400, path


def _mastered(folder: Path, out_dir: str, take: str, lufs: float = -14.0) -> None:
    """A master and its report, as a run with out_dir would have left them."""
    out = folder / out_dir
    _wav(out / f"{take}.wav")
    (out / "reports").mkdir(parents=True, exist_ok=True)
    (out / "reports" / f"{take}.json").write_text(json.dumps({
        "original": {"lra_lu": 6.0}, "trim": {"start_s": 1.0},
        "processed": {"integrated_lufs": lufs, "lra_lu": 4.0, "true_peak_dbtp": -1.0},
        "finalize": {"limiter": {"limited_pct": 5.0}}}))


def test_a_blind_comparisons_verdict_is_kept_with_the_project_and_counted_while_its_masters_stand(app, tmp_path):
    a, client = app
    p = _project(client, tmp_path / "band")
    folder, take = tmp_path / "band", "260928_231324_TrLR Vita"
    dirs = {x["name"]: x["output_dir"] for x in client.ok("GET", f"/api/projects/{p['id']}")["profiles"] if "output_dir" in x}
    for name in ("default", "albini"):
        _mastered(folder, dirs[name], take)
    url = f"/api/projects/{p['id']}/verdicts"
    assert client.ok("GET", f"/api/projects/{p['id']}")["listening"]["verdicts"] == []

    # only two profiles that both have a master of the take, and a choice that is one of them
    for bad in ({"take": take, "a": "default", "b": "default", "choice": "default"},
                {"take": take, "a": "default", "b": "indie", "choice": "default"},
                {"take": take, "a": "default", "b": "albini", "choice": "indie"},
                {"take": "no such take", "a": "default", "b": "albini", "choice": "albini"},
                {"take": take, "a": "default", "b": "albini", "choice": "albini", "loop_s": [30.0, 10.0]}):
        assert client.call("POST", url, bad)[0] == 400, bad
    assert not (folder / "listening.json").exists()

    client.ok("POST", url, {"take": take, "a": "albini", "b": "default", "choice": "albini", "note": " more air ", "loop_s": [61.52, 74.0]})
    client.ok("POST", url, {"take": take, "a": "default", "b": "albini", "choice": "albini"})
    heard = client.ok("POST", url, {"take": take, "a": "default", "b": "albini", "choice": "same"})
    assert [v["choice"] for v in heard["verdicts"]] == ["same", "albini", "albini"]  # the newest first
    first = heard["verdicts"][-1]
    assert (first["a"], first["b"], first["note"], first["loop_s"], first["stale"]) == ("albini", "default", "more air", [61.5, 74.0], False)
    assert heard["pairs"] == [{"profiles": ["albini", "default"], "rounds": 3, "wins": {"albini": 2, "default": 0}, "same": 1, "takes": 1}]
    on_disk = json.loads((folder / "listening.json").read_text())["verdicts"]  # in the project's folder, oldest first
    assert [v["id"] for v in on_disk] == [v["id"] for v in reversed(heard["verdicts"])] and "stale" not in on_disk[0]

    # a verdict pressed by mistake is taken back
    heard = client.ok("DELETE", f"{url}/{heard['verdicts'][0]['id']}")
    assert [v["choice"] for v in heard["verdicts"]] == ["albini", "albini"] and heard["pairs"][0]["rounds"] == 2
    assert client.call("DELETE", f"{url}/nothing")[0] == 400

    # one of the two masters is made again: what was heard was the old one, so it is shown and not counted
    report = folder / dirs["albini"] / "reports" / f"{take}.json"
    os.utime(report, (report.stat().st_atime, report.stat().st_mtime + 60))
    heard = client.ok("GET", f"/api/projects/{p['id']}")["listening"]
    assert [v["stale"] for v in heard["verdicts"]] == [True, True] and heard["pairs"] == []

    # a file that is not ours to read is said, and left alone
    (folder / "listening.json").write_text("not json")
    d = client.ok("GET", f"/api/projects/{p['id']}")
    assert "cannot be read" in d["listening"]["error"] and d["takes"]
    assert client.call("POST", url, {"take": take, "a": "default", "b": "albini", "choice": "albini"})[0] == 400
    assert (folder / "listening.json").read_text() == "not json"


def test_the_form_writes_the_settings_file_and_keeps_its_comments(app, tmp_path):
    _, client = app
    rec = tmp_path / "rec"
    _recordings(rec)
    cfg_path = rec / "rehearsal.toml"
    cfg_path.write_text(SETTINGS)
    p = client.ok("POST", "/api/projects", {"name": "R", "folder": str(rec)})
    url = f"/api/projects/{p['id']}/settings"

    fields = {f["key"]: f for g in client.ok("GET", url)["groups"] for f in g["fields"]}
    assert (fields["loudness.target_lufs"]["value"], fields["loudness.target_lufs"]["set"]) == (-14.0, True)
    assert (fields["mix.original_share"]["value"], fields["mix.original_share"]["set"]) == (0.25, False)

    s = client.ok("POST", url, {"values": {"loudness.target_lufs": -16, "balance.offsets.vocals": 1.5, "output.mp3": None,
                                           "tracks.guitar.width": 0.8, "output.artist": "Circhi"},
                                "run": ["default", "simple_bass"]})
    text = cfg_path.read_text()
    assert "# The band's settings." in text and "# where streaming plays" in text and '# artist = "Circhi"' in text
    cfg = config.load(cfg_path)
    assert cfg["loudness"]["target_lufs"] == -16.0 and cfg["balance"]["offsets"]["vocals"] == 1.5
    assert cfg["balance"]["offsets"]["drums"] == 0.0  # the rest of the table keeps its defaults
    assert cfg["tracks"]["guitar"]["width"] == 0.8 and cfg["output"]["artist"] == "Circhi"
    assert cfg["output"]["mp3"] is True and "mp3 = false" not in text  # back to the default: the key is gone
    assert cfg["input"]["presets"] == ["", "simple_bass"] and s["run"] == ["default", "simple_bass"]
    assert (rec / "rehearsal.toml.bak").read_text() == SETTINGS  # the previous version is kept


def test_what_does_not_load_is_not_saved(app, tmp_path):
    _, client = app
    rec = tmp_path / "rec"
    _recordings(rec)
    cfg_path = rec / "rehearsal.toml"
    cfg_path.write_text(SETTINGS)
    p = client.ok("POST", "/api/projects", {"name": "R", "folder": str(rec)})
    url = f"/api/projects/{p['id']}/settings"
    for body, says in (({"toml": "[loudness]\ntarget_lufz = -14.0\n"}, "unknown setting 'loudness.target_lufz'"),
                       ({"toml": "[loudness\n"}, ""),
                       ({"values": {"loudness.target_lufs": 3}}, "between"),
                       ({"values": {"stems.backend": "x"}}, "not a setting of the form"),
                       ({"run": ["nope"]}, "unknown preset 'nope'"),
                       ({"run": []}, "at least one profile")):
        status, data, _ = client.call("POST", url, body)
        assert status == 400 and says in data["error"], data
        assert cfg_path.read_text() == SETTINGS
    assert sorted(f.name for f in rec.iterdir() if f.is_file()) == ["260928_231324_TrLR Vita.WAV", "rehearsal.toml"]


def test_a_profile_of_the_projects_own_is_a_preset_the_command_line_finds(app, tmp_path):
    _, client = app
    rec = tmp_path / "rec"
    p = _project(client, rec)
    cfg_path = rec / "rehearsal.toml"
    url = f"/api/projects/{p['id']}/profiles"

    start = client.ok("GET", f"{url}/loud?new=1")
    assert 'dir = "processed/loud"' in start["text"] and not (rec / "presets").exists()
    copy = client.ok("GET", f"{url}/drier?new=1&copy=albini")["text"]
    assert 'dir = "processed/drier"' in copy and "Albini" in copy

    client.ok("POST", f"{url}/loud", {"text": start["text"] + "\n[loudness]\ntarget_lufs = -11.0\n"})
    assert "loud" in config.presets(cfg_path) and "loud" not in config.presets()
    cfg = config.load(cfg_path, "loud")
    assert cfg["loudness"]["target_lufs"] == -11.0 and cfg["output"]["dir"] == "processed/loud"
    listed = {x["name"]: x for x in client.ok("GET", f"/api/projects/{p['id']}")["profiles"]}
    assert listed["loud"]["builtin"] is False and listed["albini"]["builtin"] is True

    for name, text, says in (("loud", "[loudness]\ntarget = 1\n", "unknown setting"), ("Bad Name", "", "name"), ("default", "", "name")):
        status, data, _ = client.call("POST", f"{url}/{name.replace(' ', '%20')}", {"text": text})
        assert status == 400 and says in data["error"], data
    assert client.call("GET", f"{url}/loud?new=1")[0] == 400  # the name is taken
    assert client.call("DELETE", f"{url}/albini")[0] == 400  # a built-in one cannot be deleted
    client.ok("DELETE", f"{url}/loud")
    assert "loud" not in config.presets(cfg_path)


def test_a_profile_is_edited_in_a_form_and_a_built_in_one_as_the_projects_version(app, tmp_path):
    _, client = app
    rec = tmp_path / "rec"
    p = _project(client, rec)  # albini is one of the profiles it runs
    cfg_path = rec / "rehearsal.toml"
    url = f"/api/projects/{p['id']}/profiles/albini"
    builtin = (config.PRESETS_DIR / "albini.toml").read_text()

    shown = client.ok("GET", url)
    fields = {f["key"]: f for g in shown["groups"] for f in g["fields"]}
    assert (shown["builtin"], shown["own"], shown["text"]) == (True, False, builtin)
    target = fields["loudness.target_lufs"]  # the profile's value, and the project's as what it goes back to
    assert (target["value"], target["default"], target["set"]) == (-18.0, -14.0, True)
    assert fields["output.mp3"]["set"] is False

    # the form's values go on the text, comments kept; the built-in file is not touched
    saved = client.ok("POST", url, {"text": builtin + "\n# ours\n", "values": {"loudness.target_lufs": -16, "tracks.guitar.width": None,
                                                                                "balance.offsets.vocals": 1.0}})
    own = rec / "presets" / "albini.toml"
    assert (saved["builtin"], saved["own"]) == (True, True) and saved["text"] == own.read_text()
    assert "# ours" in saved["text"] and "Steve Albini" in saved["text"] and (config.PRESETS_DIR / "albini.toml").read_text() == builtin
    cfg = config.load(cfg_path, "albini")  # the command line finds the project's version first
    assert cfg["loudness"]["target_lufs"] == -16.0 and cfg["balance"]["offsets"]["vocals"] == 1.0
    assert cfg["tracks"]["guitar"]["width"] == config.load(cfg_path)["tracks"]["guitar"]["width"]  # back to the project's
    assert cfg["glue"]["enabled"] is False  # what the form does not show is still the built-in one's
    listed = {x["name"]: x for x in client.ok("GET", f"/api/projects/{p['id']}")["profiles"]}
    assert (listed["albini"]["builtin"], listed["albini"]["own"], listed["indie"]["own"]) == (True, True, False)

    for body, says in (({"text": builtin, "values": {"loudness.target_lufs": 40}}, "between"),
                       ({"text": builtin, "values": {"stems.backend": "x"}}, "not a setting of the form"),
                       ({"text": "[loudness\n", "values": {"output.mp3": False}}, "")):
        status, data, _ = client.call("POST", url, body)
        assert status == 400 and says in data["error"], data
        assert own.read_text() == saved["text"]

    client.ok("DELETE", url)  # back to the built-in one, though it runs by default: it is still there
    assert not own.exists() and config.load(cfg_path, "albini")["loudness"]["target_lufs"] == -18.0


def _flac(path: Path, seconds: float = 0.5) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, np.zeros((int(seconds * FS), 2), dtype=np.float32), FS, subtype="PCM_24", format="FLAC")


def test_a_takes_files_are_laid_out_along_the_pipeline(app, tmp_path):
    """What a run leaves (stages.py, the stem caches, the master) as the take's page gets it: every
    file in its stage and row, with what the page needs to play them at the same moment and level."""
    _, client = app
    rec = tmp_path / "rec"
    p = _project(client, rec)
    take = "261105_225442 Miramare"
    url = f"/api/projects/{p['id']}/takes/{take.replace(' ', '%20')}/stages"

    m = client.ok("GET", url)  # nothing run yet: what was recorded, and every stage still to fill
    assert [(n["id"], n["label"], n["main"]) for n in m["nodes"]] == [
        ("recording/room", "TrMic", True), ("recording/Tr2", "Tr2", False), ("recording/mixdown", "TrLR", False)]
    assert (m["mastered"], m["stage_dir"], m["rows"], m["profile"]) == (False, None, [], "default")
    assert [c["key"] for c in m["columns"]] == ["recording", "separated", "cleaned", "chain", "drum_bus", "premix", "mix", "master"]

    cache = rec / (config.load(None)["output"]["stems_dir"] + separate.cache_name(config.load(None)["stems"]))
    for stem in ("vocals", "bass", "kick", "snare"):
        _wav(cache / take / f"{stem}.wav")
    _wav(cache / take / "_work.wav")
    (cache / take / "key.json").write_text("{}")
    _flac(cache / take / "models" / "1_mix" / "some_model" / "vocals.flac")
    (cache / take / "models" / "index.json").write_text(json.dumps({"files": [
        {"file": "1_mix/some_model/vocals.flac", "step": 1, "input": "mix", "weight": 0.65, "level_db": -12.0},
        {"file": "1_mix/some_model/gone.flac", "step": 1, "input": "mix"}]}))
    m = client.ok("GET", url)  # separated only: the stems from their cache
    assert [r["key"] for r in m["rows"]] == ["vocals", "bass", "kick", "snare"]  # the kit last
    assert [n["id"] for n in m["nodes"] if n["col"] == "separated"] == ["separated/vocals", "separated/bass", "separated/kick", "separated/snare"]
    assert [(n["label"], n["group"], n["level_db"], n["whole"]) for n in m["extras"]] == [
        ("some_model: vocals", "Separation of the room track, step 1: from the mix", -12.0, True)]
    assert "short" not in m["nodes"][-1]  # no close track's stems: a pad needs no name

    out = rec / "processed"
    stage = out / "stages" / take
    writer = stages.StageWriter(stage, "flac")
    writer.link_raw(cache / take)
    for d, name in ((cache / f"{take} Tr2", "1_raw_Tr2"), (cache / "merged" / take, "1_merged")):
        d.mkdir(parents=True)
        writer.link_raw(d, name)
    _wav(cache / f"{take} Tr2" / "vocals.wav")
    _wav(cache / f"{take} Tr2" / "drums.wav")
    _wav(cache / "merged" / take / "vocals.wav")  # the close track added: a file of its own
    (cache / "merged" / take / "bass.wav").symlink_to(cache / take / "bass.wav")  # nothing added: the room's
    for rel in ("2_stems/vocals", "2_stems/bass", "2_stems/kick", "2_stems/snare", "2_stems/samples/kick_1", "3_drum_cleanup/kick",
                "3_drum_cleanup/kick_removed", "4_drum_bus", "5_premix/original", "5_premix/vocals", "5_premix/bass",
                "5_premix/drum_bus", "5_mix"):
        _flac(stage / f"{rel}.flac")
    _wav(out / f"{take}.wav")
    (out / f"{take}.mp3").write_bytes(b"x")
    (out / f"{take}_AB.mp3").write_bytes(b"x")
    (out / "reports").mkdir()
    (out / "reports" / f"{take}.json").write_text(json.dumps({
        "trim": {"start_s": 2.5, "end_s": 9.0},
        "multitrack": {"close": {"Tr2": {"vocals": {"level_lu": 0.0, "used": True}, "bass": {"level_lu": -22.0, "used": False}}},
                       "merged": {"vocals": {"tracks": ["Tr2"], "close_share": 0.9, "gain_db": 12.0, "room_gain_db": -3.0}}},
        "tracks": {"drum_bus": {"stems": ["kick", "snare"]}, "kick": {"lufs_out": -18.0}, "vocals": {"balance_gain_db": -2.4},
                   "blend": {"original_share": 0.1}}}))
    _wav(cache / f"{take} Tr2" / "bass.wav")  # on the close track, but only as bleed

    m = client.ok("GET", url + "?profile=default")
    nodes = {n["id"]: n for n in m["nodes"]}
    assert [r["key"] for r in m["rows"]] == ["original", "vocals", "bass", "kick", "snare"]
    assert (m["mastered"], m["stage_dir"], m["trim_s"], m["stage_gain_db"]) == (True, f"processed/stages/{take}", 2.5, stages.GAIN_DB)
    assert m["report"]["tracks"]["kick"]["lufs_out"] == -18.0
    assert sorted(nodes) == sorted([
        "recording/room", "recording/Tr2", "recording/mixdown",
        "separated/vocals", "separated/vocals/room", "separated/vocals/Tr2", "separated/bass", "separated/bass/Tr2", "separated/kick",
        "separated/snare",
        "cleaned/kick", "cleaned/kick/removed", "chain/vocals", "chain/bass", "chain/kick", "chain/kick/sample1", "chain/snare",
        "drum_bus/drum_bus", "premix/original", "premix/vocals", "premix/bass", "premix/drum_bus", "mix/mix", "master/master", "master/ab"])
    # the whole recording or already trimmed, and how far from its real level: what playing two files at the same moment needs
    assert [(nodes[k]["whole"], nodes[k]["level_db"]) for k in ("recording/room", "separated/kick", "cleaned/kick", "chain/kick", "mix/mix",
                                                                 "master/master")] == [(True, 0.0), (True, 0.0), (False, -12.0),
                                                                                       (False, -12.0), (False, -12.0), (False, 0.0)]
    assert nodes["separated/vocals"]["file"].endswith("1_merged/vocals.wav") and nodes["separated/vocals/room"]["main"] is False
    assert nodes["separated/bass"]["file"].endswith("1_raw/bass.wav")  # a link in the merged folder: the room's stem
    # with close tracks every pad of the Separated column has a name, and a merged stem's parts are at their level in it
    assert [(nodes[k].get("short") or nodes[k]["label"], nodes[k]["level_db"]) for k in (
        "separated/vocals", "separated/vocals/room", "separated/vocals/Tr2")] == [("merged", 0.0), ("room", 3.0), ("Tr2", -12.0)]
    assert (nodes["separated/bass"]["short"], nodes["separated/kick"]["short"]) == ("room", "room")
    left_out = nodes["separated/bass/Tr2"]  # bleed on the close track: there to hear, marked as not used
    assert left_out["unused"] is True and "not used" in left_out["what"] and "22 LU" in left_out["what"] and left_out["level_db"] == 0.0
    assert (nodes["drum_bus/drum_bus"]["rows"], nodes["premix/drum_bus"]["rows"], nodes["mix/mix"]["rows"]) == (["kick", "snare"], ["kick", "snare"], "*")
    assert nodes["premix/original"]["rows"] == ["original"] and nodes["chain/kick/sample1"]["sync"] is False
    assert nodes["master/master"]["file"].endswith(".wav") and nodes["master/ab"]["file"].endswith("_AB.mp3")
    assert ("drums", "Close track Tr2") in [(n["label"], n["group"]) for n in m["extras"]]
    for n in m["nodes"] + m["extras"]:  # every one is a file the app serves, the linked stem folders included
        status, _, r = client.call("HEAD", f"/files/{p['id']}/{m['take']['root']}/" + n["file"].replace(" ", "%20"))
        assert status == 200 and r.getheader("Content-Type").startswith("audio/"), n

    # the same as a graph: a box per file or step, an arrow where one goes into the other
    g = m["graph"]
    boxes = {b["id"]: b for b in g["boxes"]}
    arrows = {(a["from"], a["to"]): a for a in g["arrows"]}
    playable = set(nodes) | {n["id"] for n in m["extras"]}
    assert all(a in boxes and b in boxes for a, b in arrows) and all(b["play"] in playable for b in boxes.values() if not b.get("step"))
    assert [boxes[k]["layer"] for k in ("src/room", "sep/room", "stem/kick", "merged/vocals", "cleaned/kick", "chain/kick", "drum_bus/drum_bus",
                                        "premix/vocals", "mix/mix", "master/master")] == list(range(10)) and len(g["layers"]) == 10
    assert boxes["sep/room"]["step"] and boxes["sep/room"]["play"] is None and boxes["stem/kick"]["play"] == "separated/kick"
    assert (arrows["close/Tr2/vocals", "merged/vocals"]["label"], arrows["stem/vocals", "merged/vocals"]["label"]) == ("90 %", "10 %")
    assert boxes["close/Tr2/bass"]["unused"] and ("close/Tr2/bass", "merged/bass") not in arrows and "merged/bass" not in boxes
    for way in (("src/Tr2", "sep/Tr2"), ("sep/room", "stem/bass"), ("stem/bass", "chain/bass"), ("stem/kick", "cleaned/kick"),
                ("cleaned/kick", "chain/kick"), ("chain/kick", "drum_bus/drum_bus"), ("chain/snare", "drum_bus/drum_bus"),
                ("drum_bus/drum_bus", "premix/drum_bus"), ("premix/drum_bus", "mix/mix"), ("mix/mix", "master/master"), ("master/master", "master/ab")):
        assert way in arrows, way
    assert arrows["src/room", "premix/original"]["label"] == "10 %" and arrows["chain/vocals", "premix/vocals"]["label"] == "-2.4 dB"
    assert boxes["stem/kick"]["lane"] == boxes["chain/kick"]["lane"] and boxes["close/Tr2/vocals"]["lane"] < boxes["stem/vocals"]["lane"]

    other = client.ok("GET", url + "?profile=albini")  # not mastered with it: the recording and the stems
    assert not other["mastered"] and {n["col"] for n in other["nodes"]} == {"recording", "separated"}
    assert client.call("GET", url.replace("Miramare", "Nope"))[0] == 400 and client.call("GET", url + "?profile=nope")[0] == 400


EVENTS = r"""
import json, sys, time
def emit(**e):
    print("@@" + json.dumps(e), flush=True)
emit(ev="plan", units=[{"take": "A", "preset": ""}, {"take": "B", "preset": ""}, {"take": "A", "preset": "albini"}])
print("settings: rehearsal.toml", flush=True)
emit(ev="prepare", take="A", i=0, n=1)
emit(ev="separation", track="A", state="start")
emit(ev="separation", track="A", state="step", i=0, n=2, model="bs_roformer")
emit(ev="separation", track="A", state="done")
emit(ev="unit", i=0)
emit(ev="stage", key="stems", label="Stem chains", frac=0.4, detail="bass")
emit(ev="unit_done", i=0, ok=True, seconds=2.0, summary={"processed": {"integrated_lufs": -14.0}})
emit(ev="unit", i=1)
print("Traceback: boom", file=sys.stderr, flush=True)
emit(ev="unit_done", i=1, ok=False, seconds=1.0, error="ValueError: boom")
emit(ev="unit", i=2)
emit(ev="stage", key="bus", label="Mix bus", frac=0.8, detail="")
if "--hang" in sys.argv: time.sleep(60)
emit(ev="unit_done", i=2, ok=True, seconds=2.0, summary={})
emit(ev="end", failed=["B"])
sys.exit("\n1 failed: B")
"""


def test_a_job_is_followed_through_its_events(app, tmp_path):
    a, client = app
    p = _project(client, tmp_path / "rec")
    a.runner.make_command = _stand_in(EVENTS)
    body = {"project": p["id"], "takes": ["261105_225442 Miramare", "260928_231324_TrLR Vita"], "profiles": ["default", "albini"]}
    started = client.ok("POST", "/api/jobs", body)["job"]
    assert [(u["take"], u["profile"]) for u in started["units"]][::3] == [("260928_231324_TrLR Vita", "default"),
                                                                           ("261105_225442 Miramare", "albini")]
    job = _wait(client, lambda j: j["state"] != "running")
    assert job["state"] == "failed" and job["message"] == "1 of 3 failed."
    assert [(u["take"], u["profile"], u["state"]) for u in job["units"]] == [
        ("A", "default", "done"), ("B", "default", "failed"), ("A", "albini", "done")]
    assert job["units"][1]["error"] == "ValueError: boom" and job["units"][0]["summary"]["processed"]["integrated_lufs"] == -14.0
    assert job["frac"] == 1.0 and job["separation"] is None and job["ahead"] is None
    assert "settings: rehearsal.toml" in job["log"] and "Traceback: boom" in job["log"]  # what is not an event is the log
    assert not any(line.startswith("@@") for line in job["log"])
    assert client.ok("GET", f"/api/state?since={job['lines']}")["job"]["log"] == []  # only the lines not seen yet


def test_a_running_job_shows_where_it_is_and_can_be_cancelled(app, tmp_path):
    a, client = app
    p = _project(client, tmp_path / "rec")
    a.runner.make_command = lambda *args: [sys.executable, "-c", EVENTS, "--hang"]
    body = {"project": p["id"], "takes": ["261105_225442 Miramare"], "profiles": ["default"]}
    client.ok("POST", "/api/jobs", body)
    job = _wait(client, lambda j: j["current"] == 2 and j["units"][2]["frac"] == 0.8)
    assert job["state"] == "running" and (job["units"][2]["stage"], job["units"][2]["key"]) == ("Mix bus", "bus")
    assert abs(job["eta"] - 2.0 * (1 - 0.8)) < 1e-6  # the finished take's time, for what is left of this one
    assert abs(job["frac"] - (1 + 1 + 0.8) / 3) < 1e-6

    status, data, _ = client.call("POST", "/api/jobs", body)  # one job at a time
    assert status == 400 and "already running" in data["error"]
    assert client.call("DELETE", f"/api/projects/{p['id']}")[0] == 400

    t0 = time.time()
    job = client.ok("POST", "/api/job/cancel", {})["job"]
    assert time.time() - t0 < 5 and a.runner.job.proc.poll() is not None
    job = _wait(client, lambda j: j["state"] != "running")
    assert job["state"] == "cancelled" and job["units"][2]["state"] == "cancelled"


def test_the_time_left_goes_with_the_takes_lengths():
    job = jobs.Job({"id": "x", "name": "x"}, [{"name": "long", "seconds": 240.0}, {"name": "short", "seconds": 48.0}],
                   ["default", "albini"], False)
    assert job.snapshot()["eta"] is None  # nothing to go by yet
    job.event({"ev": "unit", "i": 0})
    job.event({"ev": "unit_done", "i": 0, "ok": True, "seconds": 300.0})
    job.event({"ev": "unit", "i": 1})
    job.event({"ev": "stage", "key": "bus", "label": "Mix bus", "frac": 0.5})
    # 300 s for 240 s of audio: 1.25 s per second; half the short take, then both again
    assert abs(job.snapshot()["eta"] - 1.25 * (0.5 * 48 + 240 + 48)) < 1e-6


def test_a_job_is_checked_before_it_starts_and_says_why_a_run_died(app, tmp_path):
    a, client = app
    rec = tmp_path / "rec"
    p = _project(client, rec)
    seen = []
    a.runner.make_command = lambda *args: seen.append(args) or [sys.executable, "-c", "import sys; sys.exit('no audio files found')"]
    take = "261105_225442 Miramare"
    (rec / "presets").mkdir()
    (rec / "presets" / "same.toml").write_text("[loudness]\ntarget_lufs = -12.0\n")  # no folder of its own
    for body, says in (({"takes": [], "profiles": ["default"]}, "at least one take"),
                       ({"takes": [take], "profiles": []}, "at least one profile"),
                       ({"takes": ["gone"], "profiles": ["default"]}, "not in the project"),
                       ({"takes": [take], "profiles": ["nope"]}, "no profile 'nope'"),
                       ({"takes": [take], "profiles": ["default", "same"]}, "both write to processed")):
        status, data, _ = client.call("POST", "/api/jobs", {"project": p["id"]} | body)
        assert status == 400 and says in data["error"], data
    assert not seen

    client.ok("POST", "/api/jobs", {"project": p["id"], "takes": [take], "profiles": ["default", "albini"], "ab": True})
    cfg_path, rooms, profs, ab, separate_only = seen[0]
    assert cfg_path == rec / "rehearsal.toml" and rooms == [str(rec / take / "261105_225442_TrMic.WAV")]
    assert (profs, ab, separate_only) == (["default", "albini"], True, False)
    assert jobs.command(cfg_path, rooms, profs, ab, False)[3:] == [rooms[0], "-c", str(cfg_path), "--preset", "default",
                                                                   "--preset", "albini", "--ab"]
    job = _wait(client, lambda j: j["state"] != "running")
    assert job["state"] == "failed" and job["message"] == "no audio files found"


def test_only_this_machines_own_page_is_answered(app, tmp_path):
    _, client = app
    assert client.call("GET", "/api/state", headers={"Host": "evil.example:8377"})[0] == 403  # DNS rebinding
    assert client.call("GET", "/", headers={"Host": "localhost:1"})[0] == 200
    body = {"name": "x", "folder": str(tmp_path)}
    assert client.call("POST", "/api/projects", body, {"Origin": "http://evil.example"})[0] == 403  # another site's page
    assert client.call("POST", "/api/projects", body, {"Content-Type": "text/plain"})[0] == 403    # a plain form post
    assert client.ok("GET", "/api/state")["projects"] == []
    assert client.call("GET", "/static/app.js")[0] == 200 and client.call("GET", "/static/..%2Fserver.py")[0] == 404
    status, guide, _ = client.call("GET", "/static/help.html")  # the guide is a page of the app's own
    assert status == 200 and b"How it works" in guide and b'href="#/help"' in client.call("GET", "/")[1]


def test_a_connection_the_browser_drops_is_not_reported_as_an_error(app, capfd):
    """A player that changes file resets the connection it was reading the old one on."""
    _, client = app
    for _ in range(3):
        sock = socket.create_connection(("127.0.0.1", client.port))
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))  # closing sends a reset
        sock.close()
    time.sleep(0.3)
    assert client.ok("GET", "/api/state")["projects"] == []
    assert capfd.readouterr().err == ""


def test_the_command_line_says_where_it_is(tmp_path, monkeypatch, capsys):
    """A real run, two profiles (one of the project's own), with the progress events on: the plan, every take's
    stages in order with fractions that only rise, and how each ended."""
    mix, stems = tp._song()
    sf.write(tmp_path / tp.NAME, mix.astype(np.float32), tp.FS, subtype="FLOAT")
    none = str(tmp_path / "no_samples")
    (tmp_path / "rehearsal.toml").write_text(
        f'[output]\nmp3 = false\n[track_defaults]\naugment = {{ library = {{ dir = "{none}" }} }}\n'
        f'cymbal_augment = {{ dir = "{none}" }}\n')
    (tmp_path / "presets").mkdir()
    (tmp_path / "presets" / "plain.toml").write_text('[output]\ndir = "processed/plain"\n[stems]\nenabled = false\n')
    tp._fixture_separation(monkeypatch, mix, stems, config.load(None)["highpass"])
    monkeypatch.setenv("REHEARSAL_PROGRESS", "1")

    cli.main([str(tmp_path / tp.NAME), "--preset", "default", "--preset", "plain"])

    out = capsys.readouterr().out.splitlines()
    events = [json.loads(line[2:]) for line in out if line.startswith(progress.PREFIX)]
    assert any(line.strip().startswith("limiter") for line in out)  # the usual summary is still printed
    name = Path(tp.NAME).stem
    assert events[0] == {"ev": "plan", "units": [{"take": name, "preset": ""}, {"take": name, "preset": "plain"}], "separate_only": False}
    assert events[-1] == {"ev": "end", "failed": []}
    keys = [k for k, _, _ in progress.STAGES]
    for i, with_stems in ((0, True), (1, False)):
        a = events.index({"ev": "unit", "i": i})
        b = next(k for k, e in enumerate(events) if e["ev"] == "unit_done" and e["i"] == i)
        stages = [e for e in events[a:b] if e["ev"] == "stage"]
        fracs = [e["frac"] for e in stages]
        assert fracs == sorted(fracs) and fracs[0] == 0.0 and 0.9 < fracs[-1] < 1.0
        seen = list(dict.fromkeys(e["key"] for e in stages))
        assert seen == [k for k in keys if with_stems or k not in ("profile", "stems", "drum_bus", "blend")]
        done = events[b]
        assert done["ok"] and abs(done["summary"]["processed"]["integrated_lufs"] + 14.0) < 0.1
        assert Path(done["summary"]["outputs"]["wav"]).parent == tmp_path / ("processed/plain" if i else "processed")
    chains = [e["detail"] for e in events if e["ev"] == "stage" and e["key"] == "stems"]
    assert {"kick", "bass", "vocals"} <= set(chains)
    assert progress.fraction("read") == 0.0 and abs(progress.fraction("report", 1.0) - 1.0) < 1e-9
