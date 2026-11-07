"""Settings: layers, precedence and track inheritance (config.py)."""

from pathlib import Path

import pytest

from rehearsal_master import config


def _user(tmp_path: Path, text: str) -> Path:
    p = tmp_path / "rehearsal.toml"
    p.write_text(text)
    return p


def test_layers_in_the_documented_order(tmp_path):
    """defaults < rehearsal.toml < preset < song style < [files] < command line."""
    p = _user(tmp_path, '[glue]\ntarget_gr_db = 0.4\n[files."Vita".glue]\ntarget_gr_db = 0.3\n')
    cfg = config.load(p, "albini", {"loudness": {"target_lufs": -12.0}})
    assert cfg["glue"]["target_gr_db"] == 0.4 and cfg["loudness"]["target_lufs"] == -12.0
    wall = cfg["character"]["styles"]["wall"]["glue"]["target_gr_db"]
    assert config.for_file(cfg, "x Miramare.wav", "wall")["glue"]["target_gr_db"] == wall  # style over the file's settings
    vita = config.for_file(cfg, "x Vita.wav", "wall")
    assert vita["glue"]["target_gr_db"] == 0.3  # the song's own section over the style
    assert vita["glue"]["ratio"] == cfg["character"]["styles"]["wall"]["glue"]["ratio"]  # the style's other keys stay
    assert vita["loudness"]["target_lufs"] == -12.0  # the command line last
    assert vita["loudness"]["target_lufs"] != config.load(None, "albini")["loudness"]["target_lufs"]


def test_per_song_track_settings_reach_the_tracks_based_on_it(tmp_path):
    p = _user(tmp_path, '[files."Vita".tracks.guitar]\nsaturator = { mix = 0.77 }\n')
    c = config.for_file(config.load(p), "Vita.wav")
    assert c["tracks"]["guitar"]["saturator"]["mix"] == 0.77
    assert c["tracks"]["guitar_left"]["saturator"]["mix"] == 0.77  # based_on = "guitar"
    assert c["tracks"]["guitar_left"]["width"] == 0.5  # its own settings still win over the base track


def test_for_file_can_be_applied_again(tmp_path):
    p = _user(tmp_path, '[files."Vita".glue]\ntarget_gr_db = 0.3\n')
    cfg = config.load(p)
    once = config.for_file(cfg, "Vita.wav")
    twice = config.for_file(once, "Vita.wav", "ballad")
    assert twice["glue"]["target_gr_db"] == 0.3 and twice["balance"]["offsets"]["vocals"] == 1.5


def test_typos_are_errors_in_every_layer(tmp_path):
    with pytest.raises(KeyError, match="targt"):
        config.load(None, None, {"glue": {"targt": 1.0}})
    with pytest.raises(KeyError, match="Vita"):
        config.load(_user(tmp_path, '[files."Vita".glue]\ntargt = 0.3\n'))
    with pytest.raises(KeyError, match="unknown preset"):
        config.load(None, "nope")
