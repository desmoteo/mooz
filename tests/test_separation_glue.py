"""The parts of audio-separator that separate.py relies on beyond its public use.

separate.py loads models the library does not list ([stems.custom_models]) by swapping
its download step and widening a parameter range, keeps a Separator per loaded model from
one source to the next (telling it each job's output folder) and asks for its compiled blocks.
None of that is a public interface, so the library is pinned (pyproject.toml) and this
file says what must still be true before the pin is raised. It checks names and shapes
only: no model is loaded.
"""

import inspect

import pytest

pytest.importorskip("torch")
separator = pytest.importorskip("audio_separator.separator")
Separator = separator.Separator


def test_separator_takes_the_settings_the_worker_passes():
    params = inspect.signature(Separator.__init__).parameters
    for name in ("log_level", "model_file_dir", "output_dir", "output_format", "use_soundfile", "sample_rate",
                 "use_autocast", "use_torch_compile", "normalization_threshold", "amplification_threshold"):
        assert name in params, name
    # set per model before each load: the segment size, whether it overrides the model's own, the pitch shift
    assert {"segment_size", "override_model_segment_size", "pitch_shift"} <= set(params["mdxc_params"].default)


def test_separator_keeps_the_attributes_the_worker_swaps():
    init, load = inspect.getsource(Separator.__init__), inspect.getsource(Separator.load_model)
    assert 'self.arch_specific_params = {' in init and '"MDXC": mdxc_params' in init  # updated before each load
    assert "self.model_instance = None" in init
    assert "self.model_instance is not None" in load
    # a separator is kept from one source to the next and told each job's folder: both hold it
    assert "self.output_dir = output_dir" in init
    assert "self.model_instance.output_dir = temp_dir" in inspect.getsource(Separator)
    # the swapped download step returns these five, in this order
    assert ("model_filename, model_type, model_friendly_name, model_path, yaml_config_filename = "
            "self.download_model_files(model_filename)") in load
    assert list(inspect.signature(Separator.download_model_files).parameters) == ["self", "model_filename"]
    assert "model_filename" in inspect.signature(Separator.load_model).parameters
    assert "audio_file_path" in inspect.signature(Separator.separate).parameters


def test_roformer_stem_limit_is_where_the_worker_widens_it():
    from audio_separator.separator.roformer.parameter_validator import ParameterValidator

    lo, hi = ParameterValidator.PARAMETER_RANGES["num_stems"]  # widened to 64 for the 53-stem model
    assert lo >= 1 and hi >= lo
