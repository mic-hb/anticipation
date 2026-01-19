from pathlib import Path

from conftest import patch_config_and_reload
from tests.conftest import get_tokens_from_midi_file, TestConfigPatcher
from anticipation import config

def test_patch_config_at_test_time(patch_config_and_reload: TestConfigPatcher) -> None:
    # this is an edit that makes no sense, we wouldn't set this to 0
    patch_config_and_reload(COMPOUND_SIZE=0)
    assert config.COMPOUND_SIZE == 0

def test_config_patch_rolled_back() -> None:
    # test that changes from test patching is isolated
    assert config.COMPOUND_SIZE == 5

def test_ar_tokenization(c_major_midi_path: Path, patch_config_and_reload: TestConfigPatcher) -> None:
    # we have 966 total tokens, model context is 1024
    patch_config_and_reload(MIN_TRACK_EVENTS=0, MIN_TRACK_TIME_IN_SECONDS=0, M=256)
    parse_info = get_tokens_from_midi_file(c_major_midi_path, augment_factor=10)
    midi_tokens = parse_info["midi_tokens"]
    assert len(midi_tokens) == 4276
