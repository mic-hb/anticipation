"""Pytest common fixtures and utility functions for test suite."""
import tempfile
import importlib
from pathlib import Path
from typing import Callable, Any

import pytest

from anticipation.convert import midi_to_compound
# python itself has a top-level module named `tokenize`
from anticipation.tokenize import tokenize as anticipation_tokenize

TestConfigPatcher = Callable[..., None]

TESTS_ROOT = Path(__file__).parent
TEST_DATA_PATH = TESTS_ROOT / "test_data"

@pytest.fixture
def c_major_midi_path() -> Path:
    return TEST_DATA_PATH / "cmajor.mid"

@pytest.fixture
def lmd_0_example_midi_path() -> Path:
    # from The Lakh MIDI Dataset v0.1 (https://colinraffel.com/projects/lmd/)
    # from split 0, first file in name order
    return TEST_DATA_PATH / "0a0a2b0e4d3b7bf4c5383ba025c4683e.mid"

@pytest.fixture(scope="function")
def patch_config_and_reload(monkeypatch: pytest.MonkeyPatch) -> TestConfigPatcher:
    """
    Pytest fixture for changing the values of:
    - anticipation.config variables during test time

    Any code that imports those values must be reloaded, otherwise the
    changes do not take effect.
    """
    import anticipation.tokenize as anticipation_tokenize_cl
    import anticipation.vocab as anticipation_vocab
    from anticipation import config as anticipation_config
    def _apply(**kwargs):
        for k, v in kwargs.items():
            monkeypatch.setattr(anticipation_config, k, v)
        importlib.reload(anticipation_vocab)
        importlib.reload(anticipation_tokenize_cl)
    yield _apply
    # undo any changes done to the config by reloading it (and things
    # that depend on it) during teardown
    importlib.reload(anticipation_config)
    importlib.reload(anticipation_vocab)
    importlib.reload(anticipation_tokenize_cl)

def _parse_midi_tokenized_text(midi_token_string: str) -> list[list[int]]:
    sequences = midi_token_string.strip().split("\n")
    return [list(map(int, x.split(" "))) for x in sequences]

def get_tokens_from_midi_file(
    midi_file_path: Path,
    augment_factor: int = 10,
    return_original_compound: bool = False,
    do_random_augmentation: bool = True,
) -> dict[str, Any]:
    assert midi_file_path.exists()
    assert midi_file_path.is_file()

    # MIDI -> Text
    midi_preprocess_token_list: list[int] = midi_to_compound(str(midi_file_path.absolute()))
    midi_preprocess_text = ' '.join(str(tok) for tok in midi_preprocess_token_list)

    # Text -> Token
    with tempfile.TemporaryDirectory() as td:
        td_enclosing = Path(td)
        split = "0"
        midi_preprocess_text_fname = td_enclosing / (midi_file_path.stem + ".mid.compound.txt")
        midi_preprocess_text_fname.write_text(midi_preprocess_text)
        output_fname = td_enclosing / f'tokenized-events-{split}.txt'

        # seqcount: number of total tokens in the sequences
        # rest_count: number of total rests
        # all_truncations: number of times a sequence exceeds maximum duration:
        # `truncations = sum([1 for tok in tokens[1::3] if tok >= MAX_DUR])`
        (seqcount, rest_count, num_too_short, num_too_long, num_too_many_instruments, num_inexpressible,
         all_truncations) = anticipation_tokenize(
            [midi_preprocess_text_fname],
            output_fname,
            # 1 = standard AR training
            # 10 = lowest acceptable anticipation augment factor
            augment_factor=augment_factor,
            do_random_augmentation=do_random_augmentation,
        )
        midi_tokens: str = Path(output_fname).read_text()
        parsed_midi_tokens = _parse_midi_tokenized_text(midi_tokens)
        parse_info = {
            "midi_tokens": parsed_midi_tokens,
            "seqcount": seqcount,
            "rest_count": rest_count,
            "num_too_short": num_too_short,
            "num_too_long": num_too_long,
            "num_too_many_instruments": num_too_many_instruments,
            "num_inexpressible": num_inexpressible,
            "all_truncations": all_truncations,
            "compound": None,
        }
        if return_original_compound:
            parse_info["compound"] = midi_preprocess_token_list

        return parse_info