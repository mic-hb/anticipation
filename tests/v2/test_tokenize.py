from pathlib import Path

from unittest.mock import patch

from anticipation.v2.config import AnticipationV2Settings, Vocab
from anticipation.v2.tokenize import tokenize as v2_tokenize
from anticipation.v2.ops import is_flag_token

from tests.util.entities import Event
from tests.util.visualize_sequence import get_figure_and_open

from tests.conftest import (
    VISUALIZATIONS_PATH,
)

def test_tokenize_v2_lakh_ar_only(lmd_0_example_midi_path: Path) -> None:
    settings = AnticipationV2Settings(
        vocab=Vocab(),
        num_autoregressive_seq_per_midi_file=1,
        num_instrument_anticipation_augmentations_per_midi_file=0,
        num_span_anticipation_augmentations_per_midi_file=0,
        num_random_anticipation_augmentations_per_midi_file=0,
        do_relativize_time_to_ctx=False,
        debug=True,
    )
    tokens = v2_tokenize([lmd_0_example_midi_path], settings)

    for i in range(0, len(tokens), settings.context_size):
        # sequence must always start with a flag token
        first_token_in_seq_chunk = tokens[i]
        assert is_flag_token(first_token_in_seq_chunk, settings)

    assert len(tokens) == 8598
    parsed_events = Event.from_token_seq(tokens, settings)
    get_figure_and_open(
        events=parsed_events,
        delta=settings.delta,
        time_resolution=settings.time_resolution,
        path=(VISUALIZATIONS_PATH / f"autoregressive_v2.html"),
        auto_open=False,
    )

def test_tokenize_v2_lakh_instrument(lmd_0_example_midi_path: Path) -> None:
    with patch("anticipation.tokenize.np.random.choice", return_value=[128]):
        # force the call to np.random.choice to always return [128] for tokenize.py
        # This means that the instrument code 128 will always be a control. This code
        # is the drum track for this sample. This makes it much easier to see the
        # cold start issue
        settings = AnticipationV2Settings(
            vocab=Vocab(),
            num_autoregressive_seq_per_midi_file=0,
            num_instrument_anticipation_augmentations_per_midi_file=1,
            num_span_anticipation_augmentations_per_midi_file=0,
            num_random_anticipation_augmentations_per_midi_file=0,
            do_relativize_time_to_ctx=False,
            debug=True,
        )
        instrument_anticipation_sample = v2_tokenize([lmd_0_example_midi_path], settings)

    assert len(instrument_anticipation_sample) == 8607
    for i in range(0, len(instrument_anticipation_sample), settings.context_size):
        # sequence must always start with a flag token
        first_token_in_seq_chunk = instrument_anticipation_sample[i]
        assert is_flag_token(first_token_in_seq_chunk, settings)

    parsed_events = Event.from_token_seq(instrument_anticipation_sample, settings)
    get_figure_and_open(
        events=parsed_events,
        delta=settings.delta,
        time_resolution=settings.time_resolution,
        path=(VISUALIZATIONS_PATH / f"anticipated_instr_v2.html"),
        auto_open=True,
    )