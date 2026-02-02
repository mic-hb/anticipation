from pathlib import Path

from unittest.mock import patch

from anticipation.v2.config import AnticipationV2Settings, Vocab
from anticipation.v2.tokenize import tokenize as v2_tokenize

from tests.util.entities import Event
from tests.util.visualize_sequence import get_figure_and_open

from tests.conftest import (
    VISUALIZATIONS_PATH,
)
from util.entities import EventSpecialCode


def test_tokenize_v2_lakh_ar_only_for_visualization(
    lmd_0_example_midi_path: Path,
) -> None:
    settings = AnticipationV2Settings(
        vocab=Vocab(),
        num_autoregressive_seq_per_midi_file=1,
        num_instrument_anticipation_augmentations_per_midi_file=0,
        num_span_anticipation_augmentations_per_midi_file=0,
        num_random_anticipation_augmentations_per_midi_file=0,
        debug=True,
    )
    tokens = v2_tokenize([lmd_0_example_midi_path], settings)

    for i in range(0, len(tokens), settings.context_size):
        first_token_in_seq_chunk = tokens[i]
        assert first_token_in_seq_chunk == settings.vocab.AUTOREGRESS

    assert len(tokens) == 8776
    num_seps = len([x for x in tokens if x == settings.vocab.SEPARATOR])
    assert num_seps == 1

    parsed_events = Event.from_token_seq(tokens, settings)
    get_figure_and_open(
        events=parsed_events,
        delta=settings.delta,
        time_resolution=settings.time_resolution,
        path=(VISUALIZATIONS_PATH / f"autoregressive_v2.html"),
        auto_open=True,
    )


def test_tokenize_v2_lakh_instrument_for_visualization(
    lmd_0_example_midi_path: Path,
) -> None:
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
            debug=True,
        )
        instrument_anticipation_sample = v2_tokenize(
            [lmd_0_example_midi_path], settings
        )

    assert len(instrument_anticipation_sample) == 8777
    for i in range(0, len(instrument_anticipation_sample), settings.context_size):
        # sequence must always start with a flag token
        first_token_in_seq_chunk = instrument_anticipation_sample[i]
        assert first_token_in_seq_chunk == settings.vocab.ANTICIPATE

    num_seps = len(
        [x for x in instrument_anticipation_sample if x == settings.vocab.SEPARATOR]
    )
    assert num_seps == 1

    parsed_events = Event.from_token_seq(instrument_anticipation_sample, settings)
    get_figure_and_open(
        events=parsed_events,
        delta=settings.delta,
        time_resolution=settings.time_resolution,
        path=(VISUALIZATIONS_PATH / f"anticipated_instr_v2.html"),
        auto_open=True,
    )


def test_tokenize_with_ticks_for_small_sequence_ar(
    c_major_midi_path: Path,
) -> None:
    settings = AnticipationV2Settings(
        min_track_events=1,
        vocab=Vocab(),
        # AR only
        num_autoregressive_seq_per_midi_file=1,
        num_instrument_anticipation_augmentations_per_midi_file=0,
        num_span_anticipation_augmentations_per_midi_file=0,
        num_random_anticipation_augmentations_per_midi_file=0,
        debug=True,
    )
    tokenized_seq = v2_tokenize([c_major_midi_path], settings)
    num_seps = len([x for x in tokenized_seq if x == settings.vocab.SEPARATOR])
    assert num_seps == 1

    parsed_events = Event.from_token_seq(tokenized_seq, settings)

    # check that the ticks are added at specified interval
    ticks = [x for x in parsed_events if x.is_tick()]
    ticks_abs_times = [x.absolute_time for x in ticks]
    assert ticks_abs_times == list(
        range(0, 1500 + 1, settings.tick_token_frequency_in_midi_ticks)
    )
    get_figure_and_open(
        events=parsed_events,
        delta=settings.delta,
        time_resolution=settings.time_resolution,
        path=(VISUALIZATIONS_PATH / f"ar_with_ticks_small_seq.html"),
        auto_open=True,
    )


def test_tokenize_with_ticks_for_lakh_ar(lmd_0_example_midi_path: Path) -> None:
    settings = AnticipationV2Settings(
        min_track_events=1,
        vocab=Vocab(),
        # AR only
        num_autoregressive_seq_per_midi_file=1,
        num_instrument_anticipation_augmentations_per_midi_file=0,
        num_span_anticipation_augmentations_per_midi_file=0,
        num_random_anticipation_augmentations_per_midi_file=0,
        debug=True,
    )
    tokenized_seq = v2_tokenize([lmd_0_example_midi_path], settings)
    num_seps = len([x for x in tokenized_seq if x == settings.vocab.SEPARATOR])
    assert num_seps == 1

    parsed_events = Event.from_token_seq(tokenized_seq, settings)
    get_figure_and_open(
        events=parsed_events,
        delta=settings.delta,
        time_resolution=settings.time_resolution,
        path=(VISUALIZATIONS_PATH / f"ar_with_ticks_lakh_0.html"),
        auto_open=True,
    )


def test_absolute_time_is_correct_with_ticks(lmd_0_example_midi_path: Path) -> None:
    settings = AnticipationV2Settings(
        min_track_events=1,
        vocab=Vocab(),
        # AR only
        num_autoregressive_seq_per_midi_file=1,
        num_instrument_anticipation_augmentations_per_midi_file=0,
        num_span_anticipation_augmentations_per_midi_file=0,
        num_random_anticipation_augmentations_per_midi_file=0,
        # no ticks
        debug=True,
    )
    # tokenize and parse without ticks added
    events_without_ticks = Event.from_token_seq(
        v2_tokenize([lmd_0_example_midi_path], settings), settings
    )
    events_without_ticks = [
        x
        for x in events_without_ticks
        if x.special_code == EventSpecialCode.TYPICAL_EVENT
    ]

    # tokenize and parse WITH ticks added
    settings.do_add_ticks = True
    events_include_ticks = Event.from_token_seq(
        v2_tokenize([lmd_0_example_midi_path], settings), settings
    )
    events_include_ticks = [
        x
        for x in events_include_ticks
        if x.special_code == EventSpecialCode.TYPICAL_EVENT
    ]

    # assert no information loss and that time de-relativization works
    for a, b in zip(events_without_ticks, events_include_ticks):
        # absolute times must be the same, we should be able to recover
        # the original non-relativized time from the sequence of events that
        # has the ticks
        assert a.absolute_time == b.absolute_time
        assert a.midi_note() == b.midi_note()
        assert a.midi_duration() == b.midi_duration()
        assert a.midi_instrument() == b.midi_instrument()
