import tempfile
from pathlib import Path
from unittest.mock import patch

from anticipation.v2.config import AnticipationV2Settings, Vocab
from anticipation.v2.tokenize import tokenize as v2_tokenize

from tests.util.entities import Event, EventSpecialCode
from tests.util.visualize_sequence import get_figure_and_open
from tests.conftest import get_tokens_from_midi_file_v1, get_tokens_from_text_file

from tests.conftest import (
    VISUALIZATIONS_PATH,
)


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
    tokens = []
    any_ignored = v2_tokenize([lmd_0_example_midi_path], tokens, settings)
    assert not any_ignored

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
        auto_open=False,
    )


def test_tokenize_v2_lakh_instrument_for_visualization(
    lmd_0_example_midi_path: Path,
) -> None:
    instrument_anticipation_sample = []
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
        any_ignored = v2_tokenize(
            [lmd_0_example_midi_path], instrument_anticipation_sample, settings
        )
        assert not any_ignored

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
    tokenized_seq = []
    any_ignored = v2_tokenize([c_major_midi_path], tokenized_seq, settings)
    assert not any_ignored
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
        auto_open=False,
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
    tokenized_seq = []
    any_ignored = v2_tokenize([lmd_0_example_midi_path], tokenized_seq, settings)
    assert not any_ignored
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
        debug=True,
    )
    # tokenize and parse without ticks added
    events_without_ticks = []
    any_ignored = v2_tokenize([lmd_0_example_midi_path], events_without_ticks, settings)
    assert not any_ignored
    events_without_ticks = Event.from_token_seq(events_without_ticks, settings)
    events_without_ticks = [
        x
        for x in events_without_ticks
        if x.special_code == EventSpecialCode.TYPICAL_EVENT
    ]

    # tokenize and parse WITH ticks added
    events_include_ticks = []
    any_ignored = v2_tokenize([lmd_0_example_midi_path], events_include_ticks, settings)
    assert not any_ignored
    events_include_ticks = Event.from_token_seq(events_include_ticks, settings)
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


def test_tokenize_write_to_disk_compatible_with_v1_ar_only(
    lmd_0_example_midi_path: Path,
    lmd_1_example_midi_path: Path,
) -> None:
    """
    We've made some changes from v1 to v2 that alter the way instrument augmentation happens, so
    the outputs will not be exactly the same. We should, however, be able to recover the exact
    tokenization implemented by v1 by settings some settings (for autoregressive settings).
    """
    midi_files_to_process = [lmd_0_example_midi_path, lmd_1_example_midi_path]
    settings = AnticipationV2Settings(
        min_track_events=1,
        vocab=Vocab(),
        num_autoregressive_seq_per_midi_file=1,
        num_instrument_anticipation_augmentations_per_midi_file=0,
        num_span_anticipation_augmentations_per_midi_file=0,
        num_random_anticipation_augmentations_per_midi_file=0,
        tick_token_frequency_in_midi_ticks=0,
        num_sep_tokens=3,
        omit_flag_token_after_first_sample=True,
        debug=True,
    )
    # tokenize and parse without ticks added
    with tempfile.TemporaryDirectory() as td:
        v2_sink = Path(td) / "v2_tokens.txt"
        any_ignored = v2_tokenize(midi_files_to_process, v2_sink, settings)
        assert not any_ignored
        parse_info = get_tokens_from_midi_file_v1(
            midi_files_to_process,
            augment_factor=10,
            include_original=True,
            do_instrument_augmentation=False,
            do_span_augmentation=False,
            do_random_augmentation=False,
        )

        v1_tokens = parse_info["tokens"]
        v2_tokens = get_tokens_from_text_file(v2_sink)

        v1 = v1_tokens[8]
        v2 = v2_tokens[8]
        print(v1)
        print(v2)
        print("----")

        # v1 = v1_tokens[9]
        # v2 = v2_tokens[9]
        # print(len(v1), len(v2))
        # print(v1)
        # print(v2)

        assert 1 == 2
        # assert len(v1_tokens) == len(v2_tokens)
        for k in range(len(v1_tokens)):
            assert len(v1_tokens[k]) == len(v2_tokens[k])
            assert v1_tokens[k] == v2_tokens[k], f"Error with index = {k}"
