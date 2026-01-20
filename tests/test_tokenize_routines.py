from pathlib import Path

import numpy as np
from anticipation import config
from anticipation.convert import midi_to_compound
from anticipation.tokenize import maybe_tokenize, extract_spans
import anticipation.ops as ops
import anticipation.vocab as v

from tests.conftest import get_tokens_from_midi_file, TestConfigPatcher, patch_config_and_reload

def get_note_instrument_token(instrument_midi_code: int, note_midi_code: int) -> int:
    return v.NOTE_OFFSET + (config.MAX_PITCH * instrument_midi_code + note_midi_code)

def test_patch_config_at_test_time(patch_config_and_reload: TestConfigPatcher) -> None:
    # this is an edit that makes no sense, we wouldn't set this to 0
    patch_config_and_reload(COMPOUND_SIZE=0)
    assert config.COMPOUND_SIZE == 0

def test_config_patch_rolled_back() -> None:
    # test that changes from test patching is isolated
    assert config.COMPOUND_SIZE == 5

def test_extract_span_and_anticipate_for_small_sequence(
    c_major_midi_path: Path,
    patch_config_and_reload: TestConfigPatcher
) -> None:
    padding_density = 100
    anticipation_interval_seconds = 1
    patch_config_and_reload(
        MIN_TRACK_EVENTS=0,
        MIN_TRACK_TIME_IN_SECONDS=0,
        DELTA=anticipation_interval_seconds,
    )
    assert config.TIME_RESOLUTION == 100
    midi_preprocess_token_list: list[int] = midi_to_compound(str(c_major_midi_path.absolute()))
    all_events, truncations, status = maybe_tokenize(midi_preprocess_token_list)
    end_time = ops.max_time(all_events, seconds=False)

    # span augmentation
    np.random.seed(1)
    events, controls, spans = extract_spans(all_events, 0.2)
    assert spans == [
        (269, 400),
        (1087, 1200),
        (1250, 1350),
        (1530, 1650)
    ]
    # spans are
    # start time: time of first event not within a span + offset ~ Exp(lambda)
    # end time: time of first event within the span + delta (the anticipation interval)
    assert spans == [
        (
            # d: 2.6980
            int(config.TIME_RESOLUTION * 2.69),
            # 300 is the earliest event time after 269 (within the span)
            300 + anticipation_interval_seconds * config.TIME_RESOLUTION
        ),
        (
            # 350 is the time of the first event where we escape a span, d: 6.3706
            350 + anticipation_interval_seconds * config.TIME_RESOLUTION + int(config.TIME_RESOLUTION * 6.37),
            # 1100 is the time of the first event within the span
            1100 + anticipation_interval_seconds * config.TIME_RESOLUTION
        ),
        (
            # d: 0.000571 --> 0
            1150 + anticipation_interval_seconds * config.TIME_RESOLUTION + int(config.TIME_RESOLUTION * 0.0),
            # 1250 is the time of the first event within the span
            1250 + anticipation_interval_seconds * config.TIME_RESOLUTION
        ),
        (
            # d: 1.80006 --> 1.8
            1250 + anticipation_interval_seconds * config.TIME_RESOLUTION + int(config.TIME_RESOLUTION * 1.8),
            # 1550 is the time of the first event within the span
            1550 + anticipation_interval_seconds * config.TIME_RESOLUTION
        )
    ]

    assert len(events) == 66
    assert len(controls) == 21
    assert len(all_events) == len(events) + len(controls) == 87

    events = ops.pad(events, end_time, density=padding_density)
    assert len(events) == 66 + (3 * 3)
    assert events == [
        0, 10050, get_note_instrument_token(0, 36), # C1
        50, 10050, get_note_instrument_token(0, 38), # D1
        100, 10050, get_note_instrument_token(0, 40), # E1
        150, 10050, get_note_instrument_token(0, 41), # F1
        200, 10050, get_note_instrument_token(0, 43), # G1
        250, 10050, get_note_instrument_token(0, 45), # A1
        # span: 269-400
        250 + padding_density, v.DUR_OFFSET, v.REST,
        450, 10050, get_note_instrument_token(0, 50), # D2
        500, 10050, get_note_instrument_token(0, 52), # E2
        550, 10050, get_note_instrument_token(0, 53), # F2
        600, 10050, get_note_instrument_token(0, 55), # G2
        650, 10050, get_note_instrument_token(0, 57), # A2
        700, 10050, get_note_instrument_token(0, 59), # B2
        750, 10050, get_note_instrument_token(0, 60), # C3
        850, 10050, get_note_instrument_token(0, 62), # D3
        900, 10050, get_note_instrument_token(0, 64), # E3
        950, 10050, get_note_instrument_token(0, 65), # F3
        1000, 10050, get_note_instrument_token(0, 67), # G3
        1050, 10050, get_note_instrument_token(0, 69), # A3
        # span: 1087-1200
        1050 + padding_density, v.DUR_OFFSET, v.REST,
        # span: 1250-1350
        1050 + padding_density + padding_density, v.DUR_OFFSET, v.REST,
        1350, 10050, get_note_instrument_token(0, 77), # F4
        1400, 10050, get_note_instrument_token(0, 79), # G4
        1450, 10050, get_note_instrument_token(0, 81), # A4
        1500, 10050, get_note_instrument_token(0, 83), # B4
        # span: 1530-1650
    ]
    assert controls == [
        # Q: shouldn't it instead be v.ANOTE_OFFSET + get_note_instrument_token(...)?
        # time here is the original time the note occurred in the sequence
        v.ATIME_OFFSET + 300, v.ADUR_OFFSET + 50, v.CONTROL_OFFSET + get_note_instrument_token(0, 47), # B1
        v.ATIME_OFFSET + 350, v.ADUR_OFFSET + 50, v.CONTROL_OFFSET + get_note_instrument_token(0, 48), # C2
        # ...
        v.ATIME_OFFSET + 1100, v.ADUR_OFFSET + 50, v.CONTROL_OFFSET + get_note_instrument_token(0, 71), # B3
        v.ATIME_OFFSET + 1150, v.ADUR_OFFSET + 50, v.CONTROL_OFFSET + get_note_instrument_token(0, 72), # C4
        v.ATIME_OFFSET + 1250, v.ADUR_OFFSET + 50, v.CONTROL_OFFSET + get_note_instrument_token(0, 74), # D4
        v.ATIME_OFFSET + 1300, v.ADUR_OFFSET + 50, v.CONTROL_OFFSET + get_note_instrument_token(0, 76), # E4
        # ...
        v.ATIME_OFFSET + 1550, v.ADUR_OFFSET + 50, v.CONTROL_OFFSET + get_note_instrument_token(0, 84), # C4
    ]

    # implicitly: delta = (anticipation_interval_seconds * config.TIME_RESOLUTION),
    # which we have set to delta = (100)
    tokens, controls = ops.anticipate(events, controls)
    # this function returns unconsumed controls - expect to use all of them
    assert len(controls) == 0
    assert len(tokens) == 96

    # anticipation interleaves controls and events s.t. controls appear after
    # stopping times
    assert tokens == [
        0, v.DUR_OFFSET + 50, get_note_instrument_token(0, 36),  # C1
        50, 10050, get_note_instrument_token(0, 38),   # D1
        100, 10050, get_note_instrument_token(0, 40),  # E1
        150, 10050, get_note_instrument_token(0, 41),  # F1
        200, 10050, get_note_instrument_token(0, 43),  # G1
        # B1 approx 100 ticks before its time
        v.ATIME_OFFSET + 300, v.ADUR_OFFSET + 50, v.CONTROL_OFFSET + get_note_instrument_token(0, 47),  # B1
        250, 10050, get_note_instrument_token(0, 45),  # A1
        # C2 approx 100 ticks before its time
        v.ATIME_OFFSET + 350, v.ADUR_OFFSET + 50, v.CONTROL_OFFSET + get_note_instrument_token(0, 48),  # C2
        250 + padding_density, v.DUR_OFFSET, v.REST,
        450, 10050, get_note_instrument_token(0, 50),  # D2
        500, 10050, get_note_instrument_token(0, 52),  # E2
        550, 10050, get_note_instrument_token(0, 53),  # F2
        600, 10050, get_note_instrument_token(0, 55),  # G2
        650, 10050, get_note_instrument_token(0, 57),  # A2
        700, 10050, get_note_instrument_token(0, 59),  # B2
        750, 10050, get_note_instrument_token(0, 60),  # C3
        850, 10050, get_note_instrument_token(0, 62),  # D3
        900, 10050, get_note_instrument_token(0, 64),  # E3
        950, 10050, get_note_instrument_token(0, 65),  # F3
        1000, 10050, get_note_instrument_token(0, 67), # G3
        # B3 approx 100 ticks before its time
        v.ATIME_OFFSET + 1100, v.ADUR_OFFSET + 50, v.CONTROL_OFFSET + get_note_instrument_token(0, 71), # B3
        1050, 10050, get_note_instrument_token(0, 69), # A3
        # C4 approx 100 ticks before its time
        v.ATIME_OFFSET + 1150, v.ADUR_OFFSET + 50, v.CONTROL_OFFSET + get_note_instrument_token(0, 72), # C4
        1050 + padding_density * 1, v.DUR_OFFSET, v.REST,
        # D4 approx 100 ticks before its time
        v.ATIME_OFFSET + 1250, v.ADUR_OFFSET + 50, v.CONTROL_OFFSET + get_note_instrument_token(0, 74), # D4
        1050 + padding_density * 2, v.DUR_OFFSET, v.REST,
        v.ATIME_OFFSET + 1300, v.ADUR_OFFSET + 50, v.CONTROL_OFFSET + get_note_instrument_token(0, 76), # E4
        1350, 10050, get_note_instrument_token(0, 77),  # F4
        1400, 10050, get_note_instrument_token(0, 79),  # G4
        1450, 10050, get_note_instrument_token(0, 81),  # A4
        v.ATIME_OFFSET + 1550, v.ADUR_OFFSET + 50, v.CONTROL_OFFSET + get_note_instrument_token(0, 84), # C4
        1500, 10050, get_note_instrument_token(0, 83),  # B4
    ]

def test_tokenization_small_sequence_ar(c_major_midi_path: Path, patch_config_and_reload: TestConfigPatcher) -> None:
    m = 10
    event_size = 3
    augment_factor = 1
    patch_config_and_reload(MIN_TRACK_EVENTS=0, MIN_TRACK_TIME_IN_SECONDS=0, M=m)
    parse_info = get_tokens_from_midi_file(
        c_major_midi_path,
        augment_factor=augment_factor,
        return_original_compound=True
    )
    compound = parse_info["compound"]
    sequences = parse_info["midi_tokens"]

    assert len(compound) == 145
    assert parse_info["seqcount"] == 3
    assert len(sequences) == parse_info["seqcount"]

    # check each sequence
    seq_1 = sequences[0]
    seq_2 = sequences[1]
    seq_3 = sequences[2]
    assert len(seq_1) == (1 + (m * event_size))
    assert len(seq_2) == (1 + (m * event_size))
    assert len(seq_3) == (1 + (m * event_size))

    # 50 is quarter note in 4/4 120 BPM
    assert seq_1 == [
        v.AUTOREGRESS,
        v.SEPARATOR, v.SEPARATOR, v.SEPARATOR,
        # (time, duration, instrument x note_value)
        v.TIME_OFFSET + 0, v.DUR_OFFSET + 50, get_note_instrument_token(0, 36), # C1
        v.TIME_OFFSET + 50, v.DUR_OFFSET + 50, get_note_instrument_token(0, 38), # D1
        v.TIME_OFFSET + 100, v.DUR_OFFSET + 50, get_note_instrument_token(0, 40), # E1
        v.TIME_OFFSET + 150, v.DUR_OFFSET + 50, get_note_instrument_token(0, 41), # F1
        v.TIME_OFFSET + 200, v.DUR_OFFSET + 50, get_note_instrument_token(0, 43), # G1
        v.TIME_OFFSET + 250, v.DUR_OFFSET + 50, get_note_instrument_token(0, 45), # A1
        v.TIME_OFFSET + 300, v.DUR_OFFSET + 50, get_note_instrument_token(0, 47), # B1
        v.TIME_OFFSET + 350, v.DUR_OFFSET + 50, get_note_instrument_token(0, 48), # C2
        v.TIME_OFFSET + 450, v.DUR_OFFSET + 50, get_note_instrument_token(0, 50), # D2
    ]
    assert seq_2 == [
        v.AUTOREGRESS,
        # (time, duration, instrument x note_value)
        v.TIME_OFFSET + 0, v.DUR_OFFSET + 50, get_note_instrument_token(0, 52),  # E2
        v.TIME_OFFSET + 50, v.DUR_OFFSET + 50, get_note_instrument_token(0, 53),  # F2
        v.TIME_OFFSET + 100, v.DUR_OFFSET + 50, get_note_instrument_token(0, 55),  # G2
        v.TIME_OFFSET + 150, v.DUR_OFFSET + 50, get_note_instrument_token(0, 57),  # A2
        v.TIME_OFFSET + 200, v.DUR_OFFSET + 50, get_note_instrument_token(0, 59),  # B2
        v.TIME_OFFSET + 250, v.DUR_OFFSET + 50, get_note_instrument_token(0, 60),  # C2
        v.TIME_OFFSET + 350, v.DUR_OFFSET + 50, get_note_instrument_token(0, 62),  # D3
        v.TIME_OFFSET + 400, v.DUR_OFFSET + 50, get_note_instrument_token(0, 64),  # E3
        v.TIME_OFFSET + 450, v.DUR_OFFSET + 50, get_note_instrument_token(0, 65),  # F3
        v.TIME_OFFSET + 500, v.DUR_OFFSET + 50, get_note_instrument_token(0, 67),  # G3
    ]
    assert seq_3 == [
        v.AUTOREGRESS,
        # (time, duration, instrument x note_value)
        v.TIME_OFFSET + 0, v.DUR_OFFSET + 50, get_note_instrument_token(0, 69),  # A3
        v.TIME_OFFSET + 50, v.DUR_OFFSET + 50, get_note_instrument_token(0, 71),  # B3
        v.TIME_OFFSET + 100, v.DUR_OFFSET + 50, get_note_instrument_token(0, 72),  # C3
        v.TIME_OFFSET + 200, v.DUR_OFFSET + 50, get_note_instrument_token(0, 74),  # D3
        v.TIME_OFFSET + 250, v.DUR_OFFSET + 50, get_note_instrument_token(0, 76),  # E3
        v.TIME_OFFSET + 300, v.DUR_OFFSET + 50, get_note_instrument_token(0, 77),  # F3
        v.TIME_OFFSET + 350, v.DUR_OFFSET + 50, get_note_instrument_token(0, 79),  # G3
        v.TIME_OFFSET + 400, v.DUR_OFFSET + 50, get_note_instrument_token(0, 81),  # A4
        v.TIME_OFFSET + 450, v.DUR_OFFSET + 50, get_note_instrument_token(0, 83),  # B4
        v.TIME_OFFSET + 500, v.DUR_OFFSET + 50, get_note_instrument_token(0, 84),  # C4
    ]


def test_tokenization_lakh_example(lmd_0_example_midi_path: Path) -> None:
    assert config.M == 341
    parse_info = get_tokens_from_midi_file(lmd_0_example_midi_path, augment_factor=10)
    midi_tokens = parse_info["midi_tokens"]
    seqcount = parse_info["seqcount"]
    assert parse_info["rest_count"] == 185
    assert parse_info["num_too_short"] == 0
    assert parse_info["num_too_long"] == 0
    assert parse_info["num_too_many_instruments"] == 0
    assert parse_info["num_inexpressible"] == 0
    assert parse_info["all_truncations"] == 0
    assert seqcount == 84
    assert len(midi_tokens) == seqcount
