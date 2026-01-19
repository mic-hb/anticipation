from pathlib import Path

from anticipation import config
import anticipation.vocab as v

from tests.conftest import get_tokens_from_midi_file, TestConfigPatcher, patch_config_and_reload


def test_patch_config_at_test_time(patch_config_and_reload: TestConfigPatcher) -> None:
    # this is an edit that makes no sense, we wouldn't set this to 0
    patch_config_and_reload(COMPOUND_SIZE=0)
    assert config.COMPOUND_SIZE == 0

def test_config_patch_rolled_back() -> None:
    # test that changes from test patching is isolated
    assert config.COMPOUND_SIZE == 5

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
    print(compound)
    sequences = parse_info["midi_tokens"].strip().split("\n")

    assert len(compound) == 145
    assert parse_info["seqcount"] == 3
    assert len(sequences) == parse_info["seqcount"]

    # check each sequence
    seq_1 = [int(x) for x in sequences[0].split(" ")]
    seq_2 = [int(x) for x in sequences[1].split(" ")]
    seq_3 = [int(x) for x in sequences[2].split(" ")]
    assert len(seq_1) == (1 + (m * event_size))
    assert len(seq_2) == (1 + (m * event_size))
    assert len(seq_3) == (1 + (m * event_size))

    instr = 0
    # 50 is quarter note in 4/4 120 BPM
    assert seq_1 == [
        v.AUTOREGRESS,
        v.SEPARATOR, v.SEPARATOR, v.SEPARATOR,
        # (time, duration, instrument x note_value)
        v.TIME_OFFSET + 0, v.DUR_OFFSET + 50, v.NOTE_OFFSET + (config.MAX_PITCH * instr + 36), # C1
        v.TIME_OFFSET + 50, v.DUR_OFFSET + 50, v.NOTE_OFFSET + (config.MAX_PITCH * instr + 38), # D1
        v.TIME_OFFSET + 100, v.DUR_OFFSET + 50, v.NOTE_OFFSET + (config.MAX_PITCH * instr + 40), # E1
        v.TIME_OFFSET + 150, v.DUR_OFFSET + 50, v.NOTE_OFFSET + (config.MAX_PITCH * instr + 41), # F1
        v.TIME_OFFSET + 200, v.DUR_OFFSET + 50, v.NOTE_OFFSET + (config.MAX_PITCH * instr + 43), # G1
        v.TIME_OFFSET + 250, v.DUR_OFFSET + 50, v.NOTE_OFFSET + (config.MAX_PITCH * instr + 45), # A1
        v.TIME_OFFSET + 300, v.DUR_OFFSET + 50, v.NOTE_OFFSET + (config.MAX_PITCH * instr + 47), # B1
        v.TIME_OFFSET + 350, v.DUR_OFFSET + 50, v.NOTE_OFFSET + (config.MAX_PITCH * instr + 48), # C2
        v.TIME_OFFSET + 450, v.DUR_OFFSET + 50, v.NOTE_OFFSET + (config.MAX_PITCH * instr + 50), # D2
    ]
    assert seq_2 == [
        v.AUTOREGRESS,
        # (time, duration, instrument x note_value)
        0, 10050, 11052,
        50, 10050, 11053,
        100, 10050, 11055,
        150, 10050, 11057,
        200, 10050, 11059,
        250, 10050, 11060,
        350, 10050, 11062,
        400, 10050, 11064,
        450, 10050, 11065,
        500, 10050, 11067
    ]
    assert seq_3 == [
        v.AUTOREGRESS,
        # (time, duration, instrument x note_value)
        0, 10050, 11069,
        50, 10050, 11071,
        100, 10050, 11072,
        200, 10050, 11074,
        250, 10050, 11076,
        300, 10050, 11077,
        350, 10050, 11079,
        400, 10050, 11081,
        450, 10050, 11083,
        500, 10050, 11084
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
    assert len(midi_tokens.strip().split("\n")) == seqcount
    assert len(midi_tokens) == 494_036
