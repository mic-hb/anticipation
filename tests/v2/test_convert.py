from pathlib import Path

import pytest

from anticipation.v2.config import AnticipationV2Settings, Vocab
from anticipation.v2.types import MIDITick
import anticipation.v2.convert as v2_convert
import anticipation.convert as v1_convert
import anticipation.config as v1_config
from tests.conftest import TEST_DATA_PATH


@pytest.fixture()
def default_v2_settings() -> AnticipationV2Settings:
    return AnticipationV2Settings(
        vocab=Vocab(),
        debug=True,
    )


def _assert_compound_equality(
    v1_compound: list[int],
    v2_compound: list[int],
    time_tolerance_in_ticks: MIDITick = 1,
    duration_tolerance_in_ticks: MIDITick = 1,
) -> None:
    assert len(v2_compound) == len(v2_compound)
    for i in range(0, len(v2_compound), 5):
        a = v1_compound[i : i + 5]
        b = v2_compound[i : i + 5]

        # there is some floating point weirdness here due to seconds to
        # tick conversion and what not. Time and durations can be off
        # by 1 occasionally. The unit here is ticks, so it is only
        # minutely different from v1. Is literally a rounding error.
        if abs(a[1] - b[1]) > duration_tolerance_in_ticks:
            print(a, b)

        # time
        assert abs(a[0] - b[0]) <= time_tolerance_in_ticks
        # # duration
        assert abs(a[1] - b[1]) <= duration_tolerance_in_ticks

        # # note, instrument, velocity must be exactly the same
        if a[2:] != b[2:]:
            print(a, b)
        assert a[2:] == b[2:]


def test_v2_midi_to_compound_simple(
    c_major_midi_path: Path,
    default_v2_settings: AnticipationV2Settings,
) -> None:
    v2_compound = v2_convert.midi_to_compound(c_major_midi_path, default_v2_settings)
    v1_compound = v1_convert.midi_to_compound(
        str(c_major_midi_path.absolute()),
        debug=default_v2_settings.debug,
        time_resolution=v1_config.TIME_RESOLUTION,
    )
    _assert_compound_equality(v1_compound, v2_compound)


def test_v2_midi_to_compound_lakh_0(
    lmd_0_example_midi_path: Path,
    default_v2_settings: AnticipationV2Settings,
) -> None:
    v2_compound = v2_convert.midi_to_compound(
        lmd_0_example_midi_path, default_v2_settings
    )
    v1_compound = v1_convert.midi_to_compound(
        str(lmd_0_example_midi_path.absolute()),
        debug=default_v2_settings.debug,
        time_resolution=v1_config.TIME_RESOLUTION,
    )
    _assert_compound_equality(v1_compound, v2_compound)

def test_v2_midi_to_compound_lakh_1(
    default_v2_settings: AnticipationV2Settings,
) -> None:
    lmd_1_example_midi_path = TEST_DATA_PATH / "0c6b53ce52783ec7414b1fc7ce5c0286.mid"
    v2_compound = v2_convert.midi_to_compound_fast(
        lmd_1_example_midi_path, default_v2_settings
    )
    v1_compound = v1_convert.midi_to_compound(
        str(lmd_1_example_midi_path.absolute()),
        debug=default_v2_settings.debug,
        time_resolution=v1_config.TIME_RESOLUTION,
    )
    # v1_cleaned = []
    # for i in range(0, len(v1_compound), 5):
    #     e = v1_compound[i:i+5]
    #     if e[1] > 0:
    #         v1_cleaned.extend(e)
    #
    # v2_cleaned = []
    # for i in range(0, len(v2_compound), 5):
    #     e = v2_compound[i:i+5]
    #     if e[1] > 0:
    #         v2_cleaned.extend(e)
    v1_midi = v1_convert.compound_to_midi(v1_compound)
    v2_midi = v1_convert.compound_to_midi(v2_compound)

    for x, y in zip(v1_midi, v2_midi):
        assert x == y
    # assert v1_midi == v2_midi

    #_assert_compound_equality(v1_cleaned, v2_cleaned)


def test_v2_midi_to_compound_lakh_2(
    default_v2_settings: AnticipationV2Settings,
) -> None:
    lmd_1_example_midi_path = TEST_DATA_PATH / "08c8b965fd94c13611e26ba787e26d7f.mid"
    v2_compound = v2_convert.midi_to_compound(
        lmd_1_example_midi_path, default_v2_settings
    )
    v1_compound = v1_convert.midi_to_compound(
        str(lmd_1_example_midi_path.absolute()),
        debug=default_v2_settings.debug,
        time_resolution=v1_config.TIME_RESOLUTION,
    )
    _assert_compound_equality(v1_compound, v2_compound)