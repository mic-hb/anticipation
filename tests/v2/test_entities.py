import pytest

from anticipation.v2.config import AnticipationV2Settings, Vocab
from tests.util.entities import get_note_instrument_token


def test_get_note_instrument_token(local_midi_settings: AnticipationV2Settings) -> None:
    # 1st program code
    assert 1100 == get_note_instrument_token(0, 0, config=local_midi_settings)
    assert 1227 == get_note_instrument_token(0, 127, config=local_midi_settings)

    # 2nd program code
    assert 1228 == get_note_instrument_token(1, 0, config=local_midi_settings)
    assert 1355 == get_note_instrument_token(1, 127, config=local_midi_settings)

    # 129th program code, drums
    assert 17484 == get_note_instrument_token(128, 0, config=local_midi_settings)
    assert 17611 == get_note_instrument_token(128, 127, config=local_midi_settings)

    with pytest.raises(ValueError):
        get_note_instrument_token(-1, -1)

    # check for some off-by-1-errors
    with pytest.raises(ValueError):
        get_note_instrument_token(129, 0)

    with pytest.raises(ValueError):
        get_note_instrument_token(0, 128)


def test_get_tick_offset() -> None:
    default_v2 = AnticipationV2Settings(vocab=Vocab())
    assert 27511 == get_note_instrument_token(128, 127, config=default_v2)
