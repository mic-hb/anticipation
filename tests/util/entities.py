"""
These are utilities for testing / developing with the codebase.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Union
import re

import anticipation.vocab as v
import anticipation.ops as ops

from tests.util.constants import MIDI_PROGRAM_NAMES


class EventSpecialCode(Enum):
    """
    Used to handle things beyond the MIDI spec but required for our tokenized
    event sequences.
    """

    TYPICAL_EVENT = 0
    AUTOREGRESSIVE_TOKEN = 1
    ANTICIPATION_TOKEN = 2
    SEQ_SEPARATION_TOKENS = 3


class Note:
    """Utility to convert notes to tokens and human-readable names.

    Spans C-2 to G8.
    We consider C3 to be middle C, MIDI note 60.
    """

    # sharps only
    NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    NAME_PC_MAP = {name: i for i, name in enumerate(NAMES)}
    _NOTE_RE = re.compile(r"^([A-G]#?)(-?\d+)$")

    def __init__(self, midi_note_int: int) -> None:
        if not (0 <= midi_note_int < 128):
            raise ValueError(
                f"MIDI note integer must be in [0, 127], got {midi_note_int}"
            )
        self.midi_note_int = midi_note_int

    @classmethod
    def make(cls, midi_note_int_or_name: Union[str, int]) -> "Note":
        if isinstance(midi_note_int_or_name, int):
            return Note(midi_note_int_or_name)
        elif isinstance(midi_note_int_or_name, str):
            return Note.from_name(midi_note_int_or_name)
        else:
            raise TypeError("midi_note_int_or_name must be int or string.")

    @classmethod
    def from_name(cls, name: str) -> "Note":
        m = cls._NOTE_RE.match(name)
        assert m

        pitch, octave_str = m.groups()
        octave = int(octave_str) + 1

        assert pitch in cls.NAME_PC_MAP

        midi_note_int = (octave + 1) * 12 + cls.NAME_PC_MAP[pitch]
        assert 0 <= midi_note_int < 128
        return cls(midi_note_int)

    def to_name(self) -> str:
        pitch_class = self.midi_note_int % 12
        # lowest is C-2 (negative 2)
        octave = (self.midi_note_int // 12) - 2
        return f"{self.NAMES[pitch_class]}{octave}"

    def __int__(self) -> int:
        return self.midi_note_int

    def __str__(self) -> str:
        return self.to_name()

    def __eq__(self, other) -> bool:
        if not isinstance(other, Note):
            return False
        return self.midi_note_int == other.midi_note_int

    def __repr__(self) -> str:
        return f"Note(midi={self.midi_note_int}, name='{self.to_name()}')"


def get_note_instrument_token(instrument_midi_code: int, note_midi_code: int) -> int:
    return v.NOTE_OFFSET + (v.MAX_PITCH * instrument_midi_code + note_midi_code)


def get_midi_instrument_name_from_midi_instrument_code(
    instrument_midi_code: int,
) -> str:
    if 0 <= instrument_midi_code < len(MIDI_PROGRAM_NAMES):
        return MIDI_PROGRAM_NAMES[instrument_midi_code]
    elif instrument_midi_code == 128:
        # this happens a lot in Lakh MID, not sure if is a convention
        return "Drums (Channel 9)"
    elif instrument_midi_code == 129:
        return "REST"
    else:
        return "?"


@dataclass
class Event:
    time: int
    duration: int
    note_instr: int
    is_control: bool
    special_code: EventSpecialCode = EventSpecialCode.TYPICAL_EVENT

    @classmethod
    def from_midi_values(
        cls,
        midi_time: int,
        midi_duration: int,
        midi_instrument: int,
        midi_note: Union[int, str],
        is_control: bool = False,
        special_code: EventSpecialCode = EventSpecialCode.TYPICAL_EVENT,
    ) -> "Event":
        note_offset = v.CONTROL_OFFSET if is_control else 0
        time_offset = v.ATIME_OFFSET if is_control else v.TIME_OFFSET
        dur_offset = v.ADUR_OFFSET if is_control else v.DUR_OFFSET

        if midi_note == "REST" or midi_note == v.REST:
            # TODO: consider, should REST be special code?
            note_instr = v.REST
            note_offset = 0
            assert is_control is False
        else:
            parsed_note = Note.make(midi_note)
            note_instr = get_note_instrument_token(
                midi_instrument, parsed_note.midi_note_int
            )

        return Event(
            time=midi_time + time_offset,
            duration=midi_duration + dur_offset,
            note_instr=note_instr + note_offset,
            is_control=is_control,
            special_code=special_code,
        )

    @classmethod
    def from_token_seq(cls, raw_event_token_seq: list[int]) -> list["Event"]:
        if not raw_event_token_seq:
            return []

        events = []
        # the function ops.min_time expects sequence to not contain any flag tokens
        prev_t = ops.min_time(
            [x for x in raw_event_token_seq if x not in (v.ANTICIPATE, v.AUTOREGRESS)],
            seconds=False,
        )

        i = 0
        while i < len(raw_event_token_seq):
            if raw_event_token_seq[i] in (v.AUTOREGRESS, v.ANTICIPATE):
                ar = raw_event_token_seq[i] == v.AUTOREGRESS
                special_code = (
                    EventSpecialCode.AUTOREGRESSIVE_TOKEN
                    if ar
                    else EventSpecialCode.ANTICIPATION_TOKEN
                )
                events.append(
                    Event(
                        time=prev_t,
                        duration=v.DUR_OFFSET,
                        note_instr=get_note_instrument_token(0, 0),
                        is_control=False,
                        special_code=special_code,
                    )
                )
                i += 1

            if raw_event_token_seq[i] == v.SEPARATOR:
                events.append(
                    Event(
                        time=prev_t + 1,
                        duration=v.DUR_OFFSET,
                        note_instr=get_note_instrument_token(0, 0),
                        is_control=False,
                        special_code=EventSpecialCode.SEQ_SEPARATION_TOKENS,
                    )
                )
                i += 3

            e = raw_event_token_seq[i : i + 3]
            t, d, n = e
            events.append(
                Event(
                    time=t, duration=d, note_instr=n, is_control=(t >= v.ATIME_OFFSET)
                )
            )
            if events[-1].is_control:
                prev_t = t - v.ATIME_OFFSET
            else:
                prev_t = t
            i += 3

        return events

    def midi_time(self) -> int:
        if self.time < v.TIME_OFFSET:
            return self.time
        else:
            if self.is_control:
                return self.time - v.ATIME_OFFSET
            else:
                return self.time - v.TIME_OFFSET

    def midi_duration(self) -> int:
        if self.is_control:
            return self.duration - v.ADUR_OFFSET
        else:
            return self.duration - v.DUR_OFFSET

    def midi_note(self) -> int:
        note, _ = self._separate_note_instr()
        return note

    def midi_instrument(self) -> int:
        # midi instrument aka midi 'program code'
        _, instrument = self._separate_note_instr()
        return instrument

    def midi_instrument_name(self) -> str:
        return get_midi_instrument_name_from_midi_instrument_code(
            self.midi_instrument()
        )

    def _separate_note_instr(self) -> tuple[int, int]:
        t = v.CONTROL_OFFSET if self.is_control else 0
        b = self.note_instr - v.NOTE_OFFSET - t
        note = b - (2**7) * (b // 2**7)
        instr = b // 2**7
        # these are midi values, not tokens
        return note, instr

    def note(self) -> Note:
        return Note(self.midi_note())

    def as_tokens(self) -> tuple[int, ...]:
        if self.special_code == 1:
            return (v.AUTOREGRESS,)
        elif self.special_code == 2:
            return (v.ANTICIPATE,)
        elif self.special_code == 3:
            return v.SEPARATOR, v.SEPARATOR, v.SEPARATOR
        else:
            return self.time, self.duration, self.note_instr

    def __repr__(self) -> str:
        return "Event(midi_time={0}, midi_duration={1}, midi_note={2} ({6}), midi_instrument={3}, is_control={4}, special_code={5})".format(
            self.midi_time(),
            self.midi_duration(),
            self.midi_note(),
            self.midi_instrument(),
            self.is_control,
            self.special_code,
            self.note().to_name(),
        )
