"""
These are operations that use the settings object instead of using global config.
Their functionality is the same as v1 ops unless stated otherwise.
"""

from collections import defaultdict
from typing import Optional, Union, Iterator

from anticipation.v2.config import AnticipationV2Settings
from anticipation.v2.types import Token


def get_punctuation_tokens_idx(
    tokens: list[Token], settings: AnticipationV2Settings
) -> dict[Token, int]:
    # new to v2
    v = settings.vocab
    investigate = {
        v.SEPARATOR: 0,
        v.REST: 0,
        v.ANTICIPATE: 0,
        v.AUTOREGRESS: 0,
        v.METRONOME: 0,
    }
    for i, e in enumerate(tokens):
        if e in investigate:
            investigate[e] += 1

    return {k: v for k, v in investigate.items() if v > 0}


def min_time(
    tokens: list[Token],
    settings: AnticipationV2Settings,
    seconds: bool = True,
    instr: Optional[int] = None,
) -> Union[int, float]:
    mt = None
    for time, dur, note in zip(tokens[0::3], tokens[1::3], tokens[2::3]):
        # stop calculating at sequence separator
        if note == settings.vocab.SEPARATOR:
            break

        if note < settings.vocab.CONTROL_OFFSET:
            time -= settings.vocab.TIME_OFFSET
            note -= settings.vocab.NOTE_OFFSET
        else:
            time -= settings.vocab.ATIME_OFFSET
            note -= settings.vocab.ANOTE_OFFSET

        # min time of a particular instrument
        if instr is not None and instr != note // 2**7:
            continue

        mt = time if mt is None else min(mt, time)

    if mt is None:
        mt = 0
    return mt / float(settings.time_resolution) if seconds else mt


def max_time(
    tokens: list[Token],
    settings: AnticipationV2Settings,
    seconds: bool = True,
    instr: Optional[int] = None,
) -> Union[int, float]:
    mt = 0
    for time, dur, note in zip(tokens[0::3], tokens[1::3], tokens[2::3]):
        # keep checking for max_time, even if it appears after a separator
        # (this is important because we use this check for vocab overflow in tokenization)
        if note == settings.vocab.SEPARATOR:
            continue

        if note < settings.vocab.CONTROL_OFFSET:
            time -= settings.vocab.TIME_OFFSET
            note -= settings.vocab.NOTE_OFFSET
        else:
            time -= settings.vocab.ATIME_OFFSET
            note -= settings.vocab.ANOTE_OFFSET

        # max time of a particular instrument
        if instr is not None and instr != note // 2**7:
            continue

        mt = max(mt, time)

    return mt / float(settings.time_resolution) if seconds else mt


def get_instruments(
    tokens: list[Token], settings: AnticipationV2Settings
) -> dict[int, int]:
    instruments = defaultdict(int)
    for time, dur, note in zip(tokens[0::3], tokens[1::3], tokens[2::3]):
        if note >= settings.vocab.SPECIAL_OFFSET:
            continue

        if note < settings.vocab.CONTROL_OFFSET:
            note -= settings.vocab.NOTE_OFFSET
        else:
            note -= settings.vocab.ANOTE_OFFSET

        instr = note // 2**7
        instruments[instr] += 1

    # (instrument type, num instances)
    # new: cast to regular dictionary just to be safe
    return dict(instruments)


def translate(
    tokens: list[Token],
    dt: Union[int, float],
    settings: AnticipationV2Settings,
    seconds: bool = False,
) -> list[Token]:
    if seconds:
        dt = int(settings.time_resolution * dt)

    new_tokens = []
    for time, dur, note in zip(tokens[0::3], tokens[1::3], tokens[2::3]):
        # stop translating after EOT
        if note == settings.vocab.SEPARATOR:
            new_tokens.extend([time, dur, note])
            dt = 0
            continue

        if note < settings.vocab.CONTROL_OFFSET:
            this_time = time - settings.vocab.TIME_OFFSET
        else:
            this_time = time - settings.vocab.ATIME_OFFSET

        assert 0 <= this_time + dt
        new_tokens.extend([time + dt, dur, note])

    return new_tokens


def anticipate(
    events: list[Token], controls: list[Token], settings: AnticipationV2Settings
) -> tuple[list[Token], list[Token]]:
    """
    Interleave a sequence of events with anticipated controls.

    Inputs:
      events   : a sequence of events
      controls : a sequence of time-localized controls
      settings    : anticipation v2 global settings object

    Returns:
      tokens   : interleaved events and anticipated controls
      controls : unconsumed controls (control time > max_time(events) + delta)
    """

    if len(controls) == 0:
        return events, controls

    delta_ticks = settings.delta * settings.time_resolution
    tokens = []
    event_time = 0
    control_time = controls[0] - settings.vocab.ATIME_OFFSET
    for time, dur, note in zip(events[0::3], events[1::3], events[2::3]):
        while event_time >= control_time - delta_ticks:
            tokens.extend(controls[0:3])
            controls = controls[3:]  # consume this control
            control_time = (
                controls[0] - settings.vocab.ATIME_OFFSET
                if len(controls) > 0
                else float("inf")
            )

        assert note < settings.vocab.CONTROL_OFFSET
        event_time = time - settings.vocab.TIME_OFFSET
        tokens.extend([time, dur, note])

    return tokens, controls


def add_rests(
    tokens: list[Token],
    settings: AnticipationV2Settings,
    start_time_in_ticks: Optional[int] = None,
    end_time_in_ticks: Optional[int] = None,
) -> list[Token]:
    """
    Equivalent to `ops.pad` in v1. We rename this 'add rests' because there is now a
    padding token, which serves a different purpose.
    """
    density = settings.time_resolution
    end_time = settings.vocab.TIME_OFFSET + (
        end_time_in_ticks
        if end_time_in_ticks is not None
        else max_time(tokens, settings, seconds=False)
    )
    previous_time = settings.vocab.TIME_OFFSET + (
        start_time_in_ticks if start_time_in_ticks is not None else 0
    )

    new_tokens = []
    for time, dur, note in zip(tokens[0::3], tokens[1::3], tokens[2::3]):
        # must pad before separation, anticipation
        assert note < settings.vocab.CONTROL_OFFSET

        # insert pad tokens to ensure the desired density
        while time > previous_time + density:
            new_tokens.extend(
                [
                    previous_time + density,
                    settings.vocab.DUR_OFFSET,
                    settings.vocab.REST,
                ]
            )
            previous_time += density

        new_tokens.extend([time, dur, note])
        previous_time = time

    while end_time > previous_time + density:
        new_tokens.extend(
            [
                previous_time + density,
                settings.vocab.DUR_OFFSET,
                settings.vocab.REST,
            ]
        )
        previous_time += density

    return new_tokens


# --- NEW TO V2 BELOW THIS LINE ---


def relativize_token_seq_time(
    seq: list[Token], settings: AnticipationV2Settings
) -> list[Token]:
    # shift all the time tokens in the sequence to 0
    # relative to the first time in the sequence
    return translate(
        seq, -min_time(seq, settings, seconds=False), settings, seconds=False
    )


def is_flag_token(token: Token, settings: AnticipationV2Settings) -> bool:
    return token == settings.vocab.ANTICIPATE or token == settings.vocab.AUTOREGRESS


def streaming_anticipate(
    events: list[Token], controls: list[Token], settings: AnticipationV2Settings
) -> Iterator[tuple[int, int, int]]:
    """
    Interleave a sequence of events with anticipated controls.

    Inputs:
      events   : a sequence of events
      controls : a sequence of time-localized controls
      settings    : anticipation v2 global settings object

    Returns:
        a generator that returns interleaved events and controls by anticipatory
        ordering that may be consumed as a stream
    """
    delta_ticks = settings.delta * settings.time_resolution
    event_time = 0
    control_time = controls[0] - settings.vocab.ATIME_OFFSET
    for i in range(0, len(events), 3):
        time, dur, note = events[i : i + 3]
        while event_time >= control_time - delta_ticks:
            # returning this as a tuple as protection against subsequent
            # mutations
            yield tuple(controls[0:3])

            controls = controls[3:]  # consume this control
            control_time = (
                controls[0] - settings.vocab.ATIME_OFFSET
                if len(controls) > 0
                else float("inf")
            )

        assert note < settings.vocab.CONTROL_OFFSET
        event_time = time - settings.vocab.TIME_OFFSET

        # returning this as a tuple as protection against subsequent
        # mutations
        yield time, dur, note
