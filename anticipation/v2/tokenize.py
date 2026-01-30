from collections import defaultdict

from pathlib import Path
from typing import Optional, Iterable
import warnings

import numpy as np

from anticipation.v2 import ops as v2_ops
from anticipation.convert import midi_to_compound
from anticipation.tokenize import (
    extract_instruments as v1_extract_instruments,
    extract_spans as v1_extract_spans,
)

from anticipation.v2.config import AnticipationV2Settings
from anticipation.v2.types import MIDIFileIgnoredReason, Token, MIDIProgramCode


def compound_to_events(
    compound: list[int], settings: AnticipationV2Settings
) -> tuple[list[Token], int]:
    assert len(compound) % 5 == 0
    tokens = compound.copy()

    # remove velocities
    del tokens[4::5]

    # combine (note, instrument)
    # in v1, a note value of -1 was used as a sentinel
    # and in this `compound_to_events` function, if -1 appeared, then it
    # was set to SEPARATOR.
    # in v2, we now enforce that this -1 never appear in the note or instrument.
    # ideally this function does not add any punctuation-like tokens (PAD, SEP,
    # AUTOREGRESS, etc.)
    assert all(0 <= tok < 2**7 for tok in tokens[2::4])
    assert all(0 <= tok < 129 for tok in tokens[3::4])
    tokens[2::4] = [
        settings.max_midi_pitch * instr + note
        for note, instr in zip(tokens[2::4], tokens[3::4])
    ]
    tokens[2::4] = [settings.vocab.NOTE_OFFSET + tok for tok in tokens[2::4]]
    del tokens[3::4]

    # max duration cutoff
    max_duration = settings.time_resolution * settings.max_note_duration_in_seconds
    num_note_truncations = sum([1 for tok in tokens[1::3] if tok >= max_duration])

    # set unknown durations to 250ms
    tokens[1::3] = [
        settings.time_resolution // 4 if tok == -1 else min(tok, max_duration - 1)
        for tok in tokens[1::3]
    ]

    tokens[1::3] = [settings.vocab.DUR_OFFSET + tok for tok in tokens[1::3]]

    assert min(tokens[0::3]) >= 0
    tokens[0::3] = [settings.vocab.TIME_OFFSET + tok for tok in tokens[0::3]]

    assert len(tokens) % 3 == 0

    # tokens are: (time, duration, note x instrument)
    return tokens, num_note_truncations


def maybe_tokenize(
    compound_tokens: list[int], settings: AnticipationV2Settings
) -> tuple[list[Token], int, Optional[MIDIFileIgnoredReason]]:
    if len(compound_tokens) < settings.compound_size * settings.min_track_events:
        # skip sequences with very few events
        return [], 0, MIDIFileIgnoredReason.TOO_FEW_EVENTS

    events, num_note_truncations = compound_to_events(compound_tokens, settings)
    end_time = v2_ops.max_time(events, settings, seconds=False)

    # don't want to deal with extremely short tracks
    if end_time < settings.time_resolution * settings.min_track_time_in_seconds:
        return [], 0, MIDIFileIgnoredReason.TOO_FEW_SECONDS

    # don't want to deal with extremely long tracks
    if end_time > settings.time_resolution * settings.max_track_time_in_seconds:
        return [], 0, MIDIFileIgnoredReason.TOO_MANY_SECONDS

    # skip sequences more instruments than MIDI channels (16)
    if len(v2_ops.get_instruments(events, settings)) > settings.max_track_instruments:
        return (
            [],
            0,
            MIDIFileIgnoredReason.TOO_MANY_INSTRUMENTS,
        )

    return events, num_note_truncations, None


def tokenize_midi_file(
    my_midi_file: Path, settings: AnticipationV2Settings
) -> tuple[
    list[Token], int, Optional[MIDIFileIgnoredReason], list[MIDIProgramCode], int
]:
    assert isinstance(my_midi_file, Path)

    # midi_to_compound does not depend on vocab choices
    midi_compound: list[int] = midi_to_compound(
        str(my_midi_file.absolute()),
        debug=settings.debug,
        time_resolution=settings.time_resolution,
    )

    # now we are tokenizing the MIDI, using several tokens from our settings
    all_events, num_note_truncations, reason_ignored = maybe_tokenize(
        midi_compound, settings
    )

    all_midi_program_codes = []
    end_time_in_ticks = 0

    if reason_ignored is None:
        # we are actually going to use this, get some information we
        # will need later
        all_midi_program_codes = list(v2_ops.get_instruments(all_events, settings))
        end_time_in_ticks: int = v2_ops.max_time(all_events, settings, seconds=False)

    return (
        all_events,
        num_note_truncations,
        reason_ignored,
        all_midi_program_codes,
        end_time_in_ticks,
    )


def tokenize(
    midi_files: Iterable[Path],
    settings: AnticipationV2Settings,
) -> list[Token]:
    stats = defaultdict(int)
    buf = []
    for midi_file in midi_files:
        (
            tokenized_midi,
            num_note_truncations,
            reason_ignored,
            all_midi_program_codes,
            end_time_in_ticks,
        ) = tokenize_midi_file(midi_file, settings)
        if reason_ignored is not None:
            warnings.warn(
                f"Sequence in file {midi_file} was ignored for reason: {reason_ignored}",
                UserWarning,
            )
            stats[reason_ignored] += 1
            continue

        # 1. pure autoregressive sequence
        for _ in range(settings.num_autoregressive_seq_per_midi_file):
            buf += _get_augmentation_autoregressive(tokenized_midi, settings)

        # 2. instrument anticipation
        for _ in range(
            settings.num_instrument_anticipation_augmentations_per_midi_file
        ):
            buf += _get_augmentation_instrument(
                tokenized_midi,
                all_midi_program_codes,
                settings,
            )

        # 3. span anticipation
        for _ in range(settings.num_span_anticipation_augmentations_per_midi_file):
            span_anticipation_seq = _get_span_augmentation(
                tokenized_midi, end_time_in_ticks, settings
            )
            buf.extend(span_anticipation_seq)

        # 4. random anticipation augmentations (?)
    return buf


def _get_augmentation_autoregressive(
    tokens: list[Token], settings: AnticipationV2Settings
) -> list[Token]:
    assert len(tokens) % 3 == 0, "bad length"
    if settings.debug:
        _check_no_punctuation_tokens(tokens, settings)

    # step through events and ticks
    stream = v2_ops.streaming_relativize_to_tick(
        v2_ops.streaming_add_ticks(tokens, settings), settings
    )

    return v2_ops.pack(
        stream, seq_header=(settings.vocab.AUTOREGRESS,), settings=settings
    )


def _sample_instrument_subset(all_midi_program_codes: list[int]) -> list[int]:
    # instrument augmentation: at least one, but not all instruments
    # each time this is called it is like a random subset draw!
    u = 1 + np.random.randint(len(all_midi_program_codes) - 1)
    instrument_subset = np.random.choice(all_midi_program_codes, u, replace=False)
    return list(instrument_subset)


def _get_augmentation_instrument(
    tokens: list[Token],
    all_midi_program_codes: list[int],
    settings: AnticipationV2Settings,
) -> list[Token]:
    assert len(tokens) % 3 == 0, "bad length"
    if settings.debug:
        _check_no_punctuation_tokens(tokens, settings)

    # in v1, REST tokens are not present in token sequence before calling
    # instrument extraction...
    events, controls = v1_extract_instruments(
        tokens, _sample_instrument_subset(all_midi_program_codes)
    )
    assert len(controls) % 3 == 0
    assert len(events) % 3 == 0

    # add ticks to the events only
    event_stream = v2_ops.streaming_add_ticks(events, settings)
    control_stream = (controls[i : i + 3] for i in range(0, len(controls), 3))
    stream = v2_ops.streaming_relativize_to_tick(
        v2_ops.streaming_anticipate(event_stream, control_stream, settings), settings
    )
    return v2_ops.pack(
        stream, seq_header=(settings.vocab.ANTICIPATE,), settings=settings
    )


def _get_span_augmentation(
    tokens: list[Token],
    end_time_in_ticks: int,
    settings: AnticipationV2Settings,
):
    assert len(tokens) % 3 == 0, "bad length"
    if settings.debug:
        _check_no_punctuation_tokens(tokens, settings)

    # for now, I am leaving the implementation exactly as it was in v1
    # EXCEPT for how the SEPARATE token is handled
    events, controls = v1_extract_spans(tokens, rate=settings.span_anticipation_lambda)
    events = v2_ops.add_rests(events, settings, end_time_in_ticks)
    interleaved, controls = v2_ops.anticipate(events, controls, settings)
    assert len(controls) == 0

    chunks = []
    ctx_length = settings.event_size * settings.m
    for i in range(0, len(interleaved), ctx_length):
        subsequence = interleaved[i : i + ctx_length]
        assert subsequence

        # TODO:...
        # Q: what if we get a subsequence that has no span?

        # important... flag is just AR here
        subsequence.insert(0, settings.vocab.ANTICIPATE)
        chunks.extend(subsequence)

    assert chunks[0] == settings.vocab.ANTICIPATE
    return chunks


def _check_no_punctuation_tokens(
    tokens: list[Token], settings: AnticipationV2Settings
) -> None:
    punctuations = v2_ops.get_punctuation_tokens_idx(tokens, settings)
    assert not punctuations, (
        f"token sequence should not have separator, rest, anticipate, etc. tokens at "
        f"this point in data processing. Punctuations: {punctuations}"
    )
