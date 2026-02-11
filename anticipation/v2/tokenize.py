from collections import defaultdict

from pathlib import Path
from typing import Optional, Iterable, Union, Iterator
import warnings

import numpy as np

from anticipation.v2 import ops as v2_ops
from anticipation.tokenize import (
    extract_instruments as v1_extract_instruments,
    extract_spans as v1_extract_spans,
)

from anticipation.v2.types import MIDIFileIgnoredReason, Token, MIDIProgramCode
from anticipation.v2.config import AnticipationV2Settings
from anticipation.v2.convert import midi_to_compound, compound_to_events


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
        my_midi_file,
        settings=settings,
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


class SequencePacker:
    """Buffer-like coordinator that flushes tokens to a file or just accumulates them to a list.

    This adds punctuation-like semantic tokens to its outputs as they are streamed out.

    NB: list given in ctor is pass by reference, will mutate in place.
    """

    def __init__(
        self, target: Union[list, Path], settings: AnticipationV2Settings
    ) -> None:
        self._iterator_queue = []
        self._buf = []
        if isinstance(target, list):
            # self._buf = target
            self._target = target
        elif isinstance(target, Path):
            # self._buf = []
            # target may not exist yet, but the folder that
            # contains it should
            assert target.parent.exists()
            assert target.parent.is_dir()
            self._target = open(target, "w")
        else:
            raise TypeError("Must have path or list as target.")

        self._settings = settings

    def add_tokenized_file(
        self,
        control_prefix: tuple[Token, ...],
        tokenized_file: Iterator[tuple[Token, ...]],
    ) -> None:
        # assumption: this is called once per file augmentation
        # tokenized_file is not split in the middle and does not continue some previous
        # sequence
        self._iterator_queue.append((control_prefix, tokenized_file))

    def flush(self) -> None:
        for control_prefix, iter_curr_tokens in self._iterator_queue:
            to_add = [self._settings.vocab.SEPARATOR, *control_prefix]

            if len(self._buf) + len(to_add) == self._settings.context_size:
                # prevent the scenario where the control prefix is written in
                # a different sequence as the tokens it describes
                self._buf += [self._settings.vocab.PAD] * (
                    self._settings.context_size - len(self._buf)
                )
                self._write_seq(self._buf)
                self._buf = []

            self._buf += to_add
            for next_elem in iter_curr_tokens:
                if len(self._buf) + len(next_elem) > self._settings.context_size:
                    # prevent splitting a triple between sequences
                    self._buf += [self._settings.vocab.PAD] * (
                        self._settings.context_size - len(self._buf)
                    )

                if len(self._buf) == self._settings.context_size:
                    # write sequence
                    self._write_seq(self._buf)

                    # reset the buffer to be just the prefix
                    self._buf = [*control_prefix]

                self._buf += list(next_elem)

        # clear everything out
        self._iterator_queue = []

    def _write_seq(self, buf: list[Token]):
        if isinstance(self._target, list):
            # copy and move
            self._target.append(list(buf))
        else:
            # file
            to_write = " ".join([str(tok) for tok in self._buf]) + "\n"
            self._target.write(to_write)

    def close(self) -> None:
        if isinstance(self._target, list):
            return

        print(f"Token buffer had remaining: {len(self._buf)}")
        self._target.close()

    def __len__(self) -> int:
        return len(self._buf)


def tokenize(
    midi_files: Iterable[Path],
    output: Union[list, Path],
    settings: AnticipationV2Settings,
) -> dict[MIDIFileIgnoredReason, int]:
    """Tokenizes MIDI for v2 Anticipatory Training

    Args:
      midi_files: An iterable collection of MIDI files. These are the
        source data that we will turn into tokens.
      output: This can be either a list or a Path object to a file. In
        the v1 tokenize function, this argument would be a string to
        a file location on disk. To keep parity with that pattern, we
        can pass a file here or a list. If given a file, the v1 behavior
        is performed. If given a list, the tokens are just appended to
        the list and nothing is written to disk. The list is mutated in
        place - so keep a reference to it. In python all lists are pass
        by reference.
      settings: The Anticipation v2 settings object. Settings in here
        will affect how MIDI events are tokenized.

    Returns:
      A dictionary of (reason ignored, number of times it happened) for
      the given files. This will be empty if no files were ignored.
    """

    stats = defaultdict(int)
    buf = SequencePacker(target=output, settings=settings)
    for file_idx, midi_file in enumerate(midi_files):
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
            control_prefix, token_iterator = _get_augmentation_autoregressive(
                tokenized_midi, settings
            )
            buf.add_tokenized_file(control_prefix, token_iterator)

        # 2. instrument anticipation
        for _ in range(
            settings.num_instrument_anticipation_augmentations_per_midi_file
        ):
            control_prefix, token_iterator = _get_augmentation_instrument(
                tokenized_midi,
                all_midi_program_codes,
                settings,
            )
            buf.add_tokenized_file(control_prefix, token_iterator)

        # 3. span anticipation
        for _ in range(settings.num_span_anticipation_augmentations_per_midi_file):
            # TODO: not done w this yet
            buf.add_tokenized_file(
                _get_span_augmentation(tokenized_midi, end_time_in_ticks, settings)
            )

        # write all the sequences to a target
        buf.flush()

    # close files if necessary
    buf.close()

    # return any stats, cast to regular dictionary for safety
    return dict(stats)


def _get_augmentation_autoregressive(
    tokens: list[Token], settings: AnticipationV2Settings
) -> tuple[tuple[Token, ...], Iterator[tuple[Token, ...]]]:
    assert len(tokens) % 3 == 0, "bad length"

    if settings.tick_token_frequency_in_midi_ticks > 0:
        # step through events and ticks
        stream = v2_ops.streaming_relativize_to_tick(
            v2_ops.streaming_add_ticks(tokens, settings), settings
        )
    else:
        # when events drop below a certain density, pad them with rests
        # in the style that uses tick tokens, we do not need these
        tokens_with_rests = v2_ops.add_rests(tokens, settings)
        stream = (tokens_with_rests[i : i + 3] for i in range(0, len(tokens), 3))

    return (settings.vocab.AUTOREGRESS,), stream


def _sample_instrument_subset(all_midi_program_codes: list[int]) -> list[int]:
    if len(all_midi_program_codes) <= 1:
        # not really well-defined for this case...
        return []

    # instrument augmentation: at least one, but not all instruments
    # each time this is called it is like a random subset draw!
    u = 1 + np.random.randint(len(all_midi_program_codes) - 1)
    instrument_subset = np.random.choice(all_midi_program_codes, u, replace=False)
    return list(instrument_subset)


def _get_augmentation_instrument(
    tokens: list[Token],
    all_midi_program_codes: list[int],
    settings: AnticipationV2Settings,
) -> tuple[tuple[Token, ...], Iterator[tuple[Token, ...]]]:
    assert len(tokens) % 3 == 0, "bad length"
    # if settings.debug:
    #     _check_no_punctuation_tokens(tokens, settings)

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
    return (settings.vocab.ANTICIPATE,), stream


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
    # TODO: wait you can't reliably determine if there's punctuation this way
    # TODO: because the time (for long songs) could collide...
    # TODO: hmmmm....
    # this is kind of slow, so we only call it if in a debug context
    punctuations = v2_ops.get_punctuation_tokens_idx(tokens, settings)
    assert not punctuations, (
        f"token sequence should not have separator, rest, anticipate, etc. tokens at "
        f"this point in data processing. Punctuations: {punctuations}"
    )
