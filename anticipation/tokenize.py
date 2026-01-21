"""
Top-level functions for preprocessing data to be used for training.
"""
from enum import Enum
from turtledemo.penrose import start

from tqdm import tqdm

import numpy as np

from anticipation import ops
from anticipation.config import *
from anticipation.vocab import *
from anticipation.convert import compound_to_events, midi_to_interarrival

ANTICIPATION_RATES = 10


class MIDIFileDiscardedReason(Enum):
    DURATION_TOO_SHORT = 1
    DURATION_TOO_LONG = 2
    TOO_MANY_INSTRUMENTS = 3
    CHUNK_TOO_LONG = 4


def extract_spans(all_events, rate):
    events = []
    controls = []
    spans = []
    span = True
    next_span = end_span = TIME_OFFSET+0
    for time, dur, note in zip(all_events[0::3],all_events[1::3],all_events[2::3]):
        assert(note not in [SEPARATOR, REST]) # shouldn't be in the sequence yet

        # end of an anticipated span; decide when to do it again (next_span)
        if span and time >= end_span:
            span = False
            d = np.random.exponential(1./rate)
            next_span = time+int(TIME_RESOLUTION*d)
            # tuples of [(start_time, end_time),...]
            spans.append((next_span, ))

        # anticipate a 3-second span
        if (not span) and time >= next_span:
            span = True
            end_span = time + DELTA*TIME_RESOLUTION
            spans.append((*spans.pop(-1), end_span))

        if span:
            # mark this event as a control
            controls.extend([CONTROL_OFFSET+time, CONTROL_OFFSET+dur, CONTROL_OFFSET+note])
        else:
            events.extend([time, dur, note])

    return events, controls, spans


def extract_random(all_events, rate):
    events = []
    controls = []
    for time, dur, note in zip(all_events[0::3],all_events[1::3],all_events[2::3]):
        assert(note not in [SEPARATOR, REST]) # shouldn't be in the sequence yet

        if np.random.random() < rate/float(ANTICIPATION_RATES):
            # mark this event as a control
            controls.extend([CONTROL_OFFSET+time, CONTROL_OFFSET+dur, CONTROL_OFFSET+note])
        else:
            events.extend([time, dur, note])

    return events, controls


def extract_instruments(all_events: list[int], instruments: list[int]) -> tuple[list[int], list[int]]:
    # turn every event emitted by an instrument in given list
    # of instruments into a control, all others are events
    events = []
    controls = []
    for time, dur, note in zip(all_events[0::3],all_events[1::3],all_events[2::3]):
        assert note < CONTROL_OFFSET         # shouldn't be in the sequence yet
        assert note not in [SEPARATOR, REST] # these shouldn't either

        instr = (note-NOTE_OFFSET)//2**7
        if instr in instruments:
            # mark this event as a control
            controls.extend([CONTROL_OFFSET+time, CONTROL_OFFSET+dur, CONTROL_OFFSET+note])
        else:
            events.extend([time, dur, note])

    return events, controls


def maybe_tokenize(compound_tokens: list[int]):
    # skip sequences with very few events
    if len(compound_tokens) < COMPOUND_SIZE*MIN_TRACK_EVENTS:
        return None, None, MIDIFileDiscardedReason.DURATION_TOO_SHORT # short track

    events, truncations = compound_to_events(compound_tokens, stats=True)
    end_time = ops.max_time(events, seconds=False)

    # don't want to deal with extremely short tracks
    if end_time < TIME_RESOLUTION*MIN_TRACK_TIME_IN_SECONDS:
        return None, None, MIDIFileDiscardedReason.DURATION_TOO_SHORT # short track

    # don't want to deal with extremely long tracks
    if end_time > TIME_RESOLUTION*MAX_TRACK_TIME_IN_SECONDS:
        return None, None, MIDIFileDiscardedReason.DURATION_TOO_LONG # long track

    # skip sequences more instruments than MIDI channels (16)
    if len(ops.get_instruments(events)) > MAX_TRACK_INSTR:
        return None, None, MIDIFileDiscardedReason.TOO_MANY_INSTRUMENTS # too many instruments

    return events, truncations, 0


def tokenize_ia(datafiles, output, augment_factor, idx=0, debug=False):
    assert augment_factor == 1 # can't augment interarrival-tokenized data

    all_truncations = 0
    seqcount = rest_count = 0
    stats = 4*[0] # (short, long, too many instruments, inexpressible)
    np.random.seed(0)

    with open(output, 'w') as outfile:
        concatenated_tokens = []
        for j, filename in tqdm(list(enumerate(datafiles)), desc=f'#{idx}', position=idx+1, leave=True):
            with open(filename, 'r') as f:
                _, _, status = maybe_tokenize([int(token) for token in f.read().split()])

            if status > 0:
                stats[status-1] += 1
                continue

            filename = filename[:-len('.compound.txt')] # get the original MIDI

            # already parsed; shouldn't raise an exception
            tokens, truncations = midi_to_interarrival(filename, stats=True)
            tokens[0:0] = [MIDI_SEPARATOR]
            concatenated_tokens.extend(tokens)
            all_truncations += truncations

            # write out full sequences to file
            while len(concatenated_tokens) >= CONTEXT_SIZE:
                seq = concatenated_tokens[0:CONTEXT_SIZE]
                concatenated_tokens = concatenated_tokens[CONTEXT_SIZE:]
                outfile.write(' '.join([str(tok) for tok in seq]) + '\n')
                seqcount += 1

    if debug:
        fmt = 'Processed {} sequences (discarded {} tracks, discarded {} seqs, added {} rest tokens)'
        print(fmt.format(seqcount, stats[0]+stats[1]+stats[2], stats[3], rest_count))

    return (seqcount, rest_count, stats[0], stats[1], stats[2], stats[3], all_truncations)

def _add_anticipation_sep_prefix(tokens: list[int]) -> list[int]:
    # technically mutates in place, but for clarity return same ptr
    tokens[0:0] = [SEPARATOR, SEPARATOR, SEPARATOR]
    return tokens

def _apply_augmentation(
    all_events: list[int],
    instruments: list[int],
    curr_augment_factor_k: int,
    do_random_augmentation: bool,
    lmbda: float = 0.05
) -> tuple[list[int], list[int]]:
    if curr_augment_factor_k % 10 == 0:
        # no augmentation
        events = all_events.copy()
        controls = []
    elif curr_augment_factor_k % 10 == 1:
        # span augmentation
        events, controls, _ = extract_spans(all_events, lmbda)
    elif curr_augment_factor_k % 10 < 6:
        # random augmentation
        if do_random_augmentation:
            r = np.random.randint(1, ANTICIPATION_RATES)
            events, controls = extract_random(all_events, r)
        else:
            return [], []
    else:
        if len(instruments) > 1:
            # instrument augmentation: at least one, but not all instruments
            u = 1 + np.random.randint(len(instruments) - 1)
            subset = np.random.choice(instruments, u, replace=False)
            events, controls = extract_instruments(all_events, subset)
        else:
            # no augmentation
            # Q: would this not just create copies?
            events = all_events.copy()
            controls = []

    return events, controls

def _maybe_tokenize_compound_from_file(filename: str):
    instruments = []
    end_time = -1
    with open(filename, 'r') as f:
        all_events, truncations, status = maybe_tokenize([int(token) for token in f.read().split()])
        if status == 0:
            instruments = list(ops.get_instruments(all_events).keys())
            end_time = ops.max_time(all_events, seconds=False)

    return all_events, truncations, status, instruments, end_time


def tokenize(datafiles, output, augment_factor, idx=0, do_random_augmentation: bool = True, debug: bool = False):
    tokens = []
    all_truncations = 0
    seqcount = rest_count = 0
    stats = 4*[0] # (short, long, too many instruments, inexpressible)
    np.random.seed(0)

    with open(output, 'w') as outfile:
        # this is like a buffer that accumulates sequences for each file
        # it is flushed to disk in contiguous subsequences of specific
        # context length
        concatenated_tokens = []

        for j, filename in tqdm(list(enumerate(datafiles)), desc=f'#{idx}', position=idx+1, leave=True):
            # load compound and tokenize it, along with top level info like ending time and instruments
            # or report why we should skip it
            all_events, truncations, status, instruments, end_time = _maybe_tokenize_compound_from_file(
                filename
            )
            if status > 0:
                stats[status.value-1] += 1
                continue

            for k in range(augment_factor):
                # by this point, events and controls are in different
                # vocabulary spaces
                events, controls = _apply_augmentation(
                    all_events,
                    instruments,
                    k,
                    do_random_augmentation=do_random_augmentation
                )
                if not events:
                    continue

                if len(concatenated_tokens) == 0:
                    z = ANTICIPATE if k % 10 != 0 else AUTOREGRESS

                all_truncations += truncations

                # pad the tokens with REST, keep track of how many we add
                events = ops.pad(events, end_time)
                rest_count += sum(1 if tok == REST else 0 for tok in events[2::3])

                packed_sequences, concatenated_tokens, num_seq, num_discarded_seq = pack_sequences(
                    events,
                    controls,
                    concatenated_tokens,
                    z,
                    start_seq_ordering=True
                )

                # grab the current augmentation controls if we didn't already
                z = ANTICIPATE if k % 10 != 0 else AUTOREGRESS
                seqcount += num_seq
                stats[MIDIFileDiscardedReason.CHUNK_TOO_LONG.value - 1] += num_discarded_seq

                # write chunks to file
                for seq in packed_sequences:
                    outfile.write(' '.join([str(tok) for tok in seq]) + '\n')

    if debug:
        fmt = 'Processed {} sequences (discarded {} tracks, discarded {} seqs, added {} rest tokens)'
        print(fmt.format(seqcount, stats[0]+stats[1]+stats[2], stats[3], rest_count))

    return (seqcount, rest_count, stats[0], stats[1], stats[2], stats[3], all_truncations)

def _relativize_sequence_time_to_context(seq: list[int]) -> list[int]:
    # ensure the time token in the sequence is 0'd relative to the sequence itself
    # in a chunking context, the sequence length would be related to the context size
    time_shifted_seq = ops.translate(seq, -ops.min_time(seq, seconds=False), seconds=False)
    assert ops.min_time(time_shifted_seq, seconds=False) == 0
    return time_shifted_seq

def pack_sequences(events, controls, token_buf, z, start_seq_ordering: bool = True):
    # events and controls are already decided
    if start_seq_ordering:
        tokens, controls = ops.anticipate(events, controls)
        assert len(controls) == 0  # should have consumed all controls (because of padding)

        # add separator
        tokens = _add_anticipation_sep_prefix(tokens)
        token_buf.extend(tokens)

        return _get_chunks(
            token_buf,
            z,
            seq_chunk_size=EVENT_SIZE * M,
            max_time=MAX_TIME
        )
    else:
        pass

def _get_chunks(
    token_buf: list[int],
    sequence_type_token: int,
    seq_chunk_size: int,
    max_time: int
) -> tuple[list[list[int]], list[int], int, int]:
    num_seq = 0
    num_discarded_seq = 0
    chunks = []

    while len(token_buf) >= seq_chunk_size:
        curr_seq = token_buf[0:seq_chunk_size]
        token_buf = token_buf[seq_chunk_size:]

        # relativize time to the context
        curr_seq = _relativize_sequence_time_to_context(curr_seq)

        if ops.max_time(curr_seq, seconds=False) >= max_time:
            # sequence is too long
            num_discarded_seq += 1
            continue

        # if seq contains SEPARATOR, global controls describe the first sequence
        curr_seq.insert(0, sequence_type_token)
        chunks.append(curr_seq)
        num_seq += 1

    return chunks, token_buf, num_seq, num_discarded_seq