import heapq
from pathlib import Path
from typing import Any, Union

from symusic import Score, TimeUnit

from anticipation.v2.config import AnticipationV2Settings


def midi_to_compound_fast(
    midifile: Union[str, Path, bytes, Any],
    settings: AnticipationV2Settings,
) -> list[int]:
    """
    Uses:
      - per-track time ordering + k-way merge (O(N log K))
      - token preallocation using score.note_num() as an upper bound
    """


    time_resolution = settings.time_resolution
    score = Score(midifile, ttype=TimeUnit.second)

    events = []
    for track_idx, track in enumerate(score.tracks):
        instr = 128 if track.is_drum else int(track.program)
        for note_idx, note in enumerate(track.notes):
            vel = int(note.velocity)
            if vel <= 0: continue
            time_in_ticks = round(note.time * time_resolution)
            duration_ticks = round(note.duration * time_resolution)
            events.append( ( time_in_ticks, track_idx, note_idx, duration_ticks, int(note.pitch), instr, vel, ) )
            # approximate mido's merged-event iteration ordering (i.e. for message in midi)
            # sort by (onset time in ticks, track order, note order)
            events.sort(key=lambda x: (x[0], x[1], x[2]))
            tokens = []
            for event in events:
                time_in_ticks, _, __, duration_ticks, pitch, instr, vel = event
                tokens.extend([time_in_ticks, duration_ticks, pitch, instr, vel]) # 5-tuple of (time, duration, pitch, instrument, velocity)

    return tokens

    # pre alloc
    total_notes = score.note_num()
    tokens = [0] * (5 * total_notes)

    w = 0  # write cursor into tokens

    # heap item: (time_ticks, track_idx, tie_idx, duration_ticks, pitch, instr, vel)
    heap: list[tuple[int, int, int, int, int, int, int]] = []

    # For advancing within tracks we store each track as a list of (orig_idx, note), sorted by note.time.
    # We also maintain a position cursor per track in that sorted list.
    per_track_sorted: list[list[tuple[int, Any]]] = []
    pos: list[int] = []
    instr_by_track: list[int] = []

    for track_idx, track in enumerate(score.tracks):
        instr = 128 if track.is_drum else int(track.program)
        instr_by_track.append(instr)

        # Keep original note_idx for tie-breaking, but sort by time for mergeability.
        # order each by key (time, track_idx, note_idx),
        notes_with_idx = list(enumerate(track.notes))
        notes_with_idx.sort(key=lambda p: p[1].time)

        per_track_sorted.append(notes_with_idx)
        pos.append(0)

        # Push the first valid (vel>0) note for this track.
        j = 0
        nlen = len(notes_with_idx)
        while j < nlen:
            orig_idx, note = notes_with_idx[j]
            vel = int(note.velocity)
            if vel > 0:
                t = round(note.time * time_resolution)
                d = round(note.duration * time_resolution)
                heapq.heappush(
                    heap,
                    (
                        t,
                        -vel,
                        track_idx,
                        orig_idx,
                        d,
                        int(note.pitch),
                        instr,
                        vel,
                    ),
                )
                pos[track_idx] = j + 1
                break
            j += 1
        else:
            pos[track_idx] = nlen

    # Merge streams by (time, track_idx, orig_note_idx)
    while heap:
        t, neg_vel, track_idx, orig_idx, d, pitch, instr, vel = heapq.heappop(heap)

        tokens[w] = t
        tokens[w + 1] = d
        tokens[w + 2] = pitch
        tokens[w + 3] = instr
        tokens[w + 4] = vel
        w += 5

        # Advance in the same track: push next vel>0 note.
        notes_with_idx = per_track_sorted[track_idx]
        j = pos[track_idx]
        nlen = len(notes_with_idx)

        while j < nlen:
            orig_idx2, note2 = notes_with_idx[j]
            vel2 = int(note2.velocity)
            if vel2 > 0:
                t2 = round(note2.time * time_resolution)
                d2 = round(note2.duration * time_resolution)
                heapq.heappush(
                    heap,
                    (
                        t2,
                        track_idx,
                        orig_idx2,
                        d2,
                        int(note2.pitch),
                        instr_by_track[track_idx],
                        vel2,
                    ),
                )
                pos[track_idx] = j + 1
                break
            j += 1
        else:
            pos[track_idx] = nlen

    # Trim unused tail (covers vel<=0 skips and any slight note_num() overcount behavior)
    if w < len(tokens):
        del tokens[w:]

    # (time, duration, note, instrument, velocity)
    return tokens

import mido

from dataclasses import dataclass
from typing import Iterator, Any
from symusic import Score, TimeUnit


@dataclass
class MidiMessage:
    type: str
    time: int            # delta time (ticks)
    channel: int  = None
    note: int = None
    velocity: int = None
    program: int = None


def symusic_message_iterator(
    midifile,
    time_resolution: int,
) -> Iterator[MidiMessage]:
    """
    Emulates:
        for message in mido.MidiFile(midifile):

    using Symusic.

    Yields Mido-like MidiMessage objects with delta-time semantics.
    """

    score = Score(midifile, ttype=TimeUnit.second)


    events: list[tuple[int, int, int, MidiMessage]] = []
    seq = 0  # global tie-breaker

    # MIDI ordering convention:
    # program_change → note_on → note_off
    PRIORITY = {
        "program_change": 0,
        "note_on": 1,
        "note_off": 2,
    }

    for track_idx, track in enumerate(score.tracks):
        # this isn't quite right but symusic doesn't directly expose
        # the channel attribute of a message?
        channel = 9 if track.is_drum else track.program

        # Emit program change at time 0
        events.append(
            (
                0,
                PRIORITY["program_change"],
                seq,
                MidiMessage(
                    type="program_change",
                    time=0,
                    channel=channel,
                    program=track.program,
                ),
            )
        )
        seq += 1

        for note in track.notes:
            t_on = int(note.time * time_resolution)
            t_off = int((note.time + note.duration) * time_resolution)
            vel = int(note.velocity)

            # if vel <= 0:
            #     continue

            # note_on
            events.append(
                (
                    t_on,
                    PRIORITY["note_on"],
                    seq,
                    MidiMessage(
                        type="note_on",
                        time=0,
                        channel=channel,
                        note=int(note.pitch),
                        velocity=vel,
                    ),
                )
            )
            seq += 1

            if (note.time + note.duration) < track.end():
                # note_off (velocity 0 like Mido)
                events.append(
                    (
                        t_off,
                        PRIORITY["note_off"],
                        seq,
                        MidiMessage(
                            type="note_off",
                            time=0,
                            channel=channel,
                            note=int(note.pitch),
                            velocity=0,
                        ),
                    )
                )
                seq += 1

    # Sort like Mido's merged stream
    events.sort(key=lambda x: (x[0], x[1], x[2]))

    # Convert absolute → delta time
    last_time = 0
    for abs_time, _, _, msg in events:
        msg.time = abs_time - last_time
        last_time = abs_time
        yield msg


def midi_to_compound( midifile: Union[str, Path, bytes, Any], settings: AnticipationV2Settings ) -> list[int]:
    time_resolution = settings.time_resolution
    score = Score(midifile, ttype=TimeUnit.second)
    mido_score = mido.MidiFile(midifile)
    #mido_score.tracks
    # for track in mido_score.tracks:
    #     print(track)
    # for message in mido_score:
    #     print(message)
    #     pass
    all_events = []
    for x in symusic_message_iterator(midifile, 480):
        all_events.append(x)

    mido_events = []
    for msg in mido_score:
        mido_events.append(msg)

    # events = []
    # # for track_idx, track in enumerate(score.tracks):
    # #     instr = 128 if track.is_drum else int(track.program)
    # #     for note_idx, note in enumerate(track.notes):
    # for note in all_events:
    #     vel = int(note.velocity)
    #     if vel <= 0: continue
    #     time_in_ticks = round(note.time * time_resolution)
    #     duration_ticks = round(note.duration * time_resolution)
    #     #events.append( ( time_in_ticks, track_idx, note_idx, duration_ticks, int(note.pitch), instr, vel, ) )
    #     # approximate mido's merged-event iteration ordering (i.e. for message in midi)
    #     # sort by (onset time in ticks, track order, note order)
    #     #events.sort(key=lambda x: (x[0], x[1], x[2]))
    #     tokens = []
    #     for event in events:
    #         time_in_ticks, _, __, duration_ticks, pitch, instr, vel = event
    #         tokens.extend([time_in_ticks, duration_ticks, pitch, instr, vel]) # 5-tuple of (time, duration, pitch, instrument, velocity)
    #
    from collections import defaultdict
    tokens = []
    note_idx = 0
    open_notes = defaultdict(list)
    debug=True

    time = 0
    instruments = defaultdict(int) # default to code 0 = piano
    tempo = 500000 # default tempo: 500000 microseconds per beat
    for message in all_events:
        time += message.time

        # sanity check: negative time?
        if message.time < 0:
            raise ValueError

        if message.type == 'program_change':
            instruments[message.channel] = message.program
        elif message.type in ['note_on', 'note_off']:
            # special case: channel 9 is drums!
            instr = 128 if message.channel == 9 else instruments[message.channel]

            if message.type == 'note_on' and message.velocity > 0: # onset
                # time quantization
                time_in_ticks = round(time_resolution * time / 480) #round(time_resolution*time)

                # Our compound word is: (time, duration, note, instr, velocity)
                tokens.append(time_in_ticks) # 5ms resolution
                tokens.append(-1) # placeholder (we'll fill this in later)
                tokens.append(message.note)
                tokens.append(instr)
                tokens.append(message.velocity)

                open_notes[(instr,message.note,message.channel)].append((note_idx, time))
                note_idx += 1
            else: # offset
                try:
                    open_idx, onset_time = open_notes[(instr,message.note,message.channel)].pop(0)
                except IndexError:
                    if debug:
                        print('WARNING: ignoring bad offset')
                else:
                    # time_it = (time_resolution * time / 480)
                    # time_z = time_resolution * (onset_time / 480)
                    duration_ticks = round(time_resolution * (time - onset_time)/480)
                    tokens[5*open_idx + 1] = duration_ticks
                    #del open_notes[(instr,message.note,message.channel)]
        elif message.type == 'set_tempo':
            tempo = message.tempo
        elif message.type == 'time_signature':
            pass # we use real time
        elif message.type in ['aftertouch', 'polytouch', 'pitchwheel', 'sequencer_specific']:
            pass # we don't attempt to model these
        elif message.type == 'control_change':
            pass # this includes pedal and per-track volume: ignore for now
        elif message.type in ['track_name', 'text', 'end_of_track', 'lyrics', 'key_signature',
                              'copyright', 'marker', 'instrument_name', 'cue_marker',
                              'device_name', 'sequence_number']:
            pass # possibly useful metadata but ignore for now
        elif message.type == 'channel_prefix':
            pass # relatively common, but can we ignore this?
        elif message.type in ['midi_port', 'smpte_offset', 'sysex']:
            pass # I have no idea what this is
        else:
            if debug:
                print('UNHANDLED MESSAGE', message.type, message)

    unclosed_count = 0
    for _,v in open_notes.items():
        unclosed_count += len(v)

    if debug and unclosed_count > 0:
        print(f'v2 WARNING: {unclosed_count} unclosed notes')
        print('  ', midifile)

    return tokens