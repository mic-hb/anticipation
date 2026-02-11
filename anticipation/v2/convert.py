from pathlib import Path
from operator import itemgetter

from symusic import Score, TimeUnit

from anticipation.v2.config import AnticipationV2Settings


def midi_to_compound(midifile: Path, settings: AnticipationV2Settings) -> list[int]:
    time_resolution = settings.time_resolution
    score = Score(midifile, ttype=TimeUnit.second)
    compounds = []
    for track_idx, track in enumerate(score.tracks):
        instr = 128 if track.is_drum else track.program
        for note in track.notes:
            on_set_time_in_ticks = round(time_resolution * note.time)
            duration_time_in_ticks = round(time_resolution * note.duration)
            pitch = int(note.pitch)
            velocity = int(note.velocity)
            # (abs_time, onset, duration, pitch, instrument, velocity)
            # in mido, tracks are 'merged' by converting each event to
            # absolute time in ticks, and then sorting by that absolute value
            compounds.append(
                (
                    # need to sort by unquantized time for parity with v1
                    note.time,
                    # --- these are properties we keep for tokenization ---
                    on_set_time_in_ticks,
                    duration_time_in_ticks,
                    pitch,
                    instr,
                    velocity,
                )
            )

    # mimic mido's sort behavior
    # get the 0th element from the compound, could be faster than a lambda?
    compounds.sort(key=itemgetter(0))

    # remove the absolute time, just return the quantized one
    tokens = [x for b in compounds for x in b[1:]]
    return tokens
