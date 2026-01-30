from dataclasses import dataclass

import anticipation.vocab as v1_vocab


@dataclass
class Vocab:
    SEPARATOR: int = v1_vocab.SEPARATOR
    CONTROL_OFFSET: int = v1_vocab.CONTROL_OFFSET
    NOTE_OFFSET: int = v1_vocab.NOTE_OFFSET
    DUR_OFFSET: int = v1_vocab.DUR_OFFSET
    TIME_OFFSET: int = v1_vocab.TIME_OFFSET

    SPECIAL_OFFSET: int = v1_vocab.SPECIAL_OFFSET

    ATIME_OFFSET: int = v1_vocab.ATIME_OFFSET
    ANOTE_OFFSET: int = v1_vocab.ANOTE_OFFSET

    # I am calling these two tokens the 'flag token'
    # these are basically BOS tokens
    ANTICIPATE: int = v1_vocab.ANTICIPATE
    AUTOREGRESS: int = v1_vocab.AUTOREGRESS

    REST: int = v1_vocab.REST

    _last_token_in_v1: int = v1_vocab.VOCAB_SIZE

    # add the number of new blocks in v2 here
    # added:
    # - PAD
    # - METRONOME
    VOCAB_SIZE: int = v1_vocab.VOCAB_SIZE + 2

    # https://github.com/jthickstun/anticipation/blob/6927699c5243fd91d1d252211c29885377d9dda5/train/tokenize-new.py#L33
    # https://github.com/jthickstun/anticipation/blob/6927699c5243fd91d1d252211c29885377d9dda5/anticipation/vocabs/localmidi.py#L65
    # aka 'tick'
    METRONOME: int = _last_token_in_v1 + 1
    PAD: int = _last_token_in_v1 + 2


@dataclass
class AnticipationV2Settings:
    """
    Object for holding 'global'-like settings. If a property is frequently referenced
    by several functions, then it is a good candidate to put here.
    """

    vocab: Vocab
    compound_size: int = 5
    time_resolution: int = 100
    debug: bool = False
    min_track_events: int = 100
    min_track_time_in_seconds: int = 10
    max_track_time_in_seconds: int = 3600
    max_track_instruments: int = 16
    max_midi_pitch: int = 128
    max_note_duration_in_seconds: int = 10
    use_metronome_token: bool = False
    context_size: int = 1024
    event_size: int = 3
    m: int = 341

    # original data mixture:
    # - 10% without anticipation (standard AR)
    # - 10% span anticipation
    # - 40% instrument anticipation
    # - 40% random anticipation
    num_autoregressive_seq_per_midi_file: int = 1
    num_span_anticipation_augmentations_per_midi_file: int = 1
    num_instrument_anticipation_augmentations_per_midi_file: int = 4
    num_random_anticipation_augmentations_per_midi_file: int = 4
    span_anticipation_lambda: float = 0.05
    do_relativize_time_to_ctx: bool = True

    # anticipation interval
    delta: int = 5
