"""
A local arrival-time vocabulary to support long-context modeling 
"""

#
# configuaration
#

MIDI_QUANTIZATION = 100                  # time bins/second
MAX_TIME = 1*MIDI_QUANTIZATION           # 1 second local arrival times
MAX_DUR = 10*MIDI_QUANTIZATION           # 10 second maximum note duration
MAX_PITCH = 128                          # 128 MIDI pitches
MAX_INSTR = 128 + 1                      # 129 MIDI instruments (128 + drums)
MAX_NOTE = MAX_PITCH*MAX_INSTR           # note = pitch x instrument
MAX_DELTA = 100                          # maximum anticipation interval in seconds

#
# vocabulary
#

# the event block
EVENT_OFFSET = 0

TIME_OFFSET = EVENT_OFFSET + 0
DUR_OFFSET = EVENT_OFFSET + MAX_TIME
NOTE_OFFSET = DUR_OFFSET + MAX_DUR
TICK = NOTE_OFFSET + MAX_NOTE
EOS = TICK+1

# the control block
CONTROL_OFFSET = NOTE_OFFSET + MAX_NOTE + 2
ATIME_OFFSET = CONTROL_OFFSET + 0        # anticipated arrival times

GLOBAL_CONTROL_OFFSET = ATIME_OFFSET + MAX_TIME
CONTROL_END  = GLOBAL_CONTROL_OFFSET + 0
BOS = GLOBAL_CONTROL_OFFSET + 1
COLD_START = GLOBAL_CONTROL_OFFSET + 2
TRANSCRIPT = GLOBAL_CONTROL_OFFSET + 3
DELTA_OFFSET = GLOBAL_CONTROL_OFFSET + 4
VOCAB_SIZE = DELTA_OFFSET + MAX_DELTA

vocab = {
    'config' : {
        'name': 'local-midi',
        'midi_quantization' : MIDI_QUANTIZATION,
        'max_arrival' : MAX_TIME,
        'max_duration' : MAX_DUR,
        'size' : VOCAB_SIZE
    },

    # vocabulary offsets

    'event_offset' : EVENT_OFFSET,
    'time_offset' : TIME_OFFSET,
    'duration_offset' : DUR_OFFSET,
    'note_offset' : NOTE_OFFSET,
    'control_offset' : CONTROL_OFFSET,
    'global_control_offset' : CONTROL_OFFSET,

    # special "notes"

    'tick' : TICK,
    'sequence_end' : EOS,

    # global control tokens

    'control_end' : CONTROL_END,         # marks the end of the global control prefix

    'flags' : {                          # flags that are on appear in this order
        'sequence_start' : BOS,
        'cold_start' : COLD_START,
        'transcript' : TRANSCRIPT,
        'anticipation': DELTA_OFFSET,
    },
    
}

if __name__ == '__main__':
    print('MIDI Vocabulary Configuration:')
    print('  -> Local Arrival-time Tokenization') 
    print('  -> Combined Note Vocabulary note = (pitch, instrument)') 
    print('  -> Midi Quantization:', MIDI_QUANTIZATION)
    print('  -> Maximum Duration:', MAX_DUR)
    print('  -> Vocabulary Size:', VOCAB_SIZE)
    print('MIDI Training Sequence Format')
    print(80*'-')
    print('Midi Event Block:', EVENT_OFFSET)
    print('  -> arrival time offset :', TIME_OFFSET)
    print('  -> duration offset :', DUR_OFFSET)
    print('  -> note offset :', NOTE_OFFSET)
    print('    * tick :', TICK)
    print('    * eos:', EOS)
    print('  -> anticipated arrival time offset:', ATIME_OFFSET)
    print('Global Control Block:', GLOBAL_CONTROL_OFFSET)
    print('  -> end of control prefix:', CONTROL_END)
    print('  -> global control flags:')
    print('    * bos:', BOS)
    print('    * cold start:', COLD_START)
    print('    * transcript:', TRANSCRIPT)
    print(f'    * anticipation interval (range): {DELTA_OFFSET}-{DELTA_OFFSET+MAX_DELTA}')
