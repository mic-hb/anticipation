from enum import Enum

# type aliases
Token = int
MIDITick = int
MIDINote = int
MIDIProgramCode = int

Triplet = tuple[Token, Token, Token]
TickToken = tuple[Token]


class MIDIFileIgnoredReason(Enum):
    TOO_FEW_EVENTS = 0
    TOO_FEW_SECONDS = 1
    TOO_MANY_SECONDS = 2
    TOO_MANY_INSTRUMENTS = 3
