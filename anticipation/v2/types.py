from enum import Enum

# type aliases
Token = int
MIDIProgramCode = int


class MIDIFileIgnoredReason(Enum):
    TOO_FEW_EVENTS = 0
    TOO_FEW_SECONDS = 1
    TOO_MANY_SECONDS = 2
    TOO_MANY_INSTRUMENTS = 3
