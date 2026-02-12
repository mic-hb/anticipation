from pathlib import Path

import numpy as np

from anticipation.v2.types import Token


class TokenSequenceBinaryFile:
    """Buffered IO for integer sequences.

    This was original within the `SequencePacker` object, but that has its own
    buffer, and this also has its own buffer, and it got confusing.

    This handles writing binary data to disk when the buffer reaches some number of sequences
    and should be closed like a file when done.
    """

    def __init__(
        self,
        path: Path,
        seq_len: int,
        vocab_size: int,
        flush_every_n_sequences: int = 8_192,
        io_buffer_bytes: int = 8 * 1024 * 1024,
    ):
        self.vocab_size = vocab_size
        if self.vocab_size <= (2**16):
            self.dtype = np.uint16
        elif self.vocab_size <= (2**32):
            self.dtype = np.uint32
        elif self.vocab_size <= (2**64):
            self.dtype = np.uint64
        else:
            raise ValueError("Vocab too large.")

        self.path = path
        self.seq_len = seq_len
        self.flush_every_n_sequences = flush_every_n_sequences
        self.io_buffer_bytes = io_buffer_bytes
        self._f = open(path, "ab", buffering=io_buffer_bytes)
        self._buf = np.empty(
            (self.flush_every_n_sequences, self.seq_len),
            dtype=self.dtype,
        )
        self.i = 0

    def append(self, tokens: list[Token]) -> None:
        a = np.asarray(tokens, dtype=self.dtype)
        if not a.flags["C_CONTIGUOUS"]:
            a = np.ascontiguousarray(a)
        self._buf[self.i, :] = a
        self.i += 1
        if self.i == self.flush_every_n_sequences:
            self._buf.tofile(self._f)
            self.i = 0

    def close(self) -> None:
        if self.i:
            # write the remaining sequences
            self._buf[: self.i].tofile(self._f)
            self.i = 0
        self._f.flush()
        self._f.close()
