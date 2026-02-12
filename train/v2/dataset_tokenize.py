from typing import Iterator
from pathlib import Path
import os


from anticipation.v2.config import AnticipationV2Settings, Vocab
from anticipation.v2.tokenize import tokenize

MIDI_EXTS = (".mid", ".midi")

def iter_files(root: Path, file_extensions: tuple[str, ...]) -> Iterator[Path]:
    extensions_to_get = {e.lower() if e.startswith(".") else f".{e.lower()}" for e in file_extensions}
    stack = [os.fspath(root)]
    while stack:
        d = stack.pop()
        try:
            with os.scandir(d) as it:
                for entry in it:
                    if entry.is_dir(follow_symlinks=False):
                        stack.append(entry.path)
                        continue

                    if entry.is_file(follow_symlinks=False):
                        _, ext = os.path.splitext(entry.name)
                        if ext.lower() in extensions_to_get:
                            yield Path(entry.path)
        except (FileNotFoundError, PermissionError, NotADirectoryError):
            continue

if __name__ == "__main__":
    dataset_path = Path(__file__).parent.parent.parent / "data"
    all_files = iter_files(dataset_path, file_extensions=(".mid", ".midi"))
    total_files = sum(1 for _ in iter_files(dataset_path, file_extensions=(".mid", ".midi")))
    settings = AnticipationV2Settings(
        vocab=Vocab(),
        num_autoregressive_seq_per_midi_file=1,
        num_span_anticipation_augmentations_per_midi_file=0,
        num_instrument_anticipation_augmentations_per_midi_file=0,
        num_random_anticipation_augmentations_per_midi_file=0,
        debug=False,
    )
    tokenize(all_files, output=Path("./dataset_processed.bin"), settings=settings, total_files=total_files)