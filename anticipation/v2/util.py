from typing import Iterator, Any, ContextManager

import os
import hashlib
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path
import tempfile
from contextlib import contextmanager

import random

import numpy as np
import torch
import os
import shutil
import tempfile
from contextlib import AbstractContextManager
from pathlib import Path


def set_seed(seed: int):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def save_text(text_file_path: Path, my_text: str) -> None:
    ddp_rank = int(os.environ.get("RANK", 0))
    if ddp_rank == 0:
        text_file_path.write_text(my_text, encoding="utf-8")


@contextmanager
def temporary_directory(
    env_var: str = "CUSTOM_TMP_DIR",
    require_exists: bool = True,
    check_writable: bool = True,
) -> ContextManager[str]:
    root = os.environ.get(env_var)
    if not root:
        # if the envar is not set, default to the regular behavior
        # the temporary folder is OS/environment dependent
        with tempfile.TemporaryDirectory() as td:
            yield td
    else:
        root_path = Path(root).expanduser().resolve()

        if require_exists and not root_path.exists():
            raise RuntimeError(f"{env_var}={root_path} does not exist")

        if check_writable and not os.access(root_path, os.W_OK):
            raise RuntimeError(f"{env_var}={root_path} is not writable")

        with tempfile.TemporaryDirectory(dir=root_path) as td:
            yield td



class AtomicDirectory(AbstractContextManager["AtomicDirectory"]):
    """
    Stage output into a temporary directory and atomically promote it to `final_path`
    only if the `with` block exits without exception.

    Typical usage:
        with AtomicDirectory("output/checkpoint") as txn:
            (txn.path / "config.json").write_text("...")
            (txn.path / "weights.bin").write_bytes(b"...")
        # here output/checkpoint has been atomically replaced

    Behavior:
    - Work happens in a temp directory under `temp_parent` if provided,
      otherwise next to `final_path` so promotion can use atomic rename.
    - On success:
        - if `overwrite=True`, replaces any existing final directory atomically
        - if `overwrite=False`, raises FileExistsError if final_path exists
    - On failure:
        - deletes the temp directory by default
        - if `keep_temp_on_error=True`, leaves it behind for inspection
    """

    def __init__(
        self,
        final_path: str | Path,
        *,
        temp_parent: str | Path | None = None,
        overwrite: bool = False,
        keep_temp_on_error: bool = False,
        chmod: int | None = None,
    ) -> None:
        self.final_path = Path(final_path).resolve()
        self.temp_parent = (
            Path(temp_parent).resolve()
            if temp_parent is not None
            else self.final_path.parent
        )
        self.overwrite = overwrite
        self.keep_temp_on_error = keep_temp_on_error
        self.chmod = chmod

        self._temp_dir: Path | None = None
        self._backup_dir: Path | None = None

    @property
    def path(self) -> Path:
        if self._temp_dir is None:
            raise RuntimeError("Context has not been entered yet.")
        return self._temp_dir

    def __enter__(self) -> "AtomicDirectory":
        self.temp_parent.mkdir(parents=True, exist_ok=True)

        # Put the temp dir under temp_parent so final promotion can be done
        # with an atomic rename when temp_parent is on the same filesystem.
        temp_dir_str = tempfile.mkdtemp(
            prefix=f".{self.final_path.name}.tmp.",
            dir=str(self.temp_parent),
        )
        self._temp_dir = Path(temp_dir_str)

        if self.chmod is not None:
            os.chmod(self._temp_dir, self.chmod)

        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        assert self._temp_dir is not None
        temp_dir = self._temp_dir

        if exc_type is not None:
            if not self.keep_temp_on_error:
                shutil.rmtree(temp_dir, ignore_errors=True)
            return False

        final_path = self.final_path
        backup_path = final_path.with_name(f".{final_path.name}.old")

        if final_path.exists():
            if not self.overwrite:
                shutil.rmtree(temp_dir, ignore_errors=True)
                raise FileExistsError(f"Destination already exists: {final_path}")

            if backup_path.exists():
                if backup_path.is_dir():
                    shutil.rmtree(backup_path)
                else:
                    backup_path.unlink()

            # Move existing final directory aside, then promote temp into place.
            os.replace(final_path, backup_path)
            try:
                os.replace(temp_dir, final_path)
            except Exception:
                # Best-effort rollback.
                os.replace(backup_path, final_path)
                raise
            else:
                if backup_path.is_dir():
                    shutil.rmtree(backup_path, ignore_errors=True)
                else:
                    backup_path.unlink(missing_ok=True)
        else:
            os.replace(temp_dir, final_path)

        return False

def get_book_keeping_info() -> dict[str, Any]:
    return {
        "uuid": get_uuid_string(),
        "git_info": get_git_info(),
        "started_time": get_time_info(),
    }


def get_uuid_string() -> str:
    return str(uuid.uuid4().hex)


def get_time_info() -> dict[str, str]:
    dt = datetime.now(timezone.utc)
    ts = dt.strftime("%Y_%m_%d_%H_%M_%S")
    return {
        "readable_time": ts,
        "unix_time": str(dt.timestamp()),
        "timezone": "utc",
    }


def get_git_info(path: Path = None) -> dict[str, str]:
    if path is None:
        cwd = Path(".")
    else:
        cwd = path

    def run(cmd):
        return subprocess.check_output(
            cmd, cwd=cwd, stderr=subprocess.DEVNULL, text=True
        ).strip()

    try:
        run(["git", "rev-parse", "--is-inside-work-tree"])
        commit = run(["git", "rev-parse", "HEAD"])
        branch = run(["git", "branch", "--show-current"])
        return {
            "branch": branch,
            "commit": commit,
        }
    except subprocess.CalledProcessError:
        return {
            "branch": "",
            "commit": "",
        }


def iter_files(
    root: Path, file_extensions: tuple[str, ...], follow_symlinks: bool = False
) -> Iterator[Path]:
    """
    This ordering is NOT deterministic!
    """
    extensions_to_get = {
        e.lower() if e.startswith(".") else f".{e.lower()}" for e in file_extensions
    }
    stack = [os.fspath(root)]
    while stack:
        d = stack.pop()
        try:
            with os.scandir(d) as it:
                for entry in it:
                    if entry.is_dir(follow_symlinks=follow_symlinks):
                        stack.append(entry.path)
                        continue

                    if entry.is_file(follow_symlinks=follow_symlinks):
                        _, ext = os.path.splitext(entry.name)
                        if ext.lower() in extensions_to_get:
                            yield Path(entry.path)
        except (FileNotFoundError, PermissionError, NotADirectoryError):
            continue


def get_md5_of_string(input_str: str) -> str:
    # assuming the input string uses utf-8 encoding
    return hashlib.md5(bytes(input_str, encoding="utf8")).hexdigest()
