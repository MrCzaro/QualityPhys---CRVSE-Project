"""Output names and provenance shared by the benchmark scripts.

A summary is only a result if its name says what it covers and its contents say what
produced it. A run on a subset - one split, some tasks, the first N recordings - writes
under a name that says so, so it can never replace the full run's committed summary.
Every summary records the command, the code's commit and the weights' hash.
"""
import hashlib
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]


def unique(names):
    """The estimator names in the order given, each once.

    A repeated --estimator would otherwise run the model twice into the same
    per-estimator rows and double every count in its summary.
    """
    return list(dict.fromkeys(names))


def subset_suffix(*parts):
    """Returns the filename suffix naming a subset run, '' for the full run.

    Each part is (label, value, full_value). A part is named only when its value differs
    from the full run's, so a full run keeps the plain names the committed summaries use.
    """
    return "".join(f"_{label}{value}" for label, value, full in parts
                   if value is not None and value != full)


def file_sha256(path, chunk=1 << 20):
    """SHA-256 of a file, read in chunks."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def git_state():
    """The repository's commit and the tracked files that differ from it; None without git."""
    def git(*args):
        return subprocess.run(["git", *args], cwd=_REPO_ROOT, capture_output=True,
                              text=True, check=True).stdout
    try:
        # Porcelain lines are "XY path"; X is often a space, so lines are cut, never stripped.
        changed = [line[3:] for line in
                   git("status", "--porcelain", "--untracked-files=no").splitlines() if line.strip()]
        return dict(commit=git("rev-parse", "HEAD").strip(), dirty=bool(changed),
                    changed=changed[:20])
    except (OSError, subprocess.CalledProcessError):
        return None


def provenance(estimator):
    """What produced a summary: the command, the code's commit, the weights and when."""
    checkpoint = getattr(estimator, "checkpoint", None)
    weights = None
    if checkpoint is not None and Path(checkpoint).is_file():
        weights = dict(file=Path(checkpoint).name, sha256=file_sha256(checkpoint),
                       bytes=Path(checkpoint).stat().st_size)
    return dict(command=[Path(sys.argv[0]).name, *sys.argv[1:]], git=git_state(),
                weights=weights,
                written_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"))