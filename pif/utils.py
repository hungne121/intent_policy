"""Path and run-folder helpers of PIF (same behaviour as intent_policy.utils, without importing the simulator, so the
PIF package runs on a machine that has no MuJoCo)."""
from __future__ import annotations
import shlex
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def run_dir(kind: str, name: str, output_dir: str | Path | None = None) -> Path:
    """`output_dir` when given, else outputs/<kind>/<YYYY-MM-DD>/<HH-MM-SS>_<name> (LeRobot-style run folders)."""
    if output_dir:
        return resolve(output_dir)
    now = datetime.now()
    return ROOT / 'outputs' / kind / now.strftime('%Y-%m-%d') / f"{now.strftime('%H-%M-%S')}_{name}"


class _Tee:
    """A console stream that also writes everything into the run's log file."""

    def __init__(self, stream, log):
        self.stream, self.log = stream, log

    def write(self, s: str) -> int:
        self.stream.write(s)
        self.log.write(s)
        return len(s)

    def flush(self) -> None:
        self.stream.flush()
        self.log.flush()

    def __getattr__(self, name):
        return getattr(self.stream, name)


_LOGGED: set[Path] = set()


def log_run(out: Path, module: str | None = None) -> None:
    """Keep this process's console output in <out>/log.txt; with `module`, also append the command to command.txt."""
    out.mkdir(parents=True, exist_ok=True)
    path = (out / 'log.txt').resolve()
    if path not in _LOGGED:
        _LOGGED.add(path)
        log = open(path, 'a', buffering=1)
        sys.stdout, sys.stderr = _Tee(sys.stdout, log), _Tee(sys.stderr, log)
    if module:
        with open(out / 'command.txt', 'a') as f:
            f.write(f"{datetime.now().isoformat(timespec='seconds')}  python -m {module} {shlex.join(sys.argv[1:])}\n")
