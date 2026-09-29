"""Wait for the MuJoCo 3.3.7 passive render thread before Python/GLFW teardown."""
import signal
import threading
from typing import Any


class ViewerClosed(KeyboardInterrupt):
    """The user closed the viewer; stop the rollout and clean up normally."""


def launch_viewer(model: Any, data: Any) -> tuple[Any, threading.Thread]:
    """Launch passive viewer and retain its otherwise unexposed render thread.

    MuJoCo 3.3.7 returns only a Handle: close() requests exit but does not join.
    Its Linux launcher creates a daemon Thread targeting _launch_internal. Keep
    that exact thread (not arbitrary threads) so GLFW atexit cannot race it.
    This small version-specific adapter is isolated from robot/control code.
    """
    from mujoco import viewer
    before = set(threading.enumerate())
    handle = viewer.launch_passive(model, data)
    threads = [t for t in threading.enumerate() if t not in before
               and getattr(t, '_target', None) is viewer._launch_internal]
    if len(threads) != 1:
        handle.close()
        raise RuntimeError('Cannot identify passive viewer thread; expected MuJoCo 3.3.7 Linux launcher')
    return handle, threads[0]


def close_viewer(handle: Any, thread: threading.Thread) -> None:
    """Request shutdown, then join while the model/data and handle are still alive."""
    # Repeated Ctrl+C must not abandon teardown and recreate the GLFW exit race.
    on_main_thread = threading.current_thread() is threading.main_thread()
    previous = signal.getsignal(signal.SIGINT) if on_main_thread else None
    if on_main_thread:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        handle.close()
        thread.join(timeout=5.0)
        if thread.is_alive():
            raise RuntimeError('MuJoCo render thread failed to shut down within 5 seconds')
    finally:
        if on_main_thread:
            signal.signal(signal.SIGINT, previous)
