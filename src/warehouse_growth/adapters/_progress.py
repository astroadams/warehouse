from __future__ import annotations

import threading
from typing import Callable, TypeVar

T = TypeVar("T")


def run_with_spinner(desc: str, fn: Callable[[], T]) -> T:
    """Run a blocking zero-argument callable while displaying an elapsed-time spinner.

    The DuckDB S3 scan and Overpass HTTP request are both fully opaque blocking
    calls — there's no progress callback. This runs them in a daemon thread and
    refreshes a tqdm bar every 250ms so the terminal doesn't appear frozen.

    Falls back to a plain print if tqdm is not installed.
    """
    try:
        from tqdm import tqdm
    except ImportError:
        print(f"{desc} …", flush=True)
        return fn()

    result: list[T] = []
    exc: list[BaseException | None] = [None]
    done = threading.Event()

    def _worker() -> None:
        try:
            result.append(fn())
        except BaseException as e:
            exc[0] = e
        finally:
            done.set()

    threading.Thread(target=_worker, daemon=True).start()

    with tqdm(
        desc=desc,
        total=None,
        bar_format="{desc} [{elapsed}]",
        mininterval=0.25,
        leave=True,
    ) as bar:
        while not done.wait(0.25):
            bar.refresh()

    if exc[0] is not None:
        raise exc[0]  # type: ignore[misc]
    return result[0]
