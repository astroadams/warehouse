"""Provenance sidecar files for pipeline outputs.

Each pipeline script writes a ``<output>.provenance.json`` file alongside its
primary output after a fresh build, and checks it when reusing a cached file.
The sidecar records the git commit, dirty-tree status, config hash, and a
timestamp so you can tell at a glance whether a cached output was built from
the current codebase.

Usage in scripts::

    from warehouse_growth import provenance

    # Reusing a cached output — warn if stale:
    provenance.check(output_path)

    # After writing a new output:
    provenance.write(output_path, config_path)
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Resolve the repo root from this file's location so git commands work
# regardless of the caller's working directory.
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def _git(*args: str) -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(_REPO_ROOT), *args],
            capture_output=True,
            text=True,
        ).stdout.strip()
    except FileNotFoundError:
        return ""


def sidecar_path(output_path: Path) -> Path:
    """Return the ``.provenance.json`` path alongside an output file."""
    return output_path.parent / (output_path.stem + ".provenance.json")


def write(output_path: Path, config_path: Path | None = None, **extra: Any) -> None:
    """Write a provenance sidecar alongside *output_path*.

    Records the current git commit, dirty-tree status, ISO timestamp, and an
    optional config file path + hash.  Any additional keyword arguments are
    included verbatim (e.g. ``epoch="2022"``, ``n_patches=1186``).
    """
    commit = _git("rev-parse", "HEAD")
    dirty = bool(_git("status", "--porcelain"))
    meta: dict[str, Any] = {
        "commit": commit or "unknown",
        "dirty": dirty,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    if config_path is not None:
        meta["config"] = str(config_path)
        meta["config_hash"] = hashlib.sha256(
            Path(config_path).read_bytes()
        ).hexdigest()[:12]
    meta.update(extra)
    sidecar_path(output_path).write_text(json.dumps(meta, indent=2))


def read(output_path: Path) -> dict[str, Any] | None:
    """Return the provenance metadata for *output_path*, or ``None`` if absent."""
    p = sidecar_path(output_path)
    return json.loads(p.read_text()) if p.exists() else None


def check(output_path: Path) -> None:
    """Print a warning to stdout if *output_path*'s provenance is stale.

    Stale means either: built from a dirty working tree, or built at a
    different commit than the current HEAD.  Silent when fresh or when no
    sidecar exists (e.g. outputs predating this feature).
    """
    meta = read(output_path)
    if meta is None:
        return
    name = output_path.name
    if meta.get("dirty"):
        print(
            f"  [provenance] {name}: built from uncommitted changes"
            f" at {meta.get('commit', 'unknown')[:8]}"
        )
        return
    current = _git("rev-parse", "HEAD")
    if current and meta.get("commit") and meta["commit"] != current:
        print(
            f"  [provenance] {name}: built at {meta['commit'][:8]},"
            f" HEAD is now {current[:8]}"
        )
