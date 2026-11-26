"""Pure-Python project data path resolution, without mounting cloud drives.

This module has no import-time side effects. Existing notebooks can retain
their Colab mount logic while using NBA_DATA_DIR to select a custom folder.
"""
from __future__ import annotations

import os
from pathlib import Path

DEFAULT_CANDIDATES = (
    Path("/content/drive/MyDrive/NBA_Data_Science"),
    Path("/content/gdrive_nba/MyDrive/NBA_Data_Science"),
    Path("/content/nba_drive/MyDrive/NBA_Data_Science"),
)


def resolve_data_dir(
    override: str | None = None,
    *,
    candidates: tuple[Path, ...] = DEFAULT_CANDIDATES,
    require_database: bool = False,
) -> Path:
    """Resolve the data directory without creating or changing any files.

    Resolution order: explicit argument, NBA_DATA_DIR, an existing database
    in a known Colab mount, an existing directory, then the default path.
    """
    supplied = override if override is not None else os.environ.get("NBA_DATA_DIR")
    if supplied:
        result = Path(supplied).expanduser()
    else:
        result = next(
            (p for p in candidates if (p / "nba_database.db").is_file()), None
        )
        if result is None:
            result = next((p for p in candidates if p.is_dir()), None)
        if result is None:
            if not candidates:
                raise ValueError("At least one candidate path is required")
            result = candidates[0]
    if require_database and not (result / "nba_database.db").is_file():
        raise FileNotFoundError(
            f"NBA database missing in {result}. Run data collection first."
        )
    return result
