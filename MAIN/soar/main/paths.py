
import os
from pathlib import Path


def get_project_root() -> Path:
    override = os.environ.get("SEMICONDUCTOR_PROJECT_ROOT", "").strip()
    if override:
        return Path(override).resolve()
    return Path(__file__).resolve().parents[1]


PROJECT_ROOT = get_project_root()
