from pathlib import Path
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def output_dir(*parts: Iterable[str]) -> Path:
    return ensure_dir(OUTPUTS_DIR.joinpath(*parts))


def data_dir(*parts: Iterable[str]) -> Path:
    return DATA_DIR.joinpath(*parts)
