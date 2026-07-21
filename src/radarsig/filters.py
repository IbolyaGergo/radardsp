import tomllib
import numpy as np
from pathlib import Path

def load_filter_from_toml(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    with open(path, "rb") as f:
        data = tomllib.load(f)

    b=np.array(data["b"])
    a=np.array(data["a"])

    return b, a
