import sys
import warnings
from pathlib import Path

from nbdev.processors import NBProcessor, _do_eval


def check_nb(nb_path: str) -> None:
    with warnings.catch_warnings(record=True) as issued_warnings:
        NBProcessor(nb_path, _do_eval, process=True)
    if any(
        "Found cells containing imports and other code" in str(w)
        for w in issued_warnings
    ):
        print(f"{nb_path} has cells containing imports and code.")
        sys.exit(1)


if __name__ == "__main__":
    repo_root = Path(__file__).parents[1]
    for nb_path in (repo_root / "nbs").glob("*.ipynb"):
        check_nb(str(nb_path))
