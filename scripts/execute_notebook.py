"""Execute notebooks/batch_experiments.ipynb headlessly and save outputs.

Run:  python scripts/execute_notebook.py
"""

import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
NB_PATH = ROOT / "notebooks" / "batch_experiments.ipynb"


def main() -> None:
    nb = nbformat.read(NB_PATH, as_version=4)
    client = NotebookClient(
        nb,
        timeout=3600,
        kernel_name="python3",
        resources={"metadata": {"path": str(ROOT / "notebooks")}},
    )
    client.execute()
    nbformat.write(nb, NB_PATH)
    print(f"executed and saved {NB_PATH}")


if __name__ == "__main__":
    main()
