"""Download and extract the UCI household power-consumption dataset."""

from __future__ import annotations

import argparse
import hashlib
import shutil
import urllib.request
import zipfile
from pathlib import Path

DATASET_URL = (
    "https://archive.ics.uci.edu/static/public/235/"
    "individual+household+electric+power+consumption.zip"
)
ARCHIVE_NAME = "individual_household_electric_power_consumption.zip"
DATA_FILE = "household_power_consumption.txt"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download(destination: Path, force: bool = False) -> Path:
    destination.mkdir(parents=True, exist_ok=True)
    output = destination / DATA_FILE
    if output.exists() and not force:
        print(f"Dataset already exists: {output}")
        return output

    archive = destination / ARCHIVE_NAME
    print(f"Downloading official UCI archive to {archive} ...")
    with urllib.request.urlopen(DATASET_URL, timeout=120) as response:
        with archive.open("wb") as target:
            shutil.copyfileobj(response, target)

    with zipfile.ZipFile(archive) as zipped:
        member = next(
            (name for name in zipped.namelist() if name.endswith(DATA_FILE)),
            None,
        )
        if member is None:
            raise RuntimeError(f"{DATA_FILE} was not found in the downloaded archive.")
        with zipped.open(member) as source, output.open("wb") as target:
            shutil.copyfileobj(source, target)

    print(f"Saved: {output}")
    print(f"Rows including header: {sum(1 for _ in output.open(encoding='utf-8')):,}")
    print(f"SHA-256: {sha256(output)}")
    archive.unlink(missing_ok=True)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--destination",
        type=Path,
        default=Path("data/raw"),
        help="Destination directory (default: data/raw)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing extracted dataset.",
    )
    args = parser.parse_args()
    download(args.destination, args.force)


if __name__ == "__main__":
    main()
