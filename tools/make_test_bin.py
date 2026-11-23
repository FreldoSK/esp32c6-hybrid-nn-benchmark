"""Convert main/door_sensor_dataset_full.csv into compact binaries embedded in the firmware.

Output files:
    main/test_x.bin   int16 little-endian, N x 400 samples, value = round(x * 8192)
    main/test_y.bin   uint8, N labels (0=No passage, 1=Entry, 2=Exit, 3=Walk-by)

Usage:
    python tools/make_test_bin.py [max_samples]
"""

import csv
import struct
import sys
from pathlib import Path

CLASSES = ["No passage", "Entry", "Exit", "Walk-by"]
SCALE = 8192  # must match kInputScale in main/main.cpp
STEPS = 400
INT16_MAX = 32767

MAIN_DIR = Path(__file__).resolve().parent.parent / "main"
CSV_PATH = MAIN_DIR / "door_sensor_dataset_full.csv"
X_PATH = MAIN_DIR / "test_x.bin"
Y_PATH = MAIN_DIR / "test_y.bin"


def encode_row(row: list[str], label_col: int, first_x_col: int, row_number: int) -> tuple[int, bytes]:
    """Return (label index, packed int16 samples) for one CSV row."""
    label = CLASSES.index(row[label_col])
    values = [round(float(v) * SCALE) for v in row[first_x_col:first_x_col + STEPS]]

    if len(values) != STEPS:
        sys.exit(f"row {row_number}: expected {STEPS} values, got {len(values)}")
    if max(map(abs, values)) > INT16_MAX:
        sys.exit(f"row {row_number}: value out of int16 range")

    return label, struct.pack(f"<{STEPS}h", *values)


def main() -> None:
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
    xs, ys = bytearray(), bytearray()

    with open(CSV_PATH, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        label_col = header.index("label")
        first_x_col = header.index("x_000")

        for row_number, row in enumerate(reader):
            if limit is not None and row_number >= limit:
                break
            label, samples = encode_row(row, label_col, first_x_col, row_number)
            ys.append(label)
            xs += samples

    X_PATH.write_bytes(xs)
    Y_PATH.write_bytes(ys)
    print(f"wrote {len(ys)} samples ({len(xs)} B x, {len(ys)} B y)")


if __name__ == "__main__":
    main()
