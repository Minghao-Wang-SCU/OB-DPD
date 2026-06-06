#!/usr/bin/env python3
"""Filter a LAMMPS text dump to selected atom ids.

For OB-DPD surfactant validation runs, water beads are created by LAMMPS after
reading the solute data file. Micelle analysis only needs surfactant beads, so a
one-time filtered dump is much smaller and can be reused for threshold scans.
"""

import argparse
import re
import sys
from pathlib import Path

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover - optional dependency
    tqdm = None


class SimpleProgress:
    def __init__(self, total=None, desc="filter", disable=False):
        self.total = total
        self.desc = desc
        self.disable = disable
        self.count = 0

    def __enter__(self):
        if not self.disable:
            self._write()
        return self

    def __exit__(self, exc_type, exc, tb):
        if not self.disable:
            sys.stderr.write("\n")
            sys.stderr.flush()

    def update(self, step=1):
        self.count += step
        self._write()

    def _write(self):
        if self.disable:
            return
        if self.total:
            pct = 100.0 * self.count / self.total
            text = f"\r{self.desc}: {self.count}/{self.total} frames ({pct:5.1f}%)"
        else:
            text = f"\r{self.desc}: {self.count} frames"
        sys.stderr.write(text)
        sys.stderr.flush()


def progress_bar(total=None, desc="filter", disable=False):
    if tqdm is not None:
        return tqdm(total=total, desc=desc, unit="frame", disable=disable)
    return SimpleProgress(total=total, desc=desc, disable=disable)


def read_data_atom_ids(data_path):
    atom_ids = []
    in_atoms = False
    with Path(data_path).open(errors="ignore") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped == "Atoms":
                in_atoms = True
                next(handle, None)
                continue
            if not in_atoms:
                continue
            if not stripped:
                continue
            if re.match(r"^[A-Za-z]", stripped):
                break
            parts = stripped.split()
            if parts:
                atom_ids.append(int(parts[0]))
    return set(atom_ids)


def copy_filtered_dump(
    input_path,
    output_path,
    selected_ids,
    start_timestep=None,
    frame_stride=1,
    max_frames=None,
    no_progress=False,
):
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    seen_kept_frames = 0
    with input_path.open(errors="ignore") as src, output_path.open("w") as dst, progress_bar(
        total=max_frames,
        desc=f"filter {input_path.name}",
        disable=no_progress,
    ) as pbar:
        while True:
            line = src.readline()
            while line and not line.startswith("ITEM: TIMESTEP"):
                line = src.readline()
            if not line:
                break
            timestep = int(src.readline().strip())
            number_header = src.readline()
            if not number_header.startswith("ITEM: NUMBER OF ATOMS"):
                raise ValueError("unexpected dump format: missing NUMBER OF ATOMS")
            n_atoms = int(src.readline().strip())
            box_header = src.readline()
            bounds = [src.readline() for _ in range(3)]
            atom_header = src.readline()
            columns = atom_header.split()[2:]
            if "id" not in columns:
                raise ValueError(f"dump atom columns must contain id, got {columns}")
            id_col = columns.index("id")

            keep_frame = start_timestep is None or timestep >= start_timestep
            if keep_frame:
                keep_frame = (seen_kept_frames % frame_stride) == 0
                seen_kept_frames += 1

            kept_lines = []
            for _ in range(n_atoms):
                atom_line = src.readline()
                if not keep_frame:
                    continue
                parts = atom_line.split()
                if int(parts[id_col]) in selected_ids:
                    kept_lines.append(atom_line)

            if not keep_frame:
                continue

            dst.write("ITEM: TIMESTEP\n")
            dst.write(f"{timestep}\n")
            dst.write(number_header)
            dst.write(f"{len(kept_lines)}\n")
            dst.write(box_header)
            dst.writelines(bounds)
            dst.write(atom_header)
            dst.writelines(kept_lines)
            written += 1
            pbar.update(1)
            if max_frames and written >= max_frames:
                break
    return written


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir")
    parser.add_argument("--input", default="dump.lammpstrj")
    parser.add_argument("--output", default="dump_surf.lammpstrj")
    parser.add_argument("--data", default="packed_polymer_and_solution.data")
    parser.add_argument("--start-timestep", type=int)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    selected_ids = read_data_atom_ids(run_dir / args.data)
    frames = copy_filtered_dump(
        run_dir / args.input,
        run_dir / args.output,
        selected_ids,
        start_timestep=args.start_timestep,
        frame_stride=args.frame_stride,
        max_frames=args.max_frames,
        no_progress=args.no_progress,
    )
    print(f"selected_atoms={len(selected_ids)}")
    print(f"frames_written={frames}")
    print(f"output={run_dir / args.output}")


if __name__ == "__main__":
    main()
