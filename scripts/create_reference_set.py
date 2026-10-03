"""Build a stable reference dataset (~500 rows) from the M1 holdout.

Usage:
    python scripts/create_reference_set.py
    python scripts/create_reference_set.py --input data/lending_club_holdout.csv --output data/reference_set.csv --n-samples 500 --seed 42

The goal is to create a fixed evaluation set that remains stable across releases,
so the same model can be assessed against a constant baseline.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "data" / "lending_club_holdout.csv"
DEFAULT_OUTPUT = ROOT / "data" / "reference_set.csv"


def allocate_counts(total_rows: int, target_counts: dict[str, int]) -> dict[str, int]:
    """Allocate sample counts by class, keeping the sum exactly equal to total_rows."""
    allocations = {
        label: int((total_rows * count) / sum(target_counts.values()))
        for label, count in target_counts.items()
    }

    # Ensure every class gets at least one row when possible.
    if total_rows >= len(target_counts):
        for label in target_counts:
            allocations[label] = max(1, allocations[label])

    while sum(allocations.values()) > total_rows:
        for label in sorted(target_counts, key=lambda x: target_counts[x], reverse=True):
            if allocations[label] > 1:
                allocations[label] -= 1
            if sum(allocations.values()) == total_rows:
                break

    while sum(allocations.values()) < total_rows:
        for label in sorted(target_counts, key=lambda x: target_counts[x], reverse=True):
            if allocations[label] < target_counts[label]:
                allocations[label] += 1
            if sum(allocations.values()) == total_rows:
                break

    return allocations


def build_reference_set(
    input_path: Path,
    output_path: Path,
    n_samples: int = 500,
    seed: int = 42,
) -> int:
    if not input_path.exists():
        raise FileNotFoundError(f"Holdout not found: {input_path}")

    df = pd.read_csv(input_path)
    if "loan_status" not in df.columns:
        raise ValueError("Expected a 'loan_status' column in the holdout dataset.")

    if len(df) < n_samples:
        raise ValueError(
            f"The holdout has {len(df)} rows, which is smaller than the requested {n_samples}."
        )

    target_counts = df["loan_status"].value_counts().to_dict()
    allocations = allocate_counts(n_samples, target_counts)

    sampled_frames = []
    for label, count in allocations.items():
        class_df = df[df["loan_status"] == label]
        n_to_take = min(count, len(class_df))
        if n_to_take <= 0:
            continue
        sampled_frames.append(class_df.sample(n=n_to_take, random_state=seed))

    if not sampled_frames:
        raise ValueError("No rows could be sampled from the holdout dataset.")

    ref_df = pd.concat(sampled_frames, ignore_index=True)
    if len(ref_df) > n_samples:
        ref_df = ref_df.sample(n=n_samples, random_state=seed).reset_index(drop=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    ref_df.to_csv(output_path, index=False)

    print(f"Saved reference set to {output_path}")
    print(f"Rows: {len(ref_df)}")
    print("Target distribution:")
    print(ref_df["loan_status"].value_counts().to_string())
    return len(ref_df)


def main() -> int:
    parser = argparse.ArgumentParser(description="Create a stable evaluation reference set from the M1 holdout.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Path to the holdout CSV.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Where to save the reference set CSV.")
    parser.add_argument("--n-samples", type=int, default=500, help="Target number of rows in the reference set.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed used for reproducibility.")
    args = parser.parse_args()

    build_reference_set(
        input_path=args.input,
        output_path=args.output,
        n_samples=args.n_samples,
        seed=args.seed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
