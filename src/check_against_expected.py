"""Optional step 4: compare your results with the results reported in the paper.

Every CSV file in expected_results/ (and its tables/ folder) is compared with the
file of the same name in your results folder. Numbers must agree within a small
tolerance; run times (columns ending in "_seconds") are skipped because they
depend on the computer.

Usage:
  python src/check_against_expected.py --results results --expected expected_results
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def compare(mine, expected, tol):
    a, b = pd.read_csv(mine), pd.read_csv(expected)
    if a.shape != b.shape or list(a.columns) != list(b.columns):
        return f"different shape or columns: {a.shape} vs {b.shape}"
    worst = 0.0
    for col in a.columns:
        if col.endswith("_seconds"):
            continue
        x, y = a[col], b[col]
        if pd.api.types.is_numeric_dtype(x) and pd.api.types.is_numeric_dtype(y):
            diff = np.nanmax(np.abs(x.to_numpy(float) - y.to_numpy(float))) if len(x) else 0.0
            if not np.array_equal(np.isnan(x.to_numpy(float)), np.isnan(y.to_numpy(float))):
                return f"column {col}: missing values differ"
            worst = max(worst, float(diff))
        elif not x.astype(str).equals(y.astype(str)):
            return f"column {col}: text differs"
    return None if worst <= tol else f"largest difference {worst:.2e} is above {tol:.0e}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default="results")
    ap.add_argument("--expected", default="expected_results")
    ap.add_argument("--tol", type=float, default=1e-6, help="largest allowed absolute difference")
    args = ap.parse_args()
    res, exp = Path(args.results), Path(args.expected)
    files = sorted(exp.rglob("*.csv"))
    if not files:
        sys.exit(f"no CSV files found in {exp}")
    failed = 0
    for f in files:
        rel = f.relative_to(exp)
        mine = res / rel
        if not mine.exists():
            print(f"MISSING  {rel}")
            failed += 1
            continue
        problem = compare(mine, f, args.tol)
        print(f"{'OK      ' if problem is None else 'DIFFERS '} {rel}" + ("" if problem is None else f"  ({problem})"))
        failed += problem is not None
    print(f"\n{len(files) - failed} of {len(files)} files match the paper's results.")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
