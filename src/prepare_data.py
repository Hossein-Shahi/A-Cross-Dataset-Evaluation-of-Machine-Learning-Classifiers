"""Step 1: prepare the data.

Reads the two cleaned Kaggle parquet files in batches, so it also runs on a
laptop with 8 GB of memory, and writes four smaller files:

  main_UNSW.parquet, main_CSE.parquet
      Data for the cross-dataset experiments. Port and TTL columns are removed,
      then duplicate flows (same features and same label) are removed. All
      unique UNSW flows are kept. A stratified random sample of the same size
      is drawn from the unique CSE flows, so both training sets are equal.

  repl_UNSW.parquet, repl_CSE.parquet
      Data for the baseline replication. A stratified random sample of
      1,000,000 flows from each file, without the extra duplicate removal,
      as in the original protocol of Sarhan et al. (2022).

A summary of all counts is written to data_summary.json.

Usage:
  python src/prepare_data.py --unsw data/NF-UNSW-NB15-V2.parquet \
                             --cse data/NF-CSE-CIC-IDS2018-V2.parquet --out prepared
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ATTACK, ID_COLUMNS, LABEL, TTL_COLUMNS  # noqa: E402

BATCH_SIZE = 1_000_000


def scan(path, features):
    """First pass over the file: one hash per flow, plus its label and attack name."""
    pf = pq.ParquetFile(path)
    names, row_hash, feat_hash, labels, codes = {}, [], [], [], []
    for batch in pf.iter_batches(batch_size=BATCH_SIZE, columns=features + [LABEL, ATTACK]):
        d = batch.to_pandas()
        feat_hash.append(pd.util.hash_pandas_object(d[features], index=False).to_numpy())
        row_hash.append(pd.util.hash_pandas_object(d[features + [LABEL]], index=False).to_numpy())
        labels.append(d[LABEL].to_numpy(dtype=np.int8))
        for name in d[ATTACK].unique():
            names.setdefault(name, len(names))
        codes.append(d[ATTACK].map(names).to_numpy(dtype=np.int16))
    code_to_name = {v: k for k, v in names.items()}
    return (np.concatenate(row_hash), np.concatenate(feat_hash),
            np.concatenate(labels), np.concatenate(codes), code_to_name)


def stratified_pick(index, strata, n, seed):
    """Pick n row numbers at random, keeping the share of every attack name."""
    if n is None or n >= len(index):
        return np.sort(index)
    picked, _ = train_test_split(index, train_size=n, stratify=strata, random_state=seed)
    return np.sort(picked)


def collect(path, columns, masks):
    """Second pass over the file: keep only the selected rows (one mask per output)."""
    pf = pq.ParquetFile(path)
    parts = {key: [] for key in masks}
    start = 0
    for batch in pf.iter_batches(batch_size=BATCH_SIZE, columns=columns):
        d = batch.to_pandas()
        n = len(d)
        for key, mask in masks.items():
            selected = mask[start:start + n]
            if selected.any():
                parts[key].append(d[selected])
        start += n
    return {key: pd.concat(p, ignore_index=True) for key, p in parts.items()}


def counts(codes, labels, code_to_name):
    per_class = pd.Series(codes).map(code_to_name).value_counts()
    return {"flows": int(len(labels)), "attacks": int(labels.sum()),
            "attack_share_%": round(100 * float(labels.mean()), 2),
            "per_class": {k: int(v) for k, v in per_class.items()}}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--unsw", required=True, help="path to NF-UNSW-NB15-V2.parquet")
    ap.add_argument("--cse", required=True, help="path to NF-CSE-CIC-IDS2018-V2.parquet")
    ap.add_argument("--out", default="prepared", help="output folder")
    ap.add_argument("--repl-size", type=int, default=1_000_000, help="flows per dataset for the replication")
    ap.add_argument("--seed", type=int, default=42, help="random seed for sampling")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    summary, n_main = {}, None

    for name, path in [("UNSW", args.unsw), ("CSE", args.cse)]:
        print(f"[{name}] scanning {path}", flush=True)
        all_cols = pq.ParquetFile(path).schema_arrow.names
        features = [c for c in all_cols if c not in ID_COLUMNS + TTL_COLUMNS + [LABEL, ATTACK]]
        row_hash, feat_hash, labels, codes, code_to_name = scan(path, features)
        n_rows = len(labels)

        # A flow is a duplicate if an earlier flow has the same 37 features and the same label.
        first = ~pd.Series(row_hash).duplicated().to_numpy()
        unique_idx = np.flatnonzero(first)
        # Feature vectors that appear with both labels (cannot be separated by any model).
        both = pd.DataFrame({"h": feat_hash[first], "y": labels[first]}).groupby("h")["y"].nunique()
        conflicts = int((both > 1).sum())

        # Main data: all unique UNSW flows; the same number of unique CSE flows.
        main_idx = stratified_pick(unique_idx, codes[unique_idx], n_main, args.seed)
        if name == "UNSW":
            n_main = len(main_idx)
        # Replication data: stratified sample of the cleaned file, duplicates kept.
        repl_idx = stratified_pick(np.arange(n_rows), codes, args.repl_size, args.seed)

        masks = {"main": np.zeros(n_rows, bool), "repl": np.zeros(n_rows, bool)}
        masks["main"][main_idx] = True
        masks["repl"][repl_idx] = True
        keep = [c for c in all_cols if c not in ID_COLUMNS]
        parts = collect(path, keep, masks)
        parts["main"] = parts["main"].drop(columns=TTL_COLUMNS)
        for key, d in parts.items():
            d.to_parquet(out / f"{key}_{name}.parquet", index=False)
            print(f"[{name}] wrote {key}_{name}.parquet: {len(d):,} flows, {d.shape[1] - 2} features", flush=True)

        summary[name] = {
            "file": Path(path).name,
            "file_counts": counts(codes, labels, code_to_name),
            "unique_counts": counts(codes[first], labels[first], code_to_name),
            "conflicting_feature_vectors": conflicts,
            "main_counts": counts(codes[main_idx], labels[main_idx], code_to_name),
            "main_features": len(features),
            "repl_counts": counts(codes[repl_idx], labels[repl_idx], code_to_name),
        }
        del row_hash, feat_hash, labels, codes, parts

    with open(out / "data_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("done", flush=True)


if __name__ == "__main__":
    main()
