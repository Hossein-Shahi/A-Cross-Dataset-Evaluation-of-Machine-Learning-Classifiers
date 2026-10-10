"""Step 2: run the experiments and save the raw results as CSV files.

Experiments (choose with --exp, default: all four)
  repl   Extra Trees inside each dataset with the original protocol of
         Sarhan et al. (2022). Used to check the pipeline against their numbers.
  main   Five classifiers. Each one is trained on one dataset and tested on the
         test split of the same dataset and on the test split of the other one.
  dos    The same test, but with benign and DoS flows only.
  shift  Error analysis (first seed only): whether a model can tell the two
         datasets apart, which features the XGB models rely on, and how these
         features differ between the datasets.

Every experiment is repeated with the seeds given by --seeds. Each seed gives a
new 70/30 train/test split and a new random start for the models.

Usage:
  python src/run_experiments.py --data prepared --out results
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, matthews_corrcoef, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ATTACK, DATASETS, GROUP, LABEL, attack_group, feature_columns  # noqa: E402

MODEL_NAMES = ["LR", "DT", "ET", "XGB", "MLP"]


def make_model(name, seed):
    """The five classifiers. Library defaults, except where noted."""
    if name == "LR":
        return LogisticRegression(max_iter=1000)  # more iterations so that it converges
    if name == "DT":
        return DecisionTreeClassifier(random_state=seed)
    if name == "ET":
        return ExtraTreesClassifier(n_estimators=100, n_jobs=-1, random_state=seed)
    if name == "XGB":
        return XGBClassifier(n_estimators=100, tree_method="hist", n_jobs=-1, random_state=seed)
    if name == "MLP":
        # one hidden layer of 100 neurons; early stopping keeps the run time short
        return MLPClassifier(hidden_layer_sizes=(100,), early_stopping=True, random_state=seed)
    raise ValueError(name)


def pipeline(model):
    """log(1 + x) shrinks very large byte and rate values. Min-max scaling then maps
    each feature to 0..1. Both steps are fitted on the training data only, so the
    test data of the other dataset is never seen before testing. The two steps do
    not change the decisions of tree-based models."""
    return Pipeline([("log", FunctionTransformer(np.log1p)),
                     ("scale", MinMaxScaler()),
                     ("model", model)])


def metrics(y, pred, score):
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    precision = tp / (tp + fp) if tp + fp else 0.0
    dr = tp / (tp + fn) if tp + fn else 0.0  # detection rate (recall)
    f1 = 2 * precision * dr / (precision + dr) if precision + dr else 0.0
    return {"TP": int(tp), "FP": int(fp), "TN": int(tn), "FN": int(fn),
            "Accuracy": (tp + tn) / len(y), "Precision": precision, "DR": dr,
            "FAR": fp / (fp + tn) if fp + tn else 0.0,  # false alarm rate
            "F1": f1, "MCC": matthews_corrcoef(y, pred),
            "AUC": roc_auc_score(y, score) if len(np.unique(y)) == 2 else np.nan}


def load(data_dir, kind, name):
    df = pd.read_parquet(Path(data_dir) / f"{kind}_{name}.parquet")
    df[GROUP] = attack_group(df[ATTACK], name).to_numpy()
    return df


def xy(df, features):
    return df[features].to_numpy(dtype=np.float64), df[LABEL].to_numpy(dtype=np.int64)


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


# ---------------------------------------------------------------- replication
def run_repl(data_dir, seeds):
    rows = []
    for name in DATASETS:
        df = load(data_dir, "repl", name)
        # Original protocol: TTL features dropped for UNSW-NB15 only.
        features = feature_columns(df, drop_ttl=(name == "UNSW"))
        for seed in seeds:
            train, test = train_test_split(df, test_size=0.3, stratify=df[GROUP], random_state=seed)
            X_tr, y_tr = xy(train, features)
            X_te, y_te = xy(test, features)
            # Original protocol: min-max scaling only (no log transform), then Extra Trees.
            pipe = Pipeline([("scale", MinMaxScaler()),
                             ("model", ExtraTreesClassifier(n_estimators=100, n_jobs=-1, random_state=seed))])
            t0 = time.perf_counter()
            pipe.fit(X_tr, y_tr)
            fit_s = time.perf_counter() - t0
            score = pipe.predict_proba(X_te)[:, 1]
            pred = pipe.predict(X_te)
            row = {"dataset": name, "seed": seed, "n_features": len(features),
                   "n_train": len(y_tr), "n_test": len(y_te), "fit_seconds": fit_s}
            row.update(metrics(y_te, pred, score))
            rows.append(row)
            log(f"repl {name} seed={seed} F1={row['F1']:.4f} DR={row['DR']:.4f} FAR={row['FAR']:.4f} ({fit_s:.0f}s)")
            del pipe
    return {"repl_metrics": pd.DataFrame(rows)}


# ------------------------------------------------------- main and DoS-only test
def run_cross(data_dir, seeds, dos_only=False):
    data = {n: load(data_dir, "main", n) for n in DATASETS}
    features = feature_columns(data["UNSW"])
    assert features == feature_columns(data["CSE"]), "feature lists differ"
    rows, class_rows = [], []
    tag = "dos" if dos_only else "main"

    for seed in seeds:
        split = {n: train_test_split(d, test_size=0.3, stratify=d[GROUP], random_state=seed)
                 for n, d in data.items()}
        if dos_only:
            split = {n: tuple(part[part[GROUP].isin(["Benign", "DoS"])] for part in pair)
                     for n, pair in split.items()}

        for source in DATASETS:
            X_tr, y_tr = xy(split[source][0], features)
            for model_name in MODEL_NAMES:
                pipe = pipeline(make_model(model_name, seed))
                t0 = time.perf_counter()
                pipe.fit(X_tr, y_tr)
                fit_s = time.perf_counter() - t0

                for target in DATASETS:
                    test = split[target][1]
                    X_te, y_te = xy(test, features)
                    score = pipe.predict_proba(X_te)[:, 1]
                    pred = pipe.predict(X_te)
                    row = {"seed": seed, "model": model_name, "train": source, "test": target,
                           "setting": "same dataset" if source == target else "cross-dataset",
                           "n_train": len(y_tr), "n_test": len(y_te), "fit_seconds": fit_s}
                    row.update(metrics(y_te, pred, score))
                    rows.append(row)

                    if not dos_only:
                        # Share of flows of each class that the model flags as attacks.
                        # For attack classes this is the detection rate of that class;
                        # for the Benign class it is the false alarm rate.
                        flagged = pd.Series(pred, index=test.index)
                        levels = [("class", GROUP)] + ([("attack name", ATTACK)] if target == "CSE" else [])
                        for level, column in levels:
                            for cls, idx in test.groupby(column).groups.items():
                                class_rows.append({"seed": seed, "model": model_name, "train": source,
                                                   "test": target, "level": level, "class": cls,
                                                   "n": len(idx), "flagged_share": float(flagged[idx].mean())})


                last = rows[-2:]
                log(f"{tag} seed={seed} {model_name:3s} train={source:4s} "
                    + " ".join(f"test={r['test']}:F1={r['F1']:.3f}" for r in last) + f" ({fit_s:.0f}s)")
                del pipe

    out = {f"{tag}_metrics": pd.DataFrame(rows)}
    if not dos_only:
        out["main_per_class"] = pd.DataFrame(class_rows)
    return out


# --------------------------------------------------------------- error analysis
def run_shift(data_dir, results_dir, seed=0, top_n=3):
    data = {n: load(data_dir, "main", n) for n in DATASETS}
    features = feature_columns(data["UNSW"])
    out = {}

    # 1. Can a model tell which dataset a flow comes from? If yes, the two
    #    datasets cover very different parts of the feature space.
    rows = []
    for subset in ["all flows", "benign flows only"]:
        parts = []
        for i, n in enumerate(DATASETS):
            d = data[n] if subset == "all flows" else data[n][data[n][LABEL] == 0]
            parts.append(pd.DataFrame(d[features].to_numpy(np.float64), columns=features).assign(_dataset=i))
        both = pd.concat(parts, ignore_index=True)
        train, test = train_test_split(both, test_size=0.3, stratify=both["_dataset"], random_state=seed)
        # Same preprocessing as the main experiments. It also keeps a few impossible
        # byte-rate values in the CSE data (up to about 1e285) finite for XGBoost.
        model = pipeline(XGBClassifier(n_estimators=100, tree_method="hist", n_jobs=-1, random_state=seed))
        model.fit(train[features].to_numpy(np.float64), train["_dataset"].to_numpy())
        score = model.predict_proba(test[features].to_numpy(np.float64))[:, 1]
        pred = (score > 0.5).astype(int)
        rows.append({"subset": subset, "n_test": len(test),
                     "accuracy": float((pred == test["_dataset"]).mean()),
                     "AUC": roc_auc_score(test["_dataset"], score)})
        log(f"shift dataset classifier ({subset}): accuracy={rows[-1]['accuracy']:.4f}")
    out["dataset_classifier"] = pd.DataFrame(rows)

    # 2. Which features the XGB model of each source relies on: the same model as in the
    #    main experiment for this seed (same split, same settings). Importance = share of
    #    the total gain of all splits on that feature.
    rows = []
    for source in DATASETS:
        train, _ = train_test_split(data[source], test_size=0.3, stratify=data[source][GROUP], random_state=seed)
        X_tr, y_tr = xy(train, features)
        pipe = pipeline(make_model("XGB", seed))
        pipe.fit(X_tr, y_tr)
        total_gain = pipe.named_steps["model"].get_booster().get_score(importance_type="total_gain")
        total = sum(total_gain.values())
        for key, value in total_gain.items():
            rows.append({"train": source, "feature": features[int(key[1:])], "gain_share": value / total})
    imp = pd.DataFrame(rows)
    out["feature_importance"] = imp
    log("shift feature importance: " + ", ".join(
        f"{r.train}:{r.feature}={100 * r.gain_share:.1f}%" for r in imp.sort_values("gain_share", ascending=False).groupby("train").head(1).itertuples()))

    # 3. Median value of the most important features, per dataset and label.
    top = (imp.sort_values("gain_share", ascending=False).groupby("train").head(top_n))
    rows = []
    for _, r in top.iterrows():
        row = {"train": r["train"], "feature": r["feature"], "gain_share": r["gain_share"]}
        for n in DATASETS:
            d = data[n]
            row[f"{n} benign median"] = float(d.loc[d[LABEL] == 0, r["feature"]].median())
            row[f"{n} attack median"] = float(d.loc[d[LABEL] == 1, r["feature"]].median())
        rows.append(row)
    out["feature_medians"] = pd.DataFrame(rows)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="prepared", help="folder written by prepare_data.py")
    ap.add_argument("--out", default="results", help="folder for the CSV results")
    ap.add_argument("--exp", nargs="+", default=["repl", "main", "dos", "shift"],
                    choices=["repl", "main", "dos", "shift"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    for exp in args.exp:
        log(f"start {exp}")
        if exp == "repl":
            tables = run_repl(args.data, args.seeds)
        elif exp == "main":
            tables = run_cross(args.data, args.seeds)
        elif exp == "dos":
            tables = run_cross(args.data, args.seeds, dos_only=True)
        else:
            tables = run_shift(args.data, out, seed=args.seeds[0])
        for name, table in tables.items():
            table.to_csv(out / f"{name}.csv", index=False)
            log(f"wrote {out / (name + '.csv')} ({len(table)} rows)")


if __name__ == "__main__":
    main()
