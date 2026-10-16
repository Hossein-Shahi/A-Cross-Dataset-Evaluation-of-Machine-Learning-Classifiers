"""Step 3: turn the raw CSV results into the tables and figures of the paper.

Reads the CSV files written by run_experiments.py (and the data summary written
by prepare_data.py) and writes:
  results/tables/*.md and *.csv   tables (mean of the runs, standard deviation in parentheses)
  results/tables/facts.json       every other number quoted in the paper text
                                  (the "adapt" and "adapt_dos" parts cover Section 6.5)
  figures/fig1_f1_same_vs_cross.png
  figures/fig2_cross_dataset_per_class.png

Usage:
  python src/make_report.py --results results --figures figures --data prepared
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import FULL_NAMES  # noqa: E402

MODELS = ["LR", "DT", "ET", "XGB", "MLP"]
PCT = ["Accuracy", "Precision", "DR", "FAR", "F1", "MCC"]  # shown in percent
GREY, BLUE = "#7a7a7a", "#2563a6"

# Published Extra Trees results (binary, NetFlow v2 datasets) of Sarhan et al. (2022),
# open-access version arXiv:2101.11315, results table for the v2 datasets.
SARHAN = {"UNSW": {"Accuracy": 99.73, "AUC": 0.9845, "F1": 0.97, "DR": 97.07, "FAR": 0.16},
          "CSE": {"Accuracy": 99.35, "AUC": 0.9829, "F1": 0.97, "DR": 96.89, "FAR": 0.31}}


def fmt(mean, sd, pct):
    if pct:
        return f"{100 * mean:.1f} ({100 * sd:.1f})"
    return f"{mean:.3f} ({sd:.3f})"


def summary(df, keys, metrics):
    g = df.groupby(keys, sort=False)[metrics]
    return g.mean(), g.std(ddof=1), g.size()


def to_markdown(df):
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(v) for v in r.values) + " |")
    return "\n".join(lines) + "\n"


def save(df, out, name):
    df.to_csv(out / f"{name}.csv", index=False)
    (out / f"{name}.md").write_text(to_markdown(df))
    print(f"wrote {name}")


def replication_table(res, out):
    r = pd.read_csv(res / "repl_metrics.csv")
    mean, sd, n = summary(r, ["dataset"], ["Accuracy", "AUC", "F1", "DR", "FAR"])
    rows = []
    for ds in ["UNSW", "CSE"]:
        s = SARHAN[ds]
        rows.append({"Dataset": FULL_NAMES[ds], "Source": "Sarhan et al. (2022), full data",
                     "Accuracy (%)": f"{s['Accuracy']:.2f}", "AUC": f"{s['AUC']:.4f}", "F1": f"{s['F1']:.2f}",
                     "DR (%)": f"{s['DR']:.2f}", "FAR (%)": f"{s['FAR']:.2f}"})
        m, d = mean.loc[ds], sd.loc[ds]
        rows.append({"Dataset": FULL_NAMES[ds], "Source": f"This study, 1,000,000-flow sample, {n.loc[ds]} runs",
                     "Accuracy (%)": f"{100 * m.Accuracy:.2f} ({100 * d.Accuracy:.2f})",
                     "AUC": f"{m.AUC:.4f} ({d.AUC:.4f})", "F1": f"{m.F1:.2f} ({d.F1:.2f})",
                     "DR (%)": f"{100 * m.DR:.2f} ({100 * d.DR:.2f})",
                     "FAR (%)": f"{100 * m.FAR:.2f} ({100 * d.FAR:.2f})"})
    save(pd.DataFrame(rows), out, "table_replication")


def main_table(res, out):
    m = pd.read_csv(res / "main_metrics.csv")
    metrics = ["F1", "DR", "FAR", "MCC", "AUC"]
    mean, sd, n = summary(m, ["test", "train", "model"], metrics)
    rows = []
    for test in ["UNSW", "CSE"]:
        for train in [test, "CSE" if test == "UNSW" else "UNSW"]:
            for model in MODELS:
                key = (test, train, model)
                row = {"Tested on": test, "Trained on": train,
                       "Setting": "same dataset" if train == test else "cross-dataset", "Model": model}
                for k in metrics:
                    row[k + (" (%)" if k in PCT else "")] = fmt(mean.loc[key, k], sd.loc[key, k], k in PCT)
                rows.append(row)
    save(pd.DataFrame(rows), out, "table_main")

    # averages over the five models, used in the text
    avg = m.groupby(["train", "test", "seed"])[metrics].mean().groupby(["train", "test"]).agg(["mean", "std"])
    avg.columns = [f"{a}_{b}" for a, b in avg.columns]
    avg = avg.reset_index()
    avg.to_csv(out / "model_average.csv", index=False)
    print("wrote model_average")
    return mean, sd


def per_class(res, out):
    c = pd.read_csv(res / "main_per_class.csv")
    c = c[c["level"] == "class"]
    mean = c.groupby(["train", "test", "class", "model"])["flagged_share"].mean().unstack("model")[MODELS]
    sd = c.groupby(["train", "test", "class", "model"])["flagged_share"].std(ddof=1).unstack("model")[MODELS]
    n = c.groupby(["train", "test", "class"])["n"].first()
    rows = []
    for (train, test, cls), r in mean.iterrows():
        row = {"Trained on": train, "Tested on": test, "Class": cls, "Test flows": int(n.loc[(train, test, cls)])}
        for model in MODELS:
            row[model] = f"{100 * r[model]:.1f} ({100 * sd.loc[(train, test, cls), model]:.1f})"
        rows.append(row)
    table = pd.DataFrame(rows)
    save(table, out, "table_per_class_all")

    # detailed CSE attack names (DoS sub-types), cross-dataset only
    d = pd.read_csv(res / "main_per_class.csv")
    d = d[(d["level"] == "attack name") & (d["train"] != d["test"])]
    dm = d.groupby(["class", "model"])["flagged_share"].mean().unstack("model")[MODELS]
    dn = d.groupby("class")["n"].first()
    dm.insert(0, "Test flows", dn)
    dm = dm.reset_index()
    for model in MODELS:
        dm[model] = (100 * dm[model]).round(1)
    save(dm, out, "table_per_attack_name_cse")
    return mean, n


def dos_table(res, out):
    m = pd.read_csv(res / "dos_metrics.csv")
    metrics = ["F1", "DR", "FAR", "MCC", "AUC"]
    mean, sd, n = summary(m, ["train", "test", "model"], metrics)
    mean, sd = mean.sort_index(), sd.sort_index()
    rows = []
    for test in ["UNSW", "CSE"]:
        for train in [test, "CSE" if test == "UNSW" else "UNSW"]:
            for model in MODELS:
                key = (train, test, model)
                row = {"Tested on": test, "Trained on": train,
                       "Setting": "same dataset" if train == test else "cross-dataset", "Model": model}
                for k in metrics:
                    row[k + (" (%)" if k in PCT else "")] = fmt(mean.loc[key, k], sd.loc[key, k], k in PCT)
                rows.append(row)
    save(pd.DataFrame(rows), out, "table_dos_only")
    sizes = m.groupby(["train", "test"])[["n_train", "n_test"]].first().reset_index()
    pos = m.assign(pos=m.TP + m.FN).groupby(["train", "test"])["pos"].first().reset_index()
    sizes.merge(pos, on=["train", "test"]).to_csv(out / "dos_sizes.csv", index=False)
    print("wrote dos_sizes")

    # Table 4 of the paper: range over the five models of the mean of the runs
    rng = []
    for train, test in [("UNSW", "UNSW"), ("CSE", "UNSW"), ("CSE", "CSE"), ("UNSW", "CSE")]:
        g = mean.loc[(train, test)]
        row = {"Trained on": train, "Tested on": test}
        for k in ["F1", "DR", "FAR"]:
            lo, hi = f"{100 * g[k].min():.1f}", f"{100 * g[k].max():.1f}"
            row[k + " (%)"] = lo if lo == hi else f"{lo} to {hi}"
        lo, hi = f"{g['AUC'].min():.3f}", f"{g['AUC'].max():.3f}"
        row["AUC"] = lo if lo == hi else f"{lo} to {hi}"
        rng.append(row)
    save(pd.DataFrame(rng), out, "table_dos_only_range")


def shift_tables(res, out):
    dc = pd.read_csv(res / "dataset_classifier.csv")
    dc["accuracy (%)"] = (100 * dc["accuracy"]).round(2)
    dc["AUC"] = dc["AUC"].round(4)
    save(dc[["subset", "n_test", "accuracy (%)", "AUC"]], out, "table_dataset_classifier")
    fm = pd.read_csv(res / "feature_medians.csv")
    fm["gain_share"] = (100 * fm["gain_share"]).round(1)
    fm = fm.rename(columns={"train": "XGB trained on", "gain_share": "share of total gain (%)"})
    save(fm, out, "table_top_features")


ADAPT_SETTINGS = [("none", "default", "No fix"), ("none", "recalibrated", "Recalibrated threshold"),
                  ("CORAL", "default", "CORAL"), ("CORAL", "recalibrated", "CORAL + recalibrated")]
DIRECTIONS = [("CSE", "UNSW"), ("UNSW", "CSE")]


def adapt_tables(res, out):
    """Section 6.5: CORAL and threshold recalibration (main test and DoS-only test)."""
    metrics = ["F1", "DR", "FAR", "MCC", "AUC"]
    for name in ["adapt", "adapt_dos"]:
        path = res / f"{name}_metrics.csv"
        if not path.exists():
            continue
        a = pd.read_csv(path)
        mean, sd, n = summary(a, ["train", "test", "adaptation", "threshold_type", "model"], metrics)
        rows = []
        for train, test in DIRECTIONS:
            for adaptation in ["none", "CORAL"]:
                for kind in ["default", "recalibrated", "oracle"]:
                    for model in MODELS:
                        key = (train, test, adaptation, kind, model)
                        row = {"Trained on": train, "Tested on": test, "Adaptation": adaptation,
                               "Threshold": kind, "Model": model}
                        for k in metrics:
                            row[k + (" (%)" if k in PCT else "")] = fmt(mean.loc[key, k], sd.loc[key, k], k in PCT)
                        rows.append(row)
        save(pd.DataFrame(rows), out, f"table_{name}_all")

        # compact table for the paper: mean F1 and MCC (in %) of the five runs
        rows = []
        for train, test in DIRECTIONS:
            for model in MODELS:
                row = {"Trained on": train, "Tested on": test, "Model": model}
                for adaptation, kind, label in ADAPT_SETTINGS:
                    r = mean.loc[(train, test, adaptation, kind, model)]
                    row[f"{label}: F1 / MCC (%)"] = f"{100 * r.F1:.1f} / {100 * r.MCC:.1f}"
                rows.append(row)
        save(pd.DataFrame(rows), out, f"table_{name}")


def adapt_facts(res):
    """Numbers for the text of Section 6.5."""
    out = {}
    for name in ["adapt", "adapt_dos"]:
        path = res / f"{name}_metrics.csv"
        if not path.exists():
            continue
        a = pd.read_csv(path)
        a["flagged"] = (a.TP + a.FP) / a.n_test  # share of test flows flagged as attacks
        f = {}
        for (train, test), g in a.groupby(["train", "test"]):
            d = f"{train}->{test}"
            first = g[g.seed == g.seed.min()].iloc[0]
            prevalence = (first.TP + first.FN) / first.n_test
            f[d] = {"test_attack_share_%": round(100 * prevalence, 2),
                    # F1 of a detector that flags every flow (its MCC is 0)
                    "flag_all_F1_%": round(100 * 2 * prevalence / (1 + prevalence), 1),
                    "n_calibration": sorted(g.n_calibration.unique().tolist()),
                    "calibration_attacks": sorted(g.calibration_attacks.unique().tolist())}
            m = g.groupby(["adaptation", "threshold_type", "model"])[
                ["F1", "MCC", "DR", "FAR", "AUC", "flagged", "threshold"]].mean()
            for (adaptation, kind, model), r in m.iterrows():
                f[d].setdefault(adaptation, {}).setdefault(kind, {})[model] = {
                    "F1": round(100 * r.F1, 1), "MCC": round(100 * r.MCC, 1), "DR": round(100 * r.DR, 1),
                    "FAR": round(100 * r.FAR, 1), "AUC": round(r.AUC, 3),
                    "flagged_%": round(100 * r.flagged, 1), "threshold": round(r.threshold, 4)}
            # how close the recalibrated threshold came to the oracle threshold (F1, mean of runs)
            gap = (m.xs("oracle", level="threshold_type")["F1"] - m.xs("recalibrated", level="threshold_type")["F1"])
            f[d]["largest_oracle_minus_recalibrated_F1_points"] = round(100 * float(gap.max()), 1)
        out[name] = f
    return out


def facts(res, out, data_dir):
    """Numbers quoted in the paper text, computed from the result files."""
    m = pd.read_csv(res / "main_metrics.csv")
    c = pd.read_csv(res / "main_per_class.csv")
    f = {}
    if data_dir is not None and (Path(data_dir) / "data_summary.json").exists():
        s = json.loads((Path(data_dir) / "data_summary.json").read_text())
        for ds in ["UNSW", "CSE"]:
            file_n, uniq_n = s[ds]["file_counts"]["flows"], s[ds]["unique_counts"]["flows"]
            f[f"{ds}_flows_in_file"] = file_n
            f[f"{ds}_unique_flows"] = uniq_n
            f[f"{ds}_copies_%"] = round(100 * (1 - uniq_n / file_n), 1)
            f[f"{ds}_attack_share_file_%"] = s[ds]["file_counts"]["attack_share_%"]
            f[f"{ds}_attack_share_used_%"] = s[ds]["main_counts"]["attack_share_%"]
        # flows whose byte rate is above one terabyte per second (physically impossible here)
        for ds in ["UNSW", "CSE"]:
            d = pd.read_parquet(Path(data_dir) / f"main_{ds}.parquet",
                                columns=["SRC_TO_DST_SECOND_BYTES", "DST_TO_SRC_SECOND_BYTES"])
            f[f"{ds}_flows_above_1TB_per_s"] = int((d.max(axis=1) > 1e12).sum())
            f[f"{ds}_max_byte_rate"] = float(d.max().max())
    first = m[m.seed == m.seed.min()]
    f["n_train_main"] = sorted(first["n_train"].unique().tolist())
    f["n_test_main"] = sorted(first["n_test"].unique().tolist())
    cse_test = first[first.test == "CSE"].iloc[0]
    f["CSE_test_benign_%"] = round(100 * (cse_test.TN + cse_test.FP) / cse_test.n_test, 1)

    mean = m.groupby(["train", "test", "model"])[["F1", "DR", "FAR", "MCC", "AUC"]].mean()
    for (train, test), g in mean.groupby(level=[0, 1]):
        g = g.droplevel([0, 1])
        f[f"{train}->{test}"] = {k: {"min": round(100 * g[k].min(), 1) if k != "AUC" else round(g[k].min(), 3),
                                     "max": round(100 * g[k].max(), 1) if k != "AUC" else round(g[k].max(), 3)}
                                 for k in g.columns}
        f[f"{train}->{test}"]["negative_MCC_models"] = int((g["MCC"] < 0).sum())
    avg = m.groupby(["train", "test", "seed"])[["F1", "MCC"]].mean().groupby(["train", "test"]).mean()
    f["average_over_models_%"] = {f"{a}->{b}": {"F1": round(100 * r.F1, 1), "MCC": round(100 * r.MCC, 1)}
                                  for (a, b), r in avg.iterrows()}

    cls = c[c.level == "class"].groupby(["train", "test", "class", "model"])["flagged_share"].mean().unstack("model")
    f["DoS_detected_%"] = {f"{a}->{b}": {k: round(100 * v, 1) for k, v in cls.loc[(a, b, "DoS")].items()}
                           for a, b in [("UNSW", "UNSW"), ("CSE", "CSE"), ("CSE", "UNSW"), ("UNSW", "CSE")]}
    dos, ben = cls.loc[("UNSW", "CSE", "DoS")], cls.loc[("UNSW", "CSE", "Benign")]
    f["UNSW->CSE_models_flagging_DoS_no_more_than_benign"] = int((dos <= ben).sum())
    f["Infiltration_detected_same_dataset_%"] = {k: round(100 * v, 1) for k, v in cls.loc[("CSE", "CSE", "Infiltration")].items()}
    s = c[(c.level == "class") & (c.train == "CSE") & (c.test == "CSE") & (c["class"] != "Benign")].copy()
    s["missed"] = s["n"] * (1 - s["flagged_share"])
    share = (s[s["class"] == "Infiltration"].set_index(["seed", "model"])["missed"]
             / s.groupby(["seed", "model"])["missed"].sum())
    f["Infiltration_share_of_missed_CSE_attacks_%"] = {k: round(100 * v, 1) for k, v in share.groupby("model").mean().items()}
    d = c[(c.level == "attack name") & (c.train == "UNSW") & (c.test == "CSE")]
    loic = d[d["class"] == "DDoS attacks-LOIC-HTTP"]
    f["LOIC-HTTP_UNSW->CSE_flagged_%"] = {k: round(100 * v, 1) for k, v in loic.groupby("model")["flagged_share"].mean().items()}
    per_run = loic.groupby("seed")["n"].first()
    f["LOIC-HTTP_test_flows_per_run"] = {"mean": round(float(per_run.mean()), 1),
                                         "min": int(per_run.min()), "max": int(per_run.max())}
    if (res / "dos_metrics.csv").exists():
        dm = pd.read_csv(res / "dos_metrics.csv").groupby(["train", "test", "model"])[["F1", "AUC"]].mean()
        f["DoS_only_F1_%"] = {f"{a}->{b}": {k: round(100 * v, 1) for k, v in g.droplevel([0, 1])["F1"].items()}
                              for (a, b), g in dm.groupby(level=[0, 1])}
        f["DoS_only_AUC"] = {f"{a}->{b}": {k: round(v, 3) for k, v in g.droplevel([0, 1])["AUC"].items()}
                             for (a, b), g in dm.groupby(level=[0, 1])}
    if (res / "dataset_classifier.csv").exists():
        dc = pd.read_csv(res / "dataset_classifier.csv")
        f["dataset_classifier_accuracy_%"] = {r.subset: round(100 * r.accuracy, 2) for r in dc.itertuples()}
    f.update(adapt_facts(res))
    (out / "facts.json").write_text(json.dumps(f, indent=1))
    print("wrote facts.json")


def fig_f1(res, figs):
    m = pd.read_csv(res / "main_metrics.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    x = np.arange(len(MODELS))
    w = 0.38
    for ax, test in zip(axes, ["UNSW", "CSE"]):
        other = "CSE" if test == "UNSW" else "UNSW"
        for i, (train, colour, label) in enumerate([(test, GREY, "Same dataset (trained on the test dataset)"),
                                                    (other, BLUE, "Cross-dataset (trained on the other dataset)")]):
            sub = m[(m.test == test) & (m.train == train)]
            g = sub.groupby("model")["F1"]
            mu = g.mean().reindex(MODELS).to_numpy()
            sdv = g.std(ddof=1).reindex(MODELS).to_numpy()
            pos = x + (i - 0.5) * w
            ax.bar(pos, mu, w, color=colour, label=label, yerr=sdv, capsize=2,
                   error_kw={"elinewidth": 0.8, "ecolor": "#333333"})
            for p, v, s in zip(pos, mu, sdv):
                ax.text(p, v + s + 0.02, f"{v:.2f}", ha="center", va="bottom", fontsize=8, color="#222222")
        ax.set_title(f"Tested on {FULL_NAMES[test]}", fontsize=10)
        ax.set_xticks(x, MODELS)
        ax.set_ylim(0, 1.05)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("F1 score (mean of 5 runs)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, fontsize=9, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    path = figs / "fig1_f1_same_vs_cross.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print("wrote", path)


def fig_per_class(res, figs, min_flows=30):
    c = pd.read_csv(res / "main_per_class.csv")
    c = c[(c["level"] == "class") & (c["train"] != c["test"])]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.9), gridspec_kw={"wspace": 0.75})
    for ax, (train, test) in zip(axes, [("CSE", "UNSW"), ("UNSW", "CSE")]):
        sub = c[(c.train == train) & (c.test == test)]
        mean = sub.groupby(["class", "model"])["flagged_share"].mean().unstack("model")[MODELS]
        n = sub.groupby("class")["n"].first()
        # classes with very few test flows give unstable percentages; they stay in the table
        keep = n[(n >= min_flows) & (n.index != "Benign")].sort_values(ascending=False).index
        order = ["Benign"] + list(keep)
        mean, n = mean.loc[order], n.loc[order]
        im = ax.imshow(100 * mean.to_numpy(), cmap="Blues", vmin=0, vmax=100, aspect="auto")
        labels = [("Benign (FAR)" if k == "Benign" else k) + f"  n={n.loc[k]:,}" for k in order]
        ax.set_yticks(range(len(order)), labels, fontsize=8)
        for tick in ax.get_yticklabels():
            if tick.get_text().startswith("DoS "):
                tick.set_fontweight("bold")
        ax.set_xticks(range(len(MODELS)), MODELS, fontsize=8)
        ax.axhline(0.5, color="#333333", lw=0.8)
        for i in range(len(order)):
            for j in range(len(MODELS)):
                v = 100 * mean.iat[i, j]
                ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=7,
                        color="white" if v > 55 else "#222222")
        ax.set_title(f"Trained on {train}, tested on {test}", fontsize=10)
        ax.tick_params(length=0)
    cbar = fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02)
    cbar.set_label("flows flagged as attack (%)", fontsize=8)
    path = figs / "fig2_cross_dataset_per_class.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default="results")
    ap.add_argument("--figures", default="figures")
    ap.add_argument("--data", default="prepared", help="folder written by prepare_data.py (for Table 1 numbers)")
    args = ap.parse_args()
    res, figs = Path(args.results), Path(args.figures)
    out = res / "tables"
    out.mkdir(parents=True, exist_ok=True)
    figs.mkdir(parents=True, exist_ok=True)

    if (res / "repl_metrics.csv").exists():
        replication_table(res, out)
    main_table(res, out)
    per_class(res, out)
    if (res / "dos_metrics.csv").exists():
        dos_table(res, out)
    if (res / "dataset_classifier.csv").exists():
        shift_tables(res, out)
    adapt_tables(res, out)
    facts(res, out, args.data)
    fig_f1(res, figs)
    fig_per_class(res, figs)


if __name__ == "__main__":
    main()
