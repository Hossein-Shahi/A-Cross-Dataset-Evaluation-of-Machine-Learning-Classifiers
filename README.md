# Cross-Dataset Evaluation of Machine Learning Intrusion Detectors
A Cross-Dataset Evaluation of Machine Learning Classifiers

# Network Intrusion / DDoS Detection Using NetFlow Benchmark Datasets
Hossein Shahi* 1, Junaid Shuja 2

Department of Computer Science, Southeast Missouri State University, Cape Girardeau, Missouri

Department of Computer Science, Southeast Missouri State University, Cape Girardeau, Missouri

DOI: https://doi.org/10.5281/zenodo.23275689

This code is for the paper *Network Intrusion / DDoS Detection Using NetFlow Benchmark Datasets:
A Cross-Dataset Evaluation of Machine Learning Classifiers*.

The study trains five classifiers (logistic regression, decision tree, extra trees, XGBoost,
and a multilayer perceptron) on one NetFlow dataset and tests them on the same dataset and on the other one,
in both directions:

- **NF-UNSW-NB15-v2** (called UNSW in the code)
- **NF-CSE-CIC-IDS2018-v2** (called CSE in the code)

It also replicates the Extra Trees baseline of Sarhan et al. (2022), tests DoS attacks on their own,
and runs a short error analysis.

## Contents

| Path | What it does |
|---|---|
| `src/common.py` | Shared settings: dropped columns, CSE attack-class groups |
| `src/prepare_data.py` | Step 1. Reads the two Kaggle files in batches, removes ports and TTL features, removes duplicate flows, samples the data, writes `prepared/` |
| `src/run_experiments.py` | Step 2. Runs the replication, the main cross-dataset test, the DoS-only test and the error analysis, writes `results/*.csv` |
| `src/make_report.py` | Step 3. Builds the tables (`results/tables/`), the numbers quoted in the text (`results/tables/facts.json`) and the figures (`figures/`) |
| `src/check_against_expected.py` | Step 4 (optional). Compares your results with the paper's results in `expected_results/` |
| `expected_results/` | The result files behind every number in the paper |
| `requirements.txt` | Library versions used for the paper |
| `run_all.sh` | Runs steps 1 to 4 (Linux and macOS) |

## 1. Set up Python

Python 3.11 was used. Create a virtual environment and install the libraries. If your computer has no `python` command, use `python3` in all commands below.

```
python -m venv .venv
source .venv/bin/activate          # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 2. Get the data

The study uses the cleaned versions of the two datasets shared on Kaggle by L. D'hooge:

- NF-UNSW-NB15-v2: https://www.kaggle.com/datasets/dhoogla/nfunswnb15v2 (zip file about 50 MB)
- NF-CSE-CIC-IDS2018-v2: https://www.kaggle.com/datasets/dhoogla/nfcsecicids2018v2 (zip file about 532 MB)

Download them with the Download button on each page, or with these commands:

```
mkdir data
curl -L -o data/unsw.zip https://www.kaggle.com/api/v1/datasets/download/dhoogla/nfunswnb15v2
curl -L -o data/cse.zip https://www.kaggle.com/api/v1/datasets/download/dhoogla/nfcsecicids2018v2
unzip data/unsw.zip -d data
unzip data/cse.zip -d data
```

After this, the `data/` folder must contain `NF-UNSW-NB15-V2.parquet` and `NF-CSE-CIC-IDS2018-V2.parquet`.

The Kaggle files were cleaned with two public notebooks, which removed the IP address columns, rows with missing or infinite values, and exact duplicate rows:

- https://www.kaggle.com/code/dhoogla/nf-unsw-nb15-v2-00-cleaning
- https://www.kaggle.com/code/dhoogla/nf-cse-cic-ids2018-v2-00-cleaning

## 3. Run the study

```
python src/prepare_data.py --unsw data/NF-UNSW-NB15-V2.parquet --cse data/NF-CSE-CIC-IDS2018-V2.parquet --out prepared
python src/run_experiments.py --data prepared --out results
python src/make_report.py --results results --figures figures --data prepared
python src/check_against_expected.py --results results --expected expected_results
```

On Linux or macOS, `bash run_all.sh` runs the same four commands.

**Time and memory.** On a computer with 2 CPU cores and 8 GB of memory, step 1 takes about 1 minute, step 2 about 15 minutes, and step 3 a few seconds (16 minutes in total for the paper's run). Step 1 needs up to about 4 GB of memory, because the CSE file has 17 million rows. It reads the file in batches, so a laptop with 8 GB is enough. The other steps need less.

**Quick test.** To check that everything works before the full run, run one seed of the main experiment only (about 2 minutes):

```
python src/run_experiments.py --data prepared --out results_quick --exp main --seeds 0
```

## 4. Expected output

The last command prints one line per result file. When all files say `OK`, your run has reproduced the paper. Fixed random seeds make the runs repeatable. A different computer or library version can change the last digits of some numbers, so the check allows a difference of up to 0.000001.

Where each part of the paper comes from:

| Paper | File |
|---|---|
| Table 1 (data) | `prepared/data_summary.json` (paper copy: `expected_results/data_summary.json`) and `results/tables/facts.json` |
| Table 2 (baseline replication) | `results/tables/table_replication.md` |
| Table 3 and Figure 1 (main results) | `results/tables/table_main.md`, `figures/fig1_f1_same_vs_cross.png` |
| Figure 2 (per-class detection) | `figures/fig2_cross_dataset_per_class.png`; all classes in `results/tables/table_per_class_all.md` |
| Table 4 (DoS-only test) | `results/tables/table_dos_only_range.md`; per model in `results/tables/table_dos_only.md` |
| Table 5 (top features) | `results/tables/table_top_features.md` |
| Dataset classifier (99.96%) | `results/tables/table_dataset_classifier.md` |
| Other numbers in the text | `results/tables/facts.json` |

## What the code does, step by step

1. **Prepare.** Drops the source and destination ports (flow identifiers) and the two TTL features, which leaves 37 features. Removes flows that are exact copies of an earlier flow (same features and label). Keeps all 236,946 unique UNSW flows and draws a stratified sample of the same size from the 8,849,750 unique CSE flows (seed 42). For the replication, it draws a stratified sample of 1,000,000 flows from each file, keeping duplicates, as in the original protocol.
2. **Experiments.** Every experiment except the error analysis runs with seeds 0 to 4; the error analysis uses seed 0. Each seed makes a new stratified 70/30 train/test split. All features go through log(1 + x) and min-max scaling, fitted on the training data only. The baseline replication uses min-max scaling only, as in the original protocol. Each model is trained on one dataset and tested on the test split of both datasets.
3. **Report.** Averages the five runs and writes the tables and figures.

## Data sources and credit

The NetFlow datasets were created by Sarhan, Layeghy and Portmann (University of Queensland). The cleaned Kaggle versions were prepared by L. D'hooge. Please cite:

- Sarhan, M., Layeghy, S., & Portmann, M. (2022). Towards a standard feature set for network intrusion detection system datasets. *Mobile Networks and Applications, 27*(1), 357-370. https://doi.org/10.1007/s11036-021-01843-0
