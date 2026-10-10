#!/bin/bash
# Runs every step of the study and compares the output with the paper's results.
# Usage: bash run_all.sh   (the two parquet files must be in the data/ folder)
set -e
python src/prepare_data.py --unsw data/NF-UNSW-NB15-V2.parquet --cse data/NF-CSE-CIC-IDS2018-V2.parquet --out prepared
python src/run_experiments.py --data prepared --out results
python src/make_report.py --results results --figures figures --data prepared
python src/check_against_expected.py --results results --expected expected_results
