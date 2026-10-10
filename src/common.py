"""Settings and helper functions shared by all scripts."""
import pandas as pd

LABEL = "Label"    # 0 = benign, 1 = attack
ATTACK = "Attack"  # attack name as given in the dataset
GROUP = "Group"    # attack class used in the per-class analysis

DATASETS = ["UNSW", "CSE"]
FULL_NAMES = {"UNSW": "NF-UNSW-NB15-v2", "CSE": "NF-CSE-CIC-IDS2018-v2"}

# Flow identifiers. The two IP address columns were already removed in the
# cleaned Kaggle files (D'hooge, 2023a, 2023b), so only the ports are left.
ID_COLUMNS = ["L4_SRC_PORT", "L4_DST_PORT"]

# TTL features. Sarhan et al. (2022) dropped them for UNSW-NB15 because they are
# almost perfectly correlated with the label. The main experiments drop them
# from both datasets, so that both datasets have exactly the same 37 features.
TTL_COLUMNS = ["MIN_TTL", "MAX_TTL"]

# CSE-CIC-IDS2018 attack names grouped into the six classes used by
# Sarhan et al. (2022). The UNSW-NB15 names are already at this level.
CSE_GROUPS = {
    "Benign": "Benign",
    "DDOS attack-HOIC": "DDoS",
    "DDoS attacks-LOIC-HTTP": "DDoS",
    "DDOS attack-LOIC-UDP": "DDoS",
    "DoS attacks-Hulk": "DoS",
    "DoS attacks-GoldenEye": "DoS",
    "DoS attacks-SlowHTTPTest": "DoS",
    "DoS attacks-Slowloris": "DoS",
    "SSH-Bruteforce": "BruteForce",
    "FTP-BruteForce": "BruteForce",
    "Brute Force -Web": "Web",
    "Brute Force -XSS": "Web",
    "SQL Injection": "Web",
    "Infilteration": "Infiltration",
    "Bot": "Bot",
}


def attack_group(attack: pd.Series, dataset: str) -> pd.Series:
    """Return the attack class of every flow (Benign for normal traffic)."""
    if dataset == "CSE":
        group = attack.astype(str).map(CSE_GROUPS)
        if group.isna().any():
            unknown = sorted(attack[group.isna()].unique())
            raise ValueError(f"Unknown attack names in CSE data: {unknown}")
        return group
    return attack.astype(str)


def feature_columns(df: pd.DataFrame, drop_ttl: bool = True) -> list:
    """Input features: every column except identifiers, labels and (optionally) TTL."""
    drop = set(ID_COLUMNS + [LABEL, ATTACK, GROUP])
    if drop_ttl:
        drop |= set(TTL_COLUMNS)
    return [c for c in df.columns if c not in drop]
