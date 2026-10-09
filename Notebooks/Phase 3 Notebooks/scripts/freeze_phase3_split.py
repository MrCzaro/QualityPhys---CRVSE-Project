"""Freezes, as a file, the subject split the shipped PhysNet v2 was actually trained on.

NB_P3_18 trained v2 with NB_P3_07's `subject_split`, but its index also held PhysDrive
(scored zero-shot), so the seed-42 shuffle ran over 785 subjects rather than
NB_P3_07's 739. This script replays NB_P3_18's index and split
from the stores' attributes - no frames are read - checks the result against NB_P3_18's
own printout, adds VitalVideos' frozen 240/60 split, and writes Data/phase3_split.csv.
From then on training and evaluation read the split from that file; nothing
recomputes it.

Columns:
  dataset   the store's `dataset` attribute (MCD, DLCN, UBFC-rPPG, PhysDrive,
            VitalVideos-WW)
  subject   the `subject_id` attribute, as NB_P3_18 read it
  store     the store file
  group     the recording (HDF5 group)
  indexed   the recording passed NB_P3_18's filters (usable, cardiac_sqi >= 0.05,
            n_frames >= 160); only indexed recordings were ever trained on
  split     the subject's assignment, train or val; 'none' for a subject none of
            whose recordings was indexed (never in the shuffle, never trained on).
            VitalVideos keeps its own frozen split (Data/vitalvideos_split.csv).
PhysDrive rows record v2's assignment; PhysDrive is never trained on.

An existing output file is never overwritten: the script compares and reports instead.

Usage:
    python "Notebooks/Phase 3 Notebooks/scripts/freeze_phase3_split.py"
        [--store-dir DIR] [--out FILE] [--vv-split FILE]
"""
import argparse
import random
import sys
from collections import Counter
from pathlib import Path

import h5py
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
DEFAULT_STORE_DIR = Path(r"D:\QualityPhys\phase 3 datasets")
DEFAULT_OUT = REPO / "Data" / "phase3_split.csv"
DEFAULT_VV_SPLIT = REPO / "Data" / "vitalvideos_split.csv"

# The four stores NB_P3_18 indexed, under their dataset attribute.
V2_STORES = {"MCD": "mcd_phase3_rc_none.h5", "DLCN": "dlcn_phase3_rc_none.h5",
             "UBFC-rPPG": "ubfc_rppg_phase3_rc_none.h5",
             "PhysDrive": "physdrive_phase3_rc_none.h5"}
VV_STORE, VV_DATASET = "vitalvideos_phase3_rc_none.h5", "VitalVideos-WW"

# NB_P3_18's constants and its printout, which the replay must reproduce exactly.
SQI_FLOOR, CLIP_LEN, VAL_FRAC, SEED = 0.05, 160, 0.2, 42
EXPECTED_INDEXED = {"MCD": 1191, "DLCN": 777, "UBFC-rPPG": 42, "PhysDrive": 266}
EXPECTED_VAL = {"MCD": 232, "DLCN": 168, "UBFC-rPPG": 7, "PhysDrive": 68}
EXPECTED_SUBJECTS, EXPECTED_VAL_SUBJECTS = 785, 157

# Subjects later used as "held-out" sets, chosen from NB_P3_07's split.
REGRESSION_UBFC = ["11", "13", "24", "25", "34", "35", "42", "47"]
MCD_DEMO = ["4087", "4874", "5130", "6137", "8584", "9092"]
COLUMNS = ["dataset", "subject", "store", "group", "indexed", "split"]


def _text(value):
    """An HDF5 attribute as text, whether h5py returns it as str, bytes or a number."""
    return value.decode() if isinstance(value, bytes) else str(value)


def read_store(path):
    """One row per recording, with NB_P3_18's filters applied in its order."""
    rows = []
    with h5py.File(path, "r") as f:
        for name in f.keys():
            a = f[name].attrs
            usable = bool(a.get("usable", True))
            sqi = float(a.get("cardiac_sqi", 0.0))
            n_frames = int(a.get("n_frames", 0))
            rows.append(dict(dataset=_text(a.get("dataset", path.stem)),
                             subject=_text(a.get("subject_id", name)),
                             store=path.name, group=name,
                             indexed=usable and sqi >= SQI_FLOOR and n_frames >= CLIP_LEN,
                             vv_split=_text(a.get("split", ""))))
    return rows


def subject_split(index):
    """NB_P3_07's subject_split, as NB_P3_18 called it: the validation subject set."""
    subjects = sorted({r["dataset"] + "/" + r["subject"] for r in index})
    rng = random.Random(SEED)
    rng.shuffle(subjects)
    n_val = max(1, int(round(len(subjects) * VAL_FRAC)))
    return subjects, set(subjects[:n_val])


def check(label, got, expected, problems):
    ok = got == expected
    print(f"  {label}: {got}" + ("" if ok else f"   <- NB_P3_18 printed {expected}"))
    if not ok:
        problems.append(label)


def replay_v2(store_dir):
    rows = []
    for dataset, fn in V2_STORES.items():
        path = store_dir / fn
        if not path.exists():
            raise SystemExit(f"ERROR: store not found: {path}")
        store_rows = read_store(path)
        found = {r["dataset"] for r in store_rows}
        if found != {dataset}:
            raise SystemExit(f"ERROR: {fn} holds dataset attribute(s) {sorted(found)}, "
                             f"expected '{dataset}'")
        rows += store_rows
    table = pd.DataFrame(rows)
    index = table[table["indexed"]].to_dict("records")
    subjects, val = subject_split(index)
    key = table["dataset"] + "/" + table["subject"]
    in_shuffle = key.isin(set(subjects))
    table["split"] = "none"
    table.loc[in_shuffle, "split"] = "train"
    table.loc[in_shuffle & key.isin(val), "split"] = "val"

    print("Replaying NB_P3_18's index and split (seed 42) from the stores' attributes:")
    problems = []
    indexed = table[table["indexed"]]
    check("indexed recordings", dict(sorted(Counter(indexed["dataset"]).items())),
          dict(sorted(EXPECTED_INDEXED.items())), problems)
    check("subjects in the shuffle", len(subjects), EXPECTED_SUBJECTS, problems)
    check("validation subjects", len(val), EXPECTED_VAL_SUBJECTS, problems)
    check("validation recordings",
          dict(sorted(Counter(indexed.loc[indexed["split"] == "val", "dataset"]).items())),
          dict(sorted(EXPECTED_VAL.items())), problems)
    if problems:
        raise SystemExit("ERROR: the replay does not reproduce NB_P3_18 ("
                         + ", ".join(problems) + "); nothing written")
    return table


def add_vitalvideos(store_dir, vv_split_path):
    path = store_dir / VV_STORE
    if not path.exists():
        raise SystemExit(f"ERROR: store not found: {path}")
    table = pd.DataFrame(read_store(path))
    if set(table["dataset"]) != {VV_DATASET}:
        raise SystemExit(f"ERROR: {VV_STORE} holds dataset attribute(s) "
                         f"{sorted(set(table['dataset']))}")
    frozen = pd.read_csv(vv_split_path, dtype=str).set_index("guid")["split"]
    table["split"] = table["subject"].map(frozen)
    missing = table["split"].isna()
    if missing.any() or len(table) != len(frozen):
        raise SystemExit(f"ERROR: {VV_STORE} and {vv_split_path.name} disagree on the subjects "
                         f"({int(missing.sum())} store subjects not in the file; "
                         f"{len(table)} groups against {len(frozen)} rows)")
    differ = table["vv_split"] != table["split"]
    if differ.any():
        raise SystemExit(f"ERROR: the store's split attribute disagrees with "
                         f"{vv_split_path.name} for {int(differ.sum())} subject(s)")
    print(f"\nVitalVideos: {len(table)} recordings, "
          f"{dict(sorted(Counter(table['split']).items()))}, matching {vv_split_path.name} "
          f"and the store's split attribute; indexed {int(table['indexed'].sum())}")
    return table


def report_heldout_claims(table):
    """Where the subjects later called held-out sit in v2's split."""
    def where(dataset, ids):
        sub = table[table["dataset"] == dataset].drop_duplicates("subject").set_index("subject")
        return {k: (sub.loc[k, "split"] if k in sub.index else "absent") for k in ids}

    print("\nSubjects chosen from NB_P3_07's split, as v2 actually saw them:")
    for label, dataset, ids in (("UBFC regression set", "UBFC-rPPG", REGRESSION_UBFC),
                                ("MCD demo subset", "MCD", MCD_DEMO)):
        seen = where(dataset, ids)
        trained = [k for k, v in seen.items() if v == "train"]
        print(f"  {label}: {len(trained)} of {len(ids)} in v2's training split "
              f"({', '.join(trained) or 'none'}); held out: "
              f"{', '.join(k for k, v in seen.items() if v != 'train') or 'none'}")


def write_or_compare(table, out):
    table = (table[COLUMNS].sort_values(["dataset", "store", "group"])
             .reset_index(drop=True))
    if out.exists():
        old = pd.read_csv(out, dtype=str, keep_default_na=False)
        new = table.astype(str)
        if list(old.columns) == COLUMNS and old.equals(new):
            print(f"\n{out.name} already exists and matches this replay; left unchanged.")
            return 0
        print(f"\nERROR: {out} exists and differs from this replay; left unchanged. "
              "Move it aside only if you know why it differs.")
        return 1
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False)
    counts = table.groupby(["dataset", "split"]).size().unstack(fill_value=0)
    print(f"\nwrote {out} ({len(table)} recordings)")
    print(counts.to_string())
    return 0


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--store-dir", type=Path, default=DEFAULT_STORE_DIR)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--vv-split", type=Path, default=DEFAULT_VV_SPLIT)
    args = p.parse_args()
    if not args.vv_split.exists():
        raise SystemExit(f"ERROR: {args.vv_split} not found")
    v2 = replay_v2(args.store_dir)
    vv = add_vitalvideos(args.store_dir, args.vv_split)
    report_heldout_claims(v2)
    return write_or_compare(pd.concat([v2, vv], ignore_index=True), args.out)


if __name__ == "__main__":
    sys.exit(main())