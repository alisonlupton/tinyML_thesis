# prepare_dogmove_data.py (paper-aligned cleaner for CNN pipeline)

import os
import sys
from typing import List, Optional
import argparse
import pandas as pd
import yaml

def main():
    parser = argparse.ArgumentParser(description="Clean DogMoveData (paper-style): keep Behavior_1..3, canonicalize.")
    parser.add_argument("--config", type=str, default=os.path.join(os.path.dirname(__file__), "dog_config.yaml"))
    parser.add_argument("--random_breeds", type=int, default=0)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        conf = yaml.safe_load(f)

    dog_clean = conf.get("dogmove_clean", {})
    raw_path = dog_clean.get("raw_data_path", "../data/DogMoveData_csv_format/DogMoveData.csv")
    info_path = dog_clean.get("dog_info_path", "../data/DogMoveData_csv_format/DogInfo.xlsx")
    out_path = dog_clean.get("output_path", "../data/DogMoveData_csv_format/DogMoveData_clean.csv")
    select_breeds: List[str] = dog_clean.get("select_breeds", []) or []
    select_dog_ids: List[int] = dog_clean.get("select_dog_ids", []) or []

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # Load dog info
    try:
        df_info = pd.read_excel(info_path)
    except ImportError:
        print("openpyxl is required: pip install openpyxl", file=sys.stderr)
        raise
    if "DogID" not in df_info.columns:
        raise ValueError("DogInfo.xlsx must contain 'DogID'")
    if "Breed" not in df_info.columns:
        raise ValueError("Breed column not found in DogInfo.xlsx")
    df_info_small = df_info[["DogID", "Breed"]].copy()

    # Read subset of columns
    feature_cols = [
        "ABack_x","ABack_y","ABack_z",
        "ANeck_x","ANeck_y","ANeck_z",
        "GBack_x","GBack_y","GBack_z",
        "GNeck_x","GNeck_y","GNeck_z",
    ]
    base_cols = ["DogID","TestNum","t_sec","Task"]  # keep Task if present; it's ok if missing
    label_cols = ["Behavior_1","Behavior_2","Behavior_3"]
    usecols = [c for c in (base_cols + feature_cols + label_cols) if c]  # some may be absent

    print(f"Reading raw CSV: {raw_path}")
    
    df = pd.read_csv(raw_path, usecols=lambda c: c in usecols, low_memory=False)
    # Merge breed info
    df = df.merge(df_info_small, on="DogID", how="left")
    

    # Optional filtering
    if select_breeds:
        df = df[df["Breed"].isin(select_breeds)]
        print(f"Filtered to breeds: {select_breeds} -> {df['DogID'].nunique()} dogs, {len(df)} rows")
    if select_dog_ids:
        df = df[df["DogID"].isin(select_dog_ids)]
        print(f"Filtered to dog IDs: {select_dog_ids} -> {df['DogID'].nunique()} dogs, {len(df)} rows")

            
    # Sort for consistency
    sort_cols = [c for c in ["DogID","TestNum","t_sec"] if c in df.columns]
    df = df.sort_values(sort_cols, kind="mergesort")
    
    
    # Drop rows where ALL behaviors are undefined
    behavior_cols = ["Behavior_1","Behavior_2","Behavior_3"]
    undefined_mask = df[behavior_cols].apply(lambda x: x == '<undefined>').all(axis=1)
    df = df[~undefined_mask].copy()
    print(f"Dropped {undefined_mask.sum()} rows with all undefined behaviors")
    

    # Save
    df.to_csv(out_path, index=False)
    print(f"Saved cleaned dataset: {out_path}")
    print(f"Rows: {len(df):,} | Dogs: {df['DogID'].nunique()} | Breeds: {df['Breed'].nunique()}")

if __name__ == "__main__":
    main()
    
    
    
    
#                           NumDogs  NumBehaviors
# Breed                                      
# Border Collie               4            18
# Crossbreed                  4            18
# Beauceron                   3            15
# German Shepherd             3            18
# Golden Retriever            3            13
# Labrador Retriever          3            19
# Bouvier des Ardennes        2            19
# Dutch Shepherd              2            15
# Hovawart                    2            14
# Lapponian Herder            2            16
# Spanish Water Dog           2            17
    
# Dogs with all 7 behaviors:
# Dog 16: 17 behaviors
# Dog 18: 17 behaviors
# Dog 19: 18 behaviors
# Dog 20: 19 behaviors
# Dog 21: 17 behaviors
# Dog 22: 18 behaviors
# Dog 23: 18 behaviors
# Dog 25: 17 behaviors
# Dog 26: 17 behaviors
# Dog 27: 17 behaviors
# Dog 28: 17 behaviors
# Dog 29: 17 behaviors
# Dog 30: 17 behaviors
# Dog 34: 19 behaviors
# Dog 36: 16 behaviors
# Dog 39: 18 behaviors
# Dog 41: 18 behaviors
# Dog 43: 17 behaviors
# Dog 44: 16 behaviors
# Dog 45: 18 behaviors
# Dog 46: 16 behaviors
# Dog 47: 19 behaviors
# Dog 48: 17 behaviors
# Dog 49: 18 behaviors
# Dog 51: 17 behaviors
# Dog 52: 19 behaviors
# Dog 53: 18 behaviors
# Dog 54: 18 behaviors
# Dog 55: 18 behaviors
# Dog 56: 18 behaviors
# Dog 57: 18 behaviors
# Dog 58: 18 behaviors
# Dog 59: 18 behaviors
# Dog 60: 17 behaviors
# Dog 61: 17 behaviors
# Dog 63: 15 behaviors
# Dog 65: 17 behaviors
# Dog 66: 16 behaviors
# Dog 67: 17 behaviors
# Dog 68: 18 behaviors
# Dog 70: 18 behaviors
# Dog 72: 18 behaviors
# Dog 73: 17 behaviors
# Dog 74: 18 behaviors