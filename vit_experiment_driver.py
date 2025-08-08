#!/usr/bin/env python
"""
Grid-search ViT size × window length on %-Silica forecasting.
"""
from pathlib import Path
import pandas as pd
from prepare_datasets import main as build_dataset      # step 1
from run_vit_augreg         import run_vit                    # step 2

CSV   = "data_cleaned.csv"      # passes through prepare_datasets
VITS  = ["vit_tiny_patch16_224",
         "vit_small_patch16_224",
         "vit_base_patch16_224"]
WINDOWS = [24, 48, 96, 144]
EPOCHS = [20,100]


def main():
    records = []
    for W in WINDOWS:
        # 1) rebuild dataset if it doesn’t exist
        ds_name = f"data_gaf_W{W}.npz"
        dsraw_name = f"data_raw_W{W}.npz"
        if not Path(ds_name).exists():
            build_dataset(window=W,dsgaf_path=ds_name, dsraw_path=dsraw_name)          # calls prepare_datasets.py logic

        # 2) loop over ViT backbones
        for vit in VITS:
            for epoch in EPOCHS:
                out_json = f"results_{vit}_W{W}_EP{epoch}.json"
                _, _, metrics = run_vit_augreg(epochs=epoch, data_path=ds_name,
                                    vit_name=vit,
                                    out_json=out_json)
                records.append({"epochs": epoch,"window": W, "vit": vit, **metrics})

    # 3) summarise all runs
    df = pd.DataFrame(records).sort_values(["window", "vit"])
    df.to_csv("experiment_summary.csv", index=False)
    print(df)

if __name__ == '__main__':

    main()