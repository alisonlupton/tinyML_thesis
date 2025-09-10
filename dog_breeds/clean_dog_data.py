#clean_dog_data.py
from pathlib import Path
import pandas as pd
from scipy.io import loadmat
import json
import yaml
from utils import load_config

def _parse_list(mat_path):
    m = loadmat(mat_path)
    files  = [str(x[0]).strip() for x in m['file_list'].squeeze()]
    labels = [int(x) for x in m['labels'].squeeze()]  #1..120
    return files, labels

def main():

    cfg = load_config()
    data_root = cfg['data_root']
    out_csv = cfg['cleaned_data_path_csv']
    out_json = cfg['cleaned_data_path_json']

    root = Path(data_root)
    tr_files, tr_labels = _parse_list(root/"lists"/"train_list.mat")
    te_files, te_labels = _parse_list(root/"lists"/"test_list.mat")

    def rows(files, labels, split):
        for fp, y in zip(files, labels):
            breed = fp.split('/')[0]  #e.g., n02085620-Chihuahua
            yield dict(
                split=split,
                rel_path=fp,
                img_path=str(root/"Images"/fp),
                ann_path=str(root/"Annotations"/(fp.replace('.jpg',''))), #folder + xml name
                breed=breed,
                gid=int(y)-1   #0..119
            )

    df = pd.DataFrame([*rows(tr_files,tr_labels,"train"), *rows(te_files,te_labels,"test")])
    df.to_csv(out_csv, index=False)

    gid2breed = df.groupby("gid")["breed"].first().sort_index().to_dict()
    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(out_json).write_text(json.dumps(gid2breed, indent=2))
    print(f"Wrote {out_csv} and {out_json}")

if __name__ == "__main__":
    main()
    
    
