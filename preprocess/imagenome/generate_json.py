import json
import os

import pandas as pd
from constants import REVERSED_ANATOMICAL_REGIONS
from tqdm import tqdm

# https://github.com/ttanida/rgrg/blob/main/src/dataset/create_dataset.py
# 1. Preprocess ImaGenome with RGRG codes
# 2. run the codes below.


def preprocess_imagenome(data_path):
    df_csv = pd.read_csv(data_path)

    mimic_paths = df_csv["mimic_image_file_path"].values  # mimic path
    coords = df_csv["bbox_coordinates"].values  # coord
    anatomical_regions = df_csv["bbox_labels"].values  # anatomical region
    phrases = df_csv["bbox_phrases"].values  # phrase

    imagenome_data = []
    for (
        mimic_path_patient,
        coord_patient,
        anatomical_region_patient,
        phrase_patient,
    ) in zip(tqdm(mimic_paths), coords, anatomical_regions, phrases):
        region_bbox = []
        for n, anatomical_idx in enumerate(eval(anatomical_region_patient)):
            region = REVERSED_ANATOMICAL_REGIONS[anatomical_idx - 1]
            bbox = eval(coord_patient)[n]
            region_bbox.append(
                {
                    "region": region,
                    "bbox": bbox,
                }
            )

        region_phrase = []
        for region_n in REVERSED_ANATOMICAL_REGIONS:
            phrase = eval(phrase_patient)[region_n]
            region_phrase.append(
                {
                    "region": REVERSED_ANATOMICAL_REGIONS[region_n],
                    "phrase": phrase,
                }
            )

        imagenome_data.append(
            {
                "image": mimic_path_patient,
                "bbox": region_bbox,
                "phrase": region_phrase,
            }
        )
    return imagenome_data


if __name__ == "__main__":
    data_root_path = (
        "/share-data/advanced_tech/datasets/chest-imagenome-dataset-1.0.0/preprocess/"
    )
    split = "train"  # train, valid, test
    data_path = os.path.join(data_root_path, f"{split}.csv")

    data_dicts = preprocess_imagenome(data_path)

    save_root_path = os.path.join(data_root_path, "v2.0")
    with open(os.path.join(save_root_path, f"{split}.json"), "w") as j:
        j.write(json.dumps(data_dicts, indent=2))
