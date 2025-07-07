import json
import os
from glob import glob

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

data_root_path = "datasets/NIH"

image_paths = glob(os.path.join(data_root_path, "*", "images", "*.png"))
image_fnames = [os.path.basename(image_path) for image_path in image_paths]
cls_df = pd.read_csv(os.path.join(data_root_path, "Data_Entry_2017.csv"))
bbox_df = pd.read_csv(os.path.join(data_root_path, "BBox_List_2017.csv"))

save_root_path = os.path.join(data_root_path, "preprocess/v0.0")

cls_json = []
cls_ids = []
all_findings = []
for index, row in cls_df.iterrows():
    cls_id = os.path.join(
        *image_paths[image_fnames.index(row["Image Index"])].split("/")[4:]
    )
    cls_json.append(
        {
            "id": cls_id,
            "finding": row["Finding Labels"],
        }
    )
    cls_ids.append(cls_id)

    all_findings += [f for f in row["Finding Labels"].split("|")]

all_findings = np.unique(all_findings)
all_findings = all_findings[all_findings != "No Finding"]

bbox_json = []
bbox_ids = []
for index, row in bbox_df.iterrows():
    x1, y1 = row["Bbox [x"], row["y"]
    w, h = row["w"], row["h]"]
    bbox_id = os.path.join(
        *image_paths[image_fnames.index(row["Image Index"])].split("/")[4:]
    )
    bbox_json.append(
        {
            "id": bbox_id,
            "finding": row["Finding Label"],
            "bbox": [x1, y1, x1 + w, y1 + h],
        }
    )
    bbox_ids.append(bbox_id)

train_val_list = []
with open(os.path.join(data_root_path, "train_val_list.txt"), "r") as file:
    lines = file.readlines()
    for line in lines:
        train_val_list.append(line[:-1])


test_list = []
with open(os.path.join(data_root_path, "test_list.txt"), "r") as file:
    lines = file.readlines()
    for line in lines:
        test_list.append(line[:-1])

input_json = []
for cls_element in cls_json:
    cls_finding_list = [
        {"name": finding, "label": "positive"}
        for finding in cls_element["finding"].split("|")
    ]

    # bbox exist
    if cls_element["id"] in bbox_ids:
        bbox_idx = bbox_ids.index(cls_element["id"])
        bbox_finding = bbox_json[bbox_idx]["finding"]

        det_list = []
        for cls_finding in cls_finding_list:
            # finding with bbox
            if cls_finding["name"] == bbox_finding:
                bbox_dict = {
                    "name": bbox_finding,
                    "label": [int(b_i) for b_i in bbox_json[bbox_idx]["bbox"]],
                }
                det_list.append(bbox_dict)

        input_json.append(
            {"image": cls_element["id"], "cls": cls_finding_list, "det": det_list}
        )

    # no bbox
    else:
        # No Finding
        if cls_finding_list[0]["name"] == "No Finding":
            no_finding_cls = [
                {"name": all_finding, "label": "negative"}
                for all_finding in all_findings
            ]
            no_finding_det = [
                {"name": all_finding, "label": [[0, 0, 0, 0]]}
                for all_finding in all_findings
            ]

            input_json.append(
                {
                    "image": cls_element["id"],
                    "cls": no_finding_cls,
                    "det": no_finding_det,
                }
            )

        # Finding exist
        else:
            input_json.append(
                {
                    "image": cls_element["id"],
                    "cls": cls_finding_list,
                }
            )

with open(os.path.join(save_root_path, "input.json"), "w") as j:
    j.write(json.dumps(input_json, indent=2))

train_eval_input_json, test_input_json = [], []
for input_j in input_json:
    if os.path.basename(input_j["image"]) in train_val_list:
        train_eval_input_json.append(input_j)
    elif os.path.basename(input_j["image"]) in test_list:
        test_input_json.append(input_j)

with open(os.path.join(save_root_path, "train_eval_input.json"), "w") as j:
    j.write(json.dumps(train_eval_input_json, indent=2))

with open(os.path.join(save_root_path, "test_input.json"), "w") as j:
    j.write(json.dumps(test_input_json, indent=2))
