import json
import os
from glob import glob

import numpy as np
import pandas as pd
from tqdm import tqdm

data_root_path = "datasets/RSNA-pneumonia/"
image_root_path = os.path.join(data_root_path, "images")

image_path_list = glob(os.path.join(image_root_path, "*", "*.jpg"))
image_basename_list = [os.path.basename(image_path) for image_path in image_path_list]
image_fname_list = [image_basename[:-4] for image_basename in image_basename_list]

cls_df = pd.read_csv(os.path.join(data_root_path, "stage_2_detailed_class_info.csv"))
cls_id = cls_df["patientId"].values
cls_class = cls_df["class"].values

det_df = pd.read_csv(os.path.join(data_root_path, "stage_2_train_labels.csv"))
det_id = det_df["patientId"].values
det_x1, det_y1 = det_df["x"].values, det_df["y"].values
det_w, det_h = det_df["width"].values, det_df["height"].values

det_bbox = []
for x1, y1, w, h in zip(det_x1, det_y1, det_w, det_h):
    if np.isnan(x1):
        det_bbox.append([])
    else:
        det_bbox.append([int(x1), int(y1), int(x1 + w), int(y1 + h)])

input_json = []
for idx, cls_i in enumerate(tqdm(cls_id)):
    cls_c = cls_class[idx]
    if cls_c == "Normal":
        cls_name = "lung opacity"
        cls_label = "negative"
    elif cls_c == "Lung Opacity":
        cls_name = "lung opacity"
        cls_label = "positive"
    else:
        continue

    det_i = det_id[idx]
    det_b = det_bbox[idx]

    assert det_i == cls_i

    if cls_c == "Normal":
        det_name = "lung opacity"
        det_label = [[0, 0, 0, 0]]
    elif cls_c == "Lung Opacity":
        det_name = "lung opacity"
        det_label = [det_b]
    else:
        continue

    img_path = image_path_list[image_fname_list.index(cls_i)]
    img_path = os.path.join(*img_path.split("/")[4:])

    input_dict = {
        "image": img_path,
        "cls": [{"name": cls_name, "label": cls_label}],
        "det": [{"name": cls_name, "label": det_label}],
    }

    input_json.append(input_dict)

image_list = [d["image"] for d in input_json]
image_list_uni = np.unique(image_list)

final_input_json = []
for n, image_uni in enumerate(tqdm(image_list_uni)):
    match_idx = np.where(np.array(image_list) == image_uni)[0]

    det_list = []
    for i in match_idx:
        det_list.append(input_json[i]["det"][0]["label"][0])

    data_dict = {
        "image": image_uni,
        "cls": input_json[i]["cls"],
        "det": [{"name": "pneumonia", "label": det_list}],
    }
    final_input_json.append(data_dict)


save_root_path = os.path.join(data_root_path, "preprocess")
with open(os.path.join(save_root_path, "input.json"), "w") as j:
    json.dump(input_json, j, indent=2)
