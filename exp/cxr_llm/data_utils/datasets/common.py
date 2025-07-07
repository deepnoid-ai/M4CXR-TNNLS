import csv
import json
import os
import re
from typing import Union

import numpy as np
from PIL import Image

from ..templates import MEDIA_TOKENS


def load_json_files(input_files: Union[list[str], str], key_pattern=None):
    raw_data_lst = []

    if isinstance(input_files, str):
        input_files = [input_files]

    for input_file in input_files:
        with open(input_file, "r") as f:
            data_temp = json.load(f)
        if key_pattern is not None:
            data_temp = data_temp[key_pattern]
        raw_data_lst.extend(data_temp)

    return raw_data_lst


def csv_file_read(csv_path):
    with open(csv_path, newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        data = list(reader)
    return data


def mscxr_duplicate(data_root, mscxr_path):
    ms_cxr_dataset = csv_file_read(os.path.join(data_root, mscxr_path))

    mscxr_lst = []
    for data in ms_cxr_dataset:
        image_path = os.path.join("MIMIC-CXR", "images", data["dicom_id"] + ".jpg")
        mscxr_lst.append(image_path)

    return set(mscxr_lst)


def mimic_cxr_train(data_root, mimic_cxr_train_path):
    mimic_cxr_dataset = load_json_files(os.path.join(data_root, mimic_cxr_train_path))

    mimic_cxr_lst = []
    for data in mimic_cxr_dataset:
        image_lst = data["dicom_id"]
        for image in image_lst:
            mimic_cxr_lst.append(image)

    return set(mimic_cxr_lst)


def mimic_cxr_test(data_root, mimic_cxr_test_path):
    mimic_cxr_dataset = load_json_files(os.path.join(data_root, mimic_cxr_test_path))

    mimic_cxr_lst = []
    for data in mimic_cxr_dataset:
        image_lst = data["dicom_id"]
        for image in image_lst:
            mimic_cxr_lst.append(image)

    return set(mimic_cxr_lst)


def chunking_by_keyword(txt, keyword_patterns=["<image>\n", "\n<image>"]):
    pattern = "|".join(map(re.escape, keyword_patterns))
    chunk_strs = re.split(f"({pattern})", txt)
    chunk_strs = [x for x in chunk_strs if len(x) > 0]

    return chunk_strs


def remove_special_token_from_text(txt, patterns=None):
    if patterns is not None:
        for pattern in patterns:
            if pattern in txt:
                txt = txt.replace(pattern, "")

    # if a special media token in the conversation, replace it to a non-special token.
    for v in MEDIA_TOKENS.values():
        for v_ in v:
            txt = txt.replace(v_, "".join([c for c in v_ if c not in ["<", ">"]]))

    return txt


def normalize_bbox(x1, y1, x2, y2, width, height, resize_type="shortest_edge"):

    if resize_type == "shortest_edge":
        min_length = min(width, height)

        if height >= width:
            norm_x1 = round(100 * x1 / min_length)
            norm_x2 = round(100 * x2 / min_length)

            half_diff = (height - min_length) / 2
            norm_y1 = round(100 * (y1 - half_diff) / min_length)
            norm_y1 = np.clip(norm_y1, 0, 100)
            norm_y2 = round(100 * (y2 - half_diff) / min_length)
            norm_y2 = np.clip(norm_y2, 0, 100)

        else:
            half_diff = (width - min_length) / 2
            norm_x1 = round(100 * (x1 - half_diff) / min_length)
            norm_x1 = np.clip(norm_x1, 0, 100)
            norm_x2 = round(100 * (x2 - half_diff) / min_length)
            norm_x2 = np.clip(norm_x2, 0, 100)

            norm_y1 = round(100 * y1 / min_length)
            norm_y2 = round(100 * y2 / min_length)

        norm_bbox = [norm_x1, norm_y1, norm_x2, norm_y2]

    elif resize_type == "longest_edge":
        norm_bbox = [
            round(100 * coord / max((width, height))) for coord in [x1, y1, x2, y2]
        ]

    bbox_str = ",".join([str(i) for i in norm_bbox])
    bbox_str = f"[{bbox_str}]"
    return bbox_str


def get_image_size(file_path):
    with Image.open(file_path) as img:
        width, height = img.size
    return width, height
