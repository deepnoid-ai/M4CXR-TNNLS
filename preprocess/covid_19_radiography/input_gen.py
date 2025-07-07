import json
import os
from glob import glob

import cv2
import numpy as np
from natsort import natsorted
from PIL import Image, ImageDraw
from tqdm import tqdm

data_root_path = "/share-data/advanced_tech/datasets/COVID-19_Radiography_Dataset"
save_root_path = os.path.join(data_root_path, "preprocess/v0.0")

covid_data_root_path = os.path.join(data_root_path, "COVID")
opacity_data_root_path = os.path.join(data_root_path, "Lung_opacity")
normal_data_root_path = os.path.join(data_root_path, "Normal")
pneumonia_data_root_path = os.path.join(data_root_path, "Viral Pneumonia")


def make_dict(root_path, lesion_class):
    image_path_list = natsorted(glob(os.path.join(root_path, "images/*.png")))
    mask_path_list = natsorted(glob(os.path.join(root_path, "masks/*.png")))

    input_dict_list = []
    for image_path, mask_path in zip(tqdm(image_path_list), mask_path_list):
        assert os.path.basename(image_path) == os.path.basename(mask_path)

        lesion_pil = Image.open(mask_path)
        lesion_mask = np.array(lesion_pil) // 255
        lesion_mask = cv2.cvtColor(lesion_mask, cv2.COLOR_BGR2GRAY)
        lesion_bbox = generate_bbox(lesion_mask)

        input_dict = {
            "image": os.path.join(*image_path.split("/")[4:]),
            "cls": [
                {"name": lesion_class, "label": "positive"},
                {"name": "lung", "label": "positive"},
            ],
            "det": [{"name": "lung", "label": lesion_bbox}],
            "seg": [{"name": "lung", "label": os.path.join(*mask_path.split("/")[4:])}],
        }
        input_dict_list.append(input_dict)

    return input_dict_list


def generate_bbox(lesion_mask, lesion_num_max=2):
    lesion_mask_contour_idxs, _ = cv2.findContours(
        lesion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )

    lesion_bboxs = []
    if len(lesion_mask_contour_idxs):  # <= lesion_num_max
        for contour in lesion_mask_contour_idxs:
            x1, y1, w, h = cv2.boundingRect(contour)
            lesion_bboxs.append([x1, y1, x1 + w, y1 + h])
    return lesion_bboxs


def test_visualize(lesion_pil, lesion_bboxs):
    for bbox in lesion_bboxs:
        minr, minc, maxr, maxc = bbox

        lesion_pil = lesion_pil.convert("RGB")
        draw = ImageDraw.Draw(lesion_pil)
        draw.rectangle((minr, minc, maxr, maxc), outline=(255, 0, 0), width=3)

        lesion_pil.save("./test.png")


covid_dict_list = make_dict(covid_data_root_path, "COVID-19")
opacity_dict_list = make_dict(opacity_data_root_path, "lung opacity")
pneumonia_dict_list = make_dict(pneumonia_data_root_path, "viral pneumonia")

# Normal
image_path_list = natsorted(glob(os.path.join(normal_data_root_path, "images/*.png")))
mask_path_list = natsorted(glob(os.path.join(normal_data_root_path, "masks/*.png")))

normal_dict_list = []
for image_path, mask_path in zip(tqdm(image_path_list), mask_path_list):
    assert os.path.basename(image_path) == os.path.basename(mask_path)

    lesion_pil = Image.open(mask_path)
    lesion_mask = np.array(lesion_pil) // 255
    lesion_mask = cv2.cvtColor(lesion_mask, cv2.COLOR_BGR2GRAY)
    lesion_bbox = generate_bbox(lesion_mask)

    input_dict = {
        "image": os.path.join(*image_path.split("/")[4:]),
        "cls": [
            {"name": "COVID-19", "label": "negative"},
            {"name": "lung opacity", "label": "negative"},
            {"name": "viral pneumonia", "label": "negative"},
            {"name": "lung", "label": "positive"},
        ],
        "det": [{"name": "lung", "label": lesion_bbox}],
        "seg": [{"name": "lung", "label": os.path.join(*mask_path.split("/")[4:])}],
    }
    normal_dict_list.append(input_dict)

input_json = (
    covid_dict_list + opacity_dict_list + pneumonia_dict_list + normal_dict_list
)

with open(os.path.join(save_root_path, "input.json"), "w") as j:
    j.write(json.dumps(input_json, indent=2))
