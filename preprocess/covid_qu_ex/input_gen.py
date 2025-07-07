import json
import os
import shutil
from glob import glob

import cv2
import numpy as np
from natsort import natsorted
from PIL import Image, ImageDraw
from skimage.measure import regionprops
from tqdm import tqdm


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


def data_json(lesion, lesion_bbox, lung_bbox, lesion_mask_path, lung_mask_path):
    if lesion == "Normal":
        data = {
            "image": img_path,
            "cls": [{"name": "lung", "label": "positive"}],
            "det": [{"name": "lung", "label": lung_bbox}],
            "seg": [{"name": "lung", "label": lung_mask_path}],
            "text": "",
        }
    elif lesion == "Non-COVID":  # TODO
        data = {
            "image": img_path,
            "cls": [
                {"name": "COVID-19", "label": "negative"},
                {"name": "pneumonia", "label": "positive"},
                {"name": "lung", "label": "positive"},
            ],
            "det": [
                {"name": "COVID-19", "label": [[0, 0, 0, 0]]},
                {"name": "lung", "label": lung_bbox},
            ],
            "seg": [
                {
                    "name": "COVID-19",
                    "label": "negative",
                },
                {"name": "lung", "label": lung_mask_path},
            ],
            "text": "",
        }
    else:
        data = {
            "image": img_path,
            "cls": [
                {"name": lesion, "label": "positive"},
                {"name": "lung", "label": "positive"},
            ],
            "det": [
                {"name": lesion, "label": lesion_bbox},
                {"name": "lung", "label": lung_bbox},
            ],
            "seg": [
                {"name": lesion, "label": lesion_mask_path},
                {"name": "lung", "label": lung_mask_path},
            ],
            "text": "",
        }
    return data


root_path = "/share-data/advanced_tech/datasets"
dataset_dir = os.path.join(root_path, "COVID-QU-Ex/Infection Segmentation Data/")
target_dataset_dir = os.path.join(root_path, "COVID-QU-Ex/preprocess")
save_root_path = "/share-data/advanced_tech/datasets/COVID-QU-Ex/preprocess/v0.0"

image_file_path_list = natsorted(
    glob(os.path.join(dataset_dir, "*", "*", "images", "*.png"))
)
lesion_mask_file_path_list = natsorted(
    glob(os.path.join(dataset_dir, "*", "*", "infection masks", "*.png"))
)
lung_mask_file_path_list = natsorted(
    glob(os.path.join(dataset_dir, "*", "*", "lung masks", "*.png"))
)

input_train, input_eval, input_test = [], [], []

for img_file_path, lesion_mask_file_path, lung_mask_file_path in zip(
    image_file_path_list, tqdm(lesion_mask_file_path_list), lung_mask_file_path_list
):
    assert (
        os.path.basename(img_file_path)
        == os.path.basename(lesion_mask_file_path)
        == os.path.basename(lung_mask_file_path)
    )
    mode = lesion_mask_file_path.split("/")[-4]
    lesion = lesion_mask_file_path.split("/")[-3]
    lesion_mask_path = os.path.join(*lesion_mask_file_path.split("/")[4:])
    lung_mask_path = os.path.join(*lung_mask_file_path.split("/")[4:])
    img_path = os.path.join(*img_file_path.split("/")[4:])

    lesion_pil = Image.open(lesion_mask_file_path)
    lesion_mask = np.array(lesion_pil) // 255
    lung_pil = Image.open(lung_mask_file_path)
    lung_mask = np.array(lung_pil) // 255

    lesion_bbox = generate_bbox(lesion_mask)
    lung_bbox = generate_bbox(lung_mask)
    # test_visualize(lesion_pil,lesion_bbox)
    if mode == "Train":
        input_train.append(
            data_json(lesion, lesion_bbox, lung_bbox, lesion_mask_path, lung_mask_path)
        )
    elif mode == "Val":
        input_eval.append(
            data_json(lesion, lesion_bbox, lung_bbox, lesion_mask_path, lung_mask_path)
        )
    elif mode == "Test":
        input_test.append(
            data_json(lesion, lesion_bbox, lung_bbox, lesion_mask_path, lung_mask_path)
        )

with open(os.path.join(save_root_path, "input_train.json"), "w") as j:
    j.write(json.dumps(input_train, indent=2))

with open(os.path.join(save_root_path, "input_eval.json"), "w") as j:
    j.write(json.dumps(input_eval, indent=2))

with open(os.path.join(save_root_path, "input_test.json"), "w") as j:
    j.write(json.dumps(input_test, indent=2))
