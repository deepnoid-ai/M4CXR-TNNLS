import json
import os
from glob import glob

import cv2
import numpy as np
from natsort import natsorted
from PIL import Image, ImageDraw
from tqdm import tqdm


def generate_bbox(lesion_mask, lesion_num_max=2):
    lesion_mask_contour_idxs, _ = cv2.findContours(
        lesion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    lesion_bboxs = []
    if len(lesion_mask_contour_idxs):
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


data_root_path = "datasets/QaTa-COV19/QaTa-COV19-v2"
save_root_path = os.path.join("datasets/QaTa-COV19/preprocess")

image_path_list = natsorted(
    glob(os.path.join(data_root_path, "*", "Images/sub*.png"))
)  # remove COVID-QU-Ex duplicates
mask_path_list = natsorted(
    glob(os.path.join(data_root_path, "*", "Ground-truths/mask_sub*.png"))
)  # remove COVID-QU-Ex duplicates

input_json = []
for image_path, mask_path in zip(tqdm(image_path_list), mask_path_list):
    assert os.path.basename(image_path) == os.path.basename(mask_path)[5:]

    lesion_pil = Image.open(mask_path)
    lesion_mask = np.array(lesion_pil) // 255
    lesion_bbox = generate_bbox(lesion_mask)

    # test_visualize(lesion_pil, lesion_bbox) # visualize
    input_dict = {
        "image": os.path.join(*image_path.split("/")[4:]),
        "cls": [{"name": "COVID-19", "label": "positive"}],
        "det": [{"name": "COVID-19", "label": lesion_bbox}],
        "seg": [{"name": "COVID-19", "label": os.path.join(*mask_path.split("/")[4:])}],
        "text": "",
    }
    input_json.append(input_dict)

with open(os.path.join(save_root_path, "input.json"), "w") as j:
    j.write(json.dumps(input_json, indent=2))

# TODO train, test split
image_train_path_list = natsorted(
    glob(os.path.join(data_root_path, "Train Set", "Images/sub*.png"))
)
image_test_path_list = natsorted(
    glob(os.path.join(data_root_path, "Test Set", "Images/sub*.png"))
)

train_list = [
    os.path.basename(image_train_path) for image_train_path in image_train_path_list
]
test_list = [
    os.path.basename(image_test_path) for image_test_path in image_test_path_list
]

train_input_json, test_input_json = [], []
for input_j in input_json:
    if os.path.basename(input_j["image"]) in train_list:
        train_input_json.append(input_j)
    elif os.path.basename(input_j["image"]) in test_list:
        test_input_json.append(input_j)

with open(os.path.join(save_root_path, "train_input.json"), "w") as j:
    j.write(json.dumps(train_input_json, indent=2))

with open(os.path.join(save_root_path, "test_input.json"), "w") as j:
    j.write(json.dumps(test_input_json, indent=2))
