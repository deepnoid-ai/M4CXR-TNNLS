import json
import os
from glob import glob

import cv2
import numpy as np
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
            if w <= 2 or h <= 2:
                pass
            else:
                lesion_bboxs.append([x1, y1, x1 + w, y1 + h])
    return lesion_bboxs


def test_visualize(lesion_pil, lesion_bboxs):
    for bbox in lesion_bboxs:
        minr, minc, maxr, maxc = bbox

        lesion_pil = lesion_pil.convert("RGB")
        draw = ImageDraw.Draw(lesion_pil)
        draw.rectangle((minr, minc, maxr, maxc), outline=(255, 0, 0), width=3)

        lesion_pil.save("./test.png")


data_root_path = "datasets/chestX-Det10"

image_path_list = glob(os.path.join(data_root_path, "images/*.jpg"))
mask_root_path = os.path.join(data_root_path, "masks")
mask_path_list = glob(os.path.join(mask_root_path, "*"))
mask_jpg_list = glob(os.path.join(mask_root_path, "*", "*.jpg"))
mask_path_id_list = [mask_path.split("/")[-1] for mask_path in mask_path_list]
lesion_list = np.unique(
    [os.path.basename(mask_path)[:-4].lower() for mask_path in mask_jpg_list]
)

input_json = []
for image_path in tqdm(image_path_list):
    image_path_id = os.path.basename(image_path)[:-4]

    input_dict = {}
    if image_path_id in mask_path_id_list:  # abnormal
        matched_mask_path_list = glob(
            os.path.join(mask_root_path, image_path_id, "*.jpg")
        )

        lesion_cls, lesion_det, lesion_seg = [], [], []
        for matched_mask_path in matched_mask_path_list:
            lesion_pil = Image.open(matched_mask_path)
            lesion_mask = np.where(np.array(lesion_pil) >= 128, 1, 0).astype(np.uint8)
            lesion_mask = cv2.cvtColor(lesion_mask, cv2.COLOR_BGR2GRAY)
            lesion_bbox = generate_bbox(lesion_mask)
            # test_visualize(lesion_pil, lesion_bbox) # for visualize
            matched_lesion_name = os.path.basename(matched_mask_path)[:-4].lower()

            lesion_cls.append({"name": matched_lesion_name, "label": "positive"})
            lesion_det.append({"name": matched_lesion_name, "label": lesion_bbox})
            lesion_seg.append(
                {
                    "name": matched_lesion_name,
                    "label": os.path.join(*matched_mask_path.split("/")[4:]),
                }
            )

        input_dict = {
            "image": os.path.join(*image_path.split("/")[4:]),
            "cls": lesion_cls,
            "det": lesion_det,
            "seg": lesion_seg,
            "text": "",
        }

    else:  # normal
        lesion_cls, lesion_det, lesion_seg = [], [], []
        for lesion in lesion_list:
            lesion_cls.append({"name": lesion, "label": "negative"})
            lesion_det.append({"name": lesion, "label": [[0, 0, 0, 0]]})
            lesion_seg.append({"name": lesion, "label": "negative"})

        input_dict = {
            "image": os.path.join(*image_path.split("/")[4:]),
            "cls": lesion_cls,
            "det": lesion_det,
            "seg": lesion_seg,
            "text": "",
        }

    input_json.append(input_dict)

save_root_path = os.path.join(data_root_path, "preprocess/v0.0")
with open(os.path.join(save_root_path, "input.json"), "w") as j:
    j.write(json.dumps(input_json, indent=2))
