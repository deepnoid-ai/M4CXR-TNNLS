import json
import os
import sys

import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm

from common.utils import save_json
from exp.cxr.utils import create_inverted_mask_with_boxes, find_bounding_boxes

LOCALIZATION_TASKS = [
    "Enlarged Cardiomediastinum",
    "Cardiomegaly",
    "Lung Lesion",
    "Airspace Opacity",
    "Edema",
    "Consolidation",
    "Atelectasis",
    "Pneumothorax",
    "Pleural Effusion",
    "Support Devices",
]


def create_mask(polygons, img_dims):
    """
    Creates a binary mask (of the original matrix size) given a list of polygon
        annotations format.

    Args:
        polygons (list): [[[x11,y11],[x12,y12],...[x1n,y1n]],...]

    Returns:
        mask (np.array): binary mask, 1 where the pixel is predicted to be the,
                                                 pathology, 0 otherwise
    """
    poly = Image.new("1", (img_dims[1], img_dims[0]))
    for polygon in polygons:
        coords = [(point[0], point[1]) for point in polygon]
        ImageDraw.Draw(poly).polygon(coords, outline=1, fill=1)

    binary_mask = np.array(poly, dtype="int")
    return binary_mask


def ann2mask_chexlocalize_input(
    data_root, exist_img_path, input_path, middle_path, output_path, save_path, split
):
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(output_path, exist_ok=True)

    print(f"Reading annotations from {input_path}...")
    with open(input_path) as f:
        ann = json.load(f)

    final_output_json = []

    for img_id in tqdm(ann.keys()):
        output_json_dict = {}

        cls_list = []
        seg_list = []
        bbox_list = []
        img_id2img_path = (
            img_id.rsplit("_", 1)[0].replace("_", "/")
            + "_"
            + img_id.rsplit("_", 1)[1]
            + ".jpg"
        )
        output_json_dict["image"] = os.path.join(exist_img_path, img_id2img_path)

        assert os.path.exists(os.path.join(data_root, output_json_dict["image"]))

        for task in LOCALIZATION_TASKS:
            # create segmentation
            polygons = ann[img_id][task] if task in ann[img_id] else []

            # positive case
            if len(polygons) > 0:

                cls_dict = {"name": task, "label": "positive"}
                cls_list.append(cls_dict)

                img_dims = ann[img_id]["img_size"]
                mask = create_mask(polygons, img_dims)
                mask_image = Image.fromarray(mask.astype("uint8") * 255)

                task = task.replace(" ", "_")
                mask_name = f"{img_id}_{task}.png"
                mask_save_path = os.path.join(output_path, mask_name)

                seg_dict = {
                    "name": task,
                    "label": os.path.join(middle_path, mask_save_path).replace(
                        data_root + "/", ""
                    ),
                }
                seg_list.append(seg_dict)

                mask_image.save(mask_save_path)
                # bbox info
                bbox_info = find_bounding_boxes(mask_save_path)

                fix_bbox_info = []
                for i in range(len(bbox_info)):
                    final_bbox_info = [
                        bbox_info[i][0],
                        bbox_info[i][1],
                        bbox_info[i][0] + bbox_info[i][2],
                        bbox_info[i][1] + bbox_info[i][3],
                    ]
                    fix_bbox_info.append(final_bbox_info)

                bbox_dict = {"name": task, "label": fix_bbox_info}
                bbox_list.append(bbox_dict)

            # negative case
            else:
                cls_dict = {"name": task, "label": "negative"}
                cls_list.append(cls_dict)

                bbox_dict = {"name": task, "label": [[0, 0, 0, 0]]}
                bbox_list.append(bbox_dict)

                seg_dict = {"name": task, "label": "negative"}
                seg_list.append(seg_dict)

        output_json_dict["cls"] = cls_list
        output_json_dict["det"] = bbox_list
        output_json_dict["seg"] = seg_list
        output_json_dict["text"] = ""

        final_output_json.append(output_json_dict)

    save_json(
        final_output_json, os.path.join(save_path, f"chexlocalize_{split}_input.json")
    )


if __name__ == "__main__":
    version = "v0.0"
    data_root = "datasets"
    middle_path = "chexlocalize/CheXlocalize"

    split = "test"

    anno_json_path = os.path.join(
        data_root, middle_path, f"gt_annotations_{split}.json"
    )
    mask_output_path = os.path.join(data_root, middle_path, "masks", split)
    exist_img_path = os.path.join("chexlocalize", "CheXpert", split)

    save_path = os.path.join(data_root, "chexlocalize", "preprocess", version)

    ann2mask_chexlocalize_input(
        data_root,
        exist_img_path,
        anno_json_path,
        middle_path,
        mask_output_path,
        save_path,
        split,
    )
