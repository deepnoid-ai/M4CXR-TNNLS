import csv
import os
import shutil

import matplotlib.pyplot as plt
import numpy as np
import pydicom
from PIL import Image
from tqdm import tqdm

from common.utils import save_json


def intersection_area(bbox1, bbox2):
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])

    if x1 < x2 and y1 < y2:
        return (x2 - x1) * (y2 - y1)
    else:
        return 0


# convert dicom to png(only train)
def dcm_to_jpg_pydicom(dcm_dir, jpg_dir):
    if not os.path.exists(jpg_dir):
        os.makedirs(jpg_dir)
    for path, subdirs, files in os.walk(dcm_dir):
        # get all file in directory
        print(path)
        for file in tqdm(files):
            name, ext = os.path.splitext(file)
            if ext == ".dicom" or ext == ".dcm":
                dcm_path = os.path.join(path, file)
                # read dicom image
                dcm = pydicom.read_file(dcm_path)
                # get image array
                arr = dcm.pixel_array
                # tags add one space of each value.
                pi_tag = dcm.get(0x00280004).value
                if pi_tag == "MONOCHROME2":
                    arr = (arr - arr.min()) / (arr.max() - arr.min()) * 255.0
                else:
                    arr = (1 - ((arr - arr.min()) / (arr.max() - arr.min()))) * 255.0

                png_path = os.path.join(
                    jpg_dir, os.path.splitext(os.path.basename(dcm_path))[0] + ".jpg"
                )
                plt.imsave(
                    rf"{png_path}", arr.astype(np.uint8), cmap="gray", vmin=0, vmax=255
                )

            elif ext == ".jpg" or ext == ".png":
                img_path = os.path.join(path, file)
                png_path = os.path.join(
                    jpg_dir,
                    os.path.splitext(os.path.basename(img_path))[0].replace(
                        ext, ".jpg"
                    ),
                )
                shutil.copy(img_path, rf"{png_path}")


def bbox_area(bbox):
    return (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])


def iou(bbox1, bbox2):
    inter_area = intersection_area(bbox1, bbox2)
    union_area = bbox_area(bbox1) + bbox_area(bbox2) - inter_area
    return inter_area / union_area


def merge_bboxes(bbox1, bbox2):
    x1 = min(bbox1[0], bbox2[0])
    y1 = min(bbox1[1], bbox2[1])
    x2 = max(bbox1[2], bbox2[2])
    y2 = max(bbox1[3], bbox2[3])
    return [x1, y1, x2, y2]


def merge_overlapping_bboxes(bboxes, threshold=0.5):
    merged_bboxes = []
    while bboxes:
        bbox = bboxes.pop(0)
        to_merge = [bbox]
        i = 0
        while i < len(bboxes):
            if iou(bbox, bboxes[i]) > threshold:
                to_merge.append(bboxes.pop(i))
            else:
                i += 1
        merged_bbox = to_merge[0]
        for box in to_merge[1:]:
            merged_bbox = merge_bboxes(merged_bbox, box)
        merged_bboxes.append(merged_bbox)
    return merged_bboxes


def add_all_info(csv_path):
    add_all_info_dict = {}
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            image_id = row["image_id"]
            class_name = row["class_name"]
            bbox_info = [
                float(row["x_min"]) if row["x_min"] != "" else "",
                float(row["y_min"]) if row["y_min"] != "" else "",
                float(row["x_max"]) if row["x_max"] != "" else "",
                float(row["y_max"]) if row["y_max"] != "" else "",
            ]

            if image_id in add_all_info_dict:
                if class_name in add_all_info_dict[image_id]:
                    add_all_info_dict[image_id][class_name].append(bbox_info)
                else:
                    add_all_info_dict[image_id][class_name] = [bbox_info]
            else:
                add_all_info_dict[image_id] = {class_name: [bbox_info]}

    return add_all_info_dict


# step 3 => final output json
def vinbig_input(add_all_info_dict, save_path, label_list):
    final_output_json = []

    for k, v in tqdm(add_all_info_dict.items()):
        output_json_dict = {}

        output_json_dict["image"] = os.path.join("VinBigData", "images", k + ".jpg")

        cls_list = []
        for label in label_list:
            cls_dict = {}
            if label in v:
                cls_dict["name"] = label
                cls_dict["label"] = "positive"
            else:
                cls_dict["name"] = label
                cls_dict["label"] = "negative"

            cls_list.append(cls_dict)

        output_json_dict["cls"] = cls_list

        output_json_dict["det"] = []

        for i in output_json_dict["cls"]:
            name = i["name"]
            # negative case
            if i["label"] == "negative":
                output_json_dict["det"].append({"name": name, "label": [[0, 0, 0, 0]]})
            # positive case
            else:
                """
                output_json_dict["det"].append({"name": name, "label": v[name]})
                """

                # box
                bbox_list = v.get(name, [])

                # Save the merged boxes
                merged_boxes = merge_overlapping_bboxes(bbox_list, threshold=0.5)

                output_json_dict["det"].append({"name": name, "label": merged_boxes})

        output_json_dict["seg"] = []
        output_json_dict["text"] = ""

        assert len(output_json_dict["det"]) == len(output_json_dict["cls"])

        final_output_json.append(output_json_dict)

    os.makedirs(save_path, exist_ok=True)
    save_json(final_output_json, os.path.join(save_path, "vinbig_input.json"))
    print("saved json")


if __name__ == "__main__":
    version = "v1.0"
    data_root = "datasets"
    train_csv_path = os.path.join(data_root, "VinBigData", "train.csv")
    save_path = os.path.join(data_root, "VinBigData", "preprocess", version)

    # jpg_dir
    dcm_dir = "datasets/VinBigData/train"
    jpg_dir = os.path.join(data_root, "VinBigData/images")

    # convert dcm file to jpg
    dcm_to_jpg_pydicom(dcm_dir, jpg_dir)

    label_list = [
        "Aortic enlargement",
        "Atelectasis",
        "Calcification",
        "Cardiomegaly",
        "Consolidation",
        "ILD",
        "Infiltration",
        "Lung Opacity",
        "Nodule/Mass",
        "Other lesion",
        "Pleural effusion",
        "Pleural thickening",
        "Pneumothorax",
        "Pulmonary fibrosis",
    ]

    # step 1
    add_all_info_dict = add_all_info(train_csv_path)

    # step 2
    vinbig_input(add_all_info_dict, save_path, label_list)
