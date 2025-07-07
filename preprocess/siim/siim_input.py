import csv
import glob
import json
import os

import cv2
import numpy as np
import pydicom
from tqdm import tqdm

from common.utils import save_json
from exp.cxr.utils import create_inverted_mask_with_boxes, find_bounding_boxes


def convert_dcm_make_json(
    dcm_path,
    csv_path,
    jpg_save_path,
    mask_save_path,
    data_root,
    save_path,
):

    os.makedirs(mask_save_path, exist_ok=True)
    os.makedirs(jpg_save_path, exist_ok=True)
    os.makedirs(save_path, exist_ok=True)

    dcm_files = []
    for root, dirs, files in os.walk(dcm_path):
        for file in glob.glob(os.path.join(root, "*.dcm")):
            dcm_files.append(file)

    mask_info = {}

    csv_info = {}
    with open(csv_path, "r") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            csv_info[row["ImageId"]] = row[" EncodedPixels"]

    for dcm in dcm_files:
        only_file_name = dcm.rsplit("/", 1)[1].split(".dcm")[0]
        if only_file_name in csv_info:
            mask_info[dcm] = csv_info[only_file_name]

    final_output_json = []
    for file_name, number in tqdm(mask_info.items()):
        output_json_dict = {}
        ds = pydicom.dcmread(file_name)
        img_array = ds.pixel_array
        only_file_name = file_name.rsplit("/", 1)[1].split(".dcm")[0]

        jpg_full_save_path = os.path.join(jpg_save_path, only_file_name + ".jpg")
        cv2.imwrite(jpg_full_save_path, img_array)

        json_img_name = jpg_full_save_path.replace(data_root + "/", "")

        if number != " -1":
            rle_data = number
            rle_array = rle2mask(rle_data, 1024, 1024).T
            mask_array = np.expand_dims(rle_array, axis=2)
            mask_full_save_path = os.path.join(mask_save_path, only_file_name + ".png")
            cv2.imwrite(mask_full_save_path, mask_array * 255)

            # bbox info
            bbox_info = find_bounding_boxes(mask_full_save_path)

            fix_bbox_info = []
            for i in range(len(bbox_info)):
                final_bbox_info = [
                    bbox_info[i][0],
                    bbox_info[i][1],
                    bbox_info[i][0] + bbox_info[i][2],
                    bbox_info[i][1] + bbox_info[i][3],
                ]
                fix_bbox_info.append(final_bbox_info)

            json_seg_path = mask_full_save_path.replace(data_root + "/", "")

            # json에 info input
            output_json_dict["image"] = json_img_name
            output_json_dict["cls"] = [{"name": "pneumothorax", "label": "positive"}]
            output_json_dict["det"] = [{"name": "pneumothorax", "label": fix_bbox_info}]
            output_json_dict["seg"] = [{"name": "pneumothorax", "label": json_seg_path}]
            output_json_dict["text"] = ""

            final_output_json.append(output_json_dict)

        else:
            output_json_dict["image"] = json_img_name
            output_json_dict["cls"] = [{"name": "pneumothorax", "label": "negative"}]
            output_json_dict["det"] = [
                {"name": "pneumothorax", "label": [[0, 0, 0, 0]]}
            ]
            output_json_dict["seg"] = [{"name": "pneumothorax", "label": "negative"}]
            output_json_dict["text"] = ""

            final_output_json.append(output_json_dict)

    assert len(final_output_json) == len(csv_info)

    save_json(final_output_json, os.path.join(save_path, "siim_input.json"))


def rle2mask(rle, width, height):
    mask = np.zeros(width * height)
    array = np.asarray([int(x) for x in rle.split()])
    starts = array[0::2]
    lengths = array[1::2]

    current_position = 0
    for index, start in enumerate(starts):
        current_position += start
        mask[current_position : current_position + lengths[index]] = 255
        current_position += lengths[index]

    return mask.reshape(width, height)


if __name__ == "__main__":
    version = "v0.0"
    data_root = "datasets"

    # test는 label이 없음, train dcm만
    dcm_path = os.path.join(data_root, "SIIM", "dicom-images-train")

    # train_csv_path
    csv_path = os.path.join(data_root, "SIIM", "train-rle.csv")

    mask_save_path = os.path.join(data_root, "SIIM", "masks")
    jpg_save_path = os.path.join(data_root, "SIIM", "images")

    save_path = os.path.join(data_root, "SIIM", "preprocess", version)

    """
    Images without pneumothorax have a mask value of -1

    """

    convert_dcm_make_json(
        dcm_path,
        csv_path,
        jpg_save_path,
        mask_save_path,
        data_root,
        save_path,
    )
