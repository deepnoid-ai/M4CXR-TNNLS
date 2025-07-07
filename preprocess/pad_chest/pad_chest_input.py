# image파일들을 images라는 폴더로 옮긴다는 가정 후 시작
import ast
import csv
import os
import shutil
from datetime import datetime

import cv2
from PIL import Image
from tqdm import tqdm

from common.multi_processing import func_with_multiprocessing
from common.utils import save_json


def convert2jpg(image_path):
    image_base_name = image_path.rsplit("/", 1)[1].split(".")[0]

    try:
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image is not None:
            height, width, channels = image.shape
            resized_image = cv2.resize(image, (width, height))
            jpg_path = os.path.join(
                image_path.rsplit("/", 1)[0].replace("images", "converted_images"),
                image_base_name + ".jpg",
            )

            cv2.imwrite(jpg_path, resized_image)

            image = Image.open(jpg_path)
    except IOError:
        os.remove(jpg_path)


def make_convert2jpg_list(images_path, converted_image_path):
    os.makedirs(converted_image_path, exist_ok=True)

    all_images_path = []
    for i in os.listdir(images_path):
        all_images_path.append(os.path.join(images_path, i))
    return all_images_path


def move_png_images(data_root):
    os.makedirs(os.path.join(data_root, "BIMCV-PadChest-FULL", "images"), exist_ok=True)
    destination_path = os.path.join(data_root, "BIMCV-PadChest-FULL", "images")

    find_png_path = os.path.join(data_root, "BIMCV-PadChest-FULL")

    png_list = []

    for i in tqdm(os.listdir(find_png_path)):
        if i.endswith(".png"):
            png_list.append(os.path.join(find_png_path, i))

    for file_path in png_list:
        shutil.move(file_path, destination_path)


def parse_date(date_str):
    return datetime.strptime(date_str, "%Y%m%d").date()


def pad_input(csv_path, save_path):

    os.makedirs(save_path, exist_ok=True)

    final_output_json = []

    unchanged_count = 0
    normal_count = 0
    learning_count = 0

    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            output_json_dict = {}

            # replace png => jpg
            image_base_name = row["ImageID"].replace(".png", ".jpg")

            if "nan" not in row["Labels"]:
                label = ast.literal_eval(row["Labels"])

            else:
                continue

            # study_id = row["StudyID"]
            # date = parse_date(row["StudyDate_DICOM"])
            # patient_id = row["PatientID"]
            # projection = row["Projection"]

            if "Exclude" in label or "Suboptimal" in label:
                continue

            else:
                output_json_dict["image"] = os.path.join(
                    "BIMCV-PadChest-FULL", "converted_images", image_base_name
                )

                image_path = os.path.join(data_root, output_json_dict["image"])
                if os.path.exists(image_path):

                    cls_list = []

                    if "normal" in label:
                        normal_count += 1
                        continue

                    elif "unchanged" in label:
                        unchanged_count += 1
                        continue

                    else:
                        for i in label:
                            cls_dict = {"name": i, "label": "positive"}
                            cls_list.append(cls_dict)

                        output_json_dict["cls"] = cls_list

                        output_json_dict["det"] = []
                        output_json_dict["seg"] = []
                        output_json_dict["text"] = ""

                        final_output_json.append(output_json_dict)
                        learning_count += 1
                        continue
                else:
                    continue
    print(len(final_output_json))
    save_json(final_output_json, os.path.join(save_path, "padchest_input.json"))


if __name__ == "__main__":
    version = "v1.0"
    data_root = "datasets"
    csv_path = os.path.join(
        data_root,
        "BIMCV-PadChest-FULL",
        "PADCHEST_chest_x_ray_images_labels_160K_01.02.19.csv",
    )
    save_path = os.path.join(data_root, "BIMCV-PadChest-FULL", "preprocess", version)

    images_path = os.path.join(data_root, "BIMCV-PadChest-FULL/images")

    converted_image_path = os.path.join(
        data_root, "BIMCV-PadChest-FULL", "converted_images"
    )

    # move_png_images(data_root)
    # input_list = make_convert2jpg_list(images_path, converted_image_path)
    # func_with_multiprocessing(convert2jpg, input_list, 16)

    pad_input(csv_path, save_path)
