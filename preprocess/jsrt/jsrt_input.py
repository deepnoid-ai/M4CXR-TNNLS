import csv
import os

import cv2
import numpy as np
from tqdm import tqdm

from common.utils import save_json


def jsrt_input(data_root, csv_path, mask_save_path, save_path):

    os.makedirs(save_path, exist_ok=True)
    os.makedirs(mask_save_path, exist_ok=True)

    final_output_json = []

    # mask
    with open(csv_path) as csv_file:
        reader = csv.DictReader(csv_file)

        for row in reader:
            output_json_dict = {}

            output_json_dict["image"] = os.path.join(
                "jsrt", "images", "images", row["study_id"]
            )

            image = cv2.imread(os.path.join(data_root, output_json_dict["image"]))
            height, width = image.shape[:2]

            # abnormal case
            if row["state"] != "non-nodule":
                # mask mask
                x = int(row["x"])
                y = int(row["y"])
                size = int(row["size"])

                image = np.zeros((height, width, 1), dtype=np.uint8)

                cv2.circle(
                    image,
                    (x, y),
                    radius=size,
                    color=(255,),
                    thickness=-1,
                )

                full_mask_save_path = os.path.join(mask_save_path, row["study_id"])

                cv2.imwrite(full_mask_save_path, image)

                # bbox_info
                bbox = [x - size, y - size, x + size, y + size]

                # dict mask_path
                dict_mask_path = full_mask_save_path.replace(data_root + "/", "")

                output_json_dict["cls"] = [{"name": "nodule", "label": "positive"}]
                output_json_dict["det"] = [{"name": "nodule", "label": [bbox]}]

                output_json_dict["seg"] = [{"name": "nodule", "label": dict_mask_path}]
                output_json_dict["text"] = ""

                final_output_json.append(output_json_dict)

            # normal case
            else:
                output_json_dict["cls"] = [{"name": "nodule", "label": "negative"}]
                output_json_dict["det"] = [{"name": "nodule", "label": [[0, 0, 0, 0]]}]

                output_json_dict["seg"] = [{"name": "nodule", "label": "negative"}]
                output_json_dict["text"] = ""

                final_output_json.append(output_json_dict)

    save_json(final_output_json, os.path.join(save_path, "jsrt_input.json"))


if __name__ == "__main__":
    version = "v0.0"
    data_root = "datasets"

    csv_path = os.path.join(data_root, "jsrt", "jsrt_metadata.csv")
    img_path = os.path.join(data_root, "jsrt", "images")
    mask_save_path = os.path.join(data_root, "jsrt", "masks")
    save_path = os.path.join(data_root, "jsrt", "preprocess", version)

    jsrt_input(data_root, csv_path, mask_save_path, save_path)
