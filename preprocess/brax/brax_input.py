import csv
import os

from PIL import Image

from common.utils import save_json


def check_image_pixel_limit(image_path):
    with Image.open(image_path) as img:
        image_size = img.width * img.height

    max_pixels = Image.MAX_IMAGE_PIXELS

    if image_size > max_pixels:
        return False
    else:
        return True


def brax_input(data_root, middle_path, csv_path, finding_list, save_path):
    os.makedirs(save_path, exist_ok=True)

    dict_to_number = {
        "1.0": "positive",
        "0.0": "negative",
        "-1.0": "uncertain",
        "": "blank",
    }

    final_output_json = []
    with open(csv_path, mode="r", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            output_json_dict = {}
            output_json_dict["image"] = os.path.join(middle_path, row["PngPath"])

            image_path = os.path.join(data_root, output_json_dict["image"])
            assert os.path.exists(image_path)

            if check_image_pixel_limit(image_path):
                cls_list = []
                for key in row.keys():
                    if key in finding_list:
                        # ignore unknown
                        cls_dict = {"name": key, "label": dict_to_number[row[key]]}
                        cls_list.append(cls_dict)

                output_json_dict["cls"] = cls_list
                output_json_dict["det"] = []
                output_json_dict["seg"] = []
                output_json_dict["text"] = ""

                final_output_json.append(output_json_dict)
            else:
                print(image_path)

        save_json(final_output_json, os.path.join(save_path, "brax_input.json"))


if __name__ == "__main__":
    version = "v2.0"
    data_root = "datasets"

    middle_path = "physionet.org/files/brax/1.1.0"

    csv_path = os.path.join(data_root, middle_path, "master_spreadsheet_update.csv")

    finding_list = [
        "Enlarged Cardiomediastinum",
        "Cardiomegaly",
        "Lung Lesion",
        "Lung Opacity",
        "Edema",
        "Consolidation",
        "Pneumonia",
        "Atelectasis",
        "Pneumothorax",
        "Pleural Effusion",
        "Pleural Other",
        "Fracture",
        "Support Devices",
        "No Finding",
    ]

    save_path = os.path.join(data_root, middle_path, "preprocess", version)

    brax_input(data_root, middle_path, csv_path, finding_list, save_path)
