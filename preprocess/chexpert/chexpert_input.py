import csv
import json
import os

from tqdm import tqdm

from common.utils import save_json


def make_final_output(csv_path, save_path, label_list, split):

    os.makedirs(save_path, exist_ok=True)

    pos_number = {
        "1.0": "positive",
        "0.0": "negative",
        "-1.0": "uncertain",
        "": "blank",
    }

    # step 1
    # remove columns
    remove_list = ["Sex", "Age", "Frontal/Lateral", "AP/PA"]

    # 다 ''인 경우 개수
    count = 0

    filtered_info_list = []
    with open(csv_path, "r") as f:
        csv_reader = csv.DictReader(f)
        for row in csv_reader:
            keys_to_remove = [key for key in row.keys() if key in remove_list]

            for key in keys_to_remove:
                del row[key]

            filtered_info_list.append(row)

    # step 3
    # make final output file
    final_output = []
    for row in tqdm(filtered_info_list):
        final_output_dict = {}
        cls = []
        for key, value in row.items():
            if key == "Path":
                final_output_dict["image"] = row["Path"]

            if key in label_list:
                cls_dict = {}
                cls_dict["name"] = key
                cls_dict["label"] = pos_number[value]
                cls.append(cls_dict)

        assert len(cls) == 14
        if cls:
            final_output_dict["cls"] = cls

        else:
            count += 1
            continue

        final_output_dict["det"] = []
        final_output_dict["seg"] = []
        final_output_dict["text"] = ""

        final_output.append(final_output_dict)

    print(count)
    save_json(final_output, os.path.join(save_path, f"chexpert_{split}_input.json"))


if __name__ == "__main__":
    version = "v1.0"
    data_root = "datasets"

    split = ["train", "valid"]

    label_list = [
        "No Finding",
        "Enlarged Cardiomediastinum",
        "Cardiomegaly",
        "Lung Opacity",
        "Lung Lesion",
        "Edema",
        "Consolidation",
        "Pneumonia",
        "Atelectasis",
        "Pneumothorax",
        "Pleural Effusion",
        "Pleural Other",
        "Fracture",
        "Support Devices",
    ]

    save_path = os.path.join(data_root, "CheXpert-v1.0", "preprocess", f"{version}")

    for i in split:
        chexpert_csv_path = os.path.join(data_root, "CheXpert-v1.0", f"{i}.csv")
        make_final_output(chexpert_csv_path, save_path, label_list, i)
