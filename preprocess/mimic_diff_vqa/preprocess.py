import os

import pandas as pd
from tqdm import tqdm

from common.utils import save_json

data_root = "datasets"
mimic_all = "medical-diff-vqa-a-large-scale-medical-dataset-for-difference-visual-question-answering-on-chest-x-ray-images-1.0.0/mimic_all.csv"
mimic_pair_question = "medical-diff-vqa-a-large-scale-medical-dataset-for-difference-visual-question-answering-on-chest-x-ray-images-1.0.0/mimic_pair_questions.csv"
splits = ["train", "eval", "test"]
question_types = ["difference", "vqa"]
version = "v1.0"

mimic_all = pd.read_csv(os.path.join(data_root, mimic_all))
mimic_pair_questions = pd.read_csv(os.path.join(data_root, mimic_pair_question))

# study id to dicom id (image)
study_id_to_dicom_id = {}
for index, row in mimic_all.iterrows():
    assert row["study_id"] not in study_id_to_dicom_id
    study_id_to_dicom_id[str(row["study_id"])] = str(row["dicom_id"])

for split in splits:

    # split data
    data_split = split
    if split == "eval":
        data_split = "val"
    split_data = mimic_pair_questions[mimic_pair_questions["split"] == data_split]

    parsed_data = []
    # difference question type or vqa
    for question_type in question_types:
        if question_type == "difference":
            tmp_data = split_data[split_data["question_type"] == "difference"]
        else:
            tmp_data = split_data[split_data["question_type"] != "difference"]

        for index, row in tqdm(tmp_data.iterrows(), total=tmp_data.shape[0]):
            reference_image = os.path.join(
                "MIMIC-CXR",
                "images",
                study_id_to_dicom_id[str(row["ref_id"])] + ".jpg",
            )

            main_image = os.path.join(
                "MIMIC-CXR",
                "images",
                study_id_to_dicom_id[str(row["study_id"])] + ".jpg",
            )

            temp_dict = {}
            temp_dict["reference_image"] = reference_image
            temp_dict["main_image"] = main_image
            temp_dict["question_type"] = row["question_type"]
            temp_dict["question"] = row["question"]
            temp_dict["answer"] = row["answer"]

            parsed_data.append(temp_dict)

        # save files
        save_path = os.path.join(
            data_root,
            "medical-diff-vqa-a-large-scale-medical-dataset-for-difference-visual-question-answering-on-chest-x-ray-images-1.0.0/preprocess",
            version,
            f"{question_type}_{split}.json",
        )
        save_json(parsed_data, save_path)
