import argparse
import os
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from common.utils import save_json
from external.chexbert.f1chexbert_hf import CheXbertLabeler

parser = argparse.ArgumentParser()
parser.add_argument(
    "--mimic_cxr_sectioned_path",
    default="datasets/MIMIC-CXR/preprocess/sections/mimic_cxr_sectioned.csv",
)
parser.add_argument(
    "--mimic_cxr_split_path",
    default="datasets/MIMIC-CXR/original_data_info/mimic-cxr-2.0.0-split.csv",
)
parser.add_argument(
    "--mimic_cxr_metadata_path",
    default="datasets/MIMIC-CXR/original_data_info/mimic-cxr-2.0.0-metadata.csv",
)
parser.add_argument(
    "--save_dir",
    default="datasets/MIMIC-CXR/preprocess/v9.2",
)

chexbert_labeler = CheXbertLabeler()


def normalize_time(time_str):
    if "." in time_str:
        seconds, milliseconds = time_str.split(".")
    else:
        seconds, milliseconds = time_str, "000000"

    seconds = seconds.zfill(6)
    milliseconds = milliseconds.ljust(6, "0")

    return f"{seconds}.{milliseconds}"


def studies_for_subject(metadata):
    subject_dict = defaultdict(list)

    for i in tqdm(range(len(metadata))):
        row = metadata.iloc[i]
        subject_id = str(row["subject_id"])
        study_date = row["StudyDate"]
        study_time = str(row["StudyTime"])
        study_id = str(row["study_id"])

        # time
        normalized_time = normalize_time(study_time)
        subject_dict[subject_id].append((study_id, study_date, normalized_time))

    # sort with data, time for studies
    subject_studies = {}
    for subject_id in subject_dict:
        unique_studies = {}
        for study_id, study_date, study_time in sorted(
            subject_dict[subject_id], key=lambda x: (x[1], x[2])
        ):
            if study_id not in unique_studies:
                unique_studies[study_id] = (study_date, study_time)
        subject_studies[subject_id] = list(unique_studies.keys())
    return subject_studies


def main(args):
    args = parser.parse_args(args)

    mimic_cxr_sectioned_path = Path(args.mimic_cxr_sectioned_path)
    mimic_cxr_split_path = Path(args.mimic_cxr_split_path)
    mimic_cxr_metadata_path = Path(args.mimic_cxr_metadata_path)
    save_dir = Path(args.save_dir)

    os.makedirs(save_dir, exist_ok=True)

    sectioned = pd.read_csv(mimic_cxr_sectioned_path)
    split = pd.read_csv(mimic_cxr_split_path)
    metadata = pd.read_csv(mimic_cxr_metadata_path)

    # study to split
    study_split = {}
    for i in tqdm(range(len(split))):
        study_split[str(split.iloc[i]["study_id"])] = split.iloc[i]["split"]

    # sectioned findings
    sectioned_findings = {}
    for i in tqdm(range(len(sectioned))):
        study_id = sectioned.iloc[i]["study"]
        findings = sectioned.iloc[i]["findings"]
        sectioned_findings[study_id.replace("s", "")] = findings

    # metadata to study
    study_dict = defaultdict(dict)
    for i in tqdm(range(len(metadata))):
        row = metadata.iloc[i]

        study_id = str(row["study_id"])
        if study_id not in study_dict:

            study_dict[study_id]["study_id"] = study_id
            study_dict[study_id]["subject_id"] = str(row["subject_id"])
            study_dict[study_id]["dicom_id"] = []
            study_dict[study_id]["view_position"] = []
            study_dict[study_id]["split"] = []

        study_dict[study_id]["dicom_id"].append(row["dicom_id"] + ".jpg")
        study_dict[study_id]["view_position"].append(row["ViewPosition"])
        study_dict[study_id]["split"].append(study_split[str(study_id)])

        if study_id in sectioned_findings:
            findings = sectioned_findings[study_id]
            if type(findings) is str and findings:
                study_dict[study_id]["findings"] = findings
            else:
                study_dict[study_id]["findings"] = ""
        else:
            study_dict[study_id]["findings"] = ""

    # check split
    for study_id, study in study_dict.items():
        split_set = set(study["split"])
        assert len(split_set) == 1
        study["split"] = split_set.pop()

    # findings to chexbert labels
    for study_id, study in tqdm(study_dict.items()):
        if study["findings"]:
            study["chexbert"] = chexbert_labeler.get_label(study["findings"])
        else:
            study["chexbert"] = None

    # subject history
    subject_studies = studies_for_subject(metadata)

    # save dicom level
    save_dir = os.path.join(args.save_dir, "dicom_level")
    os.makedirs(save_dir, exist_ok=True)

    dicom_level_data = defaultdict(list)
    for study_id, study in tqdm(study_dict.items()):
        split = study["split"]

        if study["findings"]:

            for dicom_id, view in zip(study["dicom_id"], study["view_position"]):
                dicom_level = {}
                dicom_level["study_id"] = study_id
                dicom_level["subject_id"] = study["subject_id"]
                dicom_level["dicom_id"] = dicom_id
                dicom_level["view_position"] = view
                dicom_level["split"] = study["split"]
                dicom_level["findings"] = study["findings"]
                dicom_level["chexbert"] = study["chexbert"]

                if dicom_level["findings"]:
                    dicom_level_data[split].append(dicom_level)

    for split in dicom_level_data.keys():
        save_json(dicom_level_data[split], os.path.join(save_dir, split + ".json"))

    # save study level
    save_dir = os.path.join(args.save_dir, "study_level")
    os.makedirs(save_dir, exist_ok=True)

    study_level_data = defaultdict(list)
    for study_id, study in tqdm(study_dict.items()):
        split = study["split"]
        if study["findings"]:
            study_level_data[split].append(study)

    for split in study_level_data.keys():
        save_json(study_level_data[split], os.path.join(save_dir, split + ".json"))

    # save subject level
    save_dir = os.path.join(args.save_dir, "subject_level")
    os.makedirs(save_dir, exist_ok=True)

    subject_level_data = defaultdict(list)
    for subject_id in tqdm(subject_studies.keys()):
        output = {}
        split = set()
        subject_history = []
        for study_id in subject_studies[subject_id]:
            study = study_dict[study_id]
            split.update(set([study["split"]]))
            subject_history.append(study)

        if len(subject_history) > 1:
            assert len(split) == 1
            split = split.pop()

            output["subject_id"] = subject_id
            output["study_history"] = subject_history
            subject_level_data[split].append(output)

    for split in subject_level_data.keys():
        save_json(subject_level_data[split], os.path.join(save_dir, split + ".json"))


if __name__ == "__main__":
    main(sys.argv[1:])
