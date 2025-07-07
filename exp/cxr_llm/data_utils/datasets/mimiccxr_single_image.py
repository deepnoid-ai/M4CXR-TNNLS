import os

from tqdm import tqdm

from external.chexbert.f1chexbert_hf import CLASS_MAPPING, CONDITIONS

from ..utils import clean_report_mimic_cxr, print_rank_0
from .base import BaseDataset
from .common import load_json_files, mscxr_duplicate


class MimiccxrSingleImageDataset(BaseDataset):
    def __init__(
        self,
        tokenizer,
        processors,
        max_length,
        data_root: str,
        **kwargs,
    ):
        super().__init__(tokenizer, processors, max_length, data_root, **kwargs)

        self.task_type = kwargs["task_type"]

        raw_data_lst = load_json_files(os.path.join(self.data_root, kwargs[self.split]))

        # remove mscxr from train dataset
        if "mscxr_duplicate" in kwargs and self.split == "train":
            mscxr_list = mscxr_duplicate(self.data_root, kwargs["mscxr_duplicate"])
            filtered_data_lst = []
            for i in raw_data_lst:
                dicom_id = os.path.join("MIMIC-CXR", "images", i["dicom_id"])

                if dicom_id in mscxr_list:
                    continue

                filtered_data_lst.append(i)

            raw_data_lst = filtered_data_lst

        self.dataset = self.load_data(raw_data_lst)

        print_rank_0(
            f"Load {self.__class__.__name__}, {self.task_type}, {self.split}. len: {len(self.dataset)}"
        )

    def load_data(self, raw_data):
        parsed_data = []

        for dicom in tqdm(raw_data):

            temp_dict = {
                "images": [
                    os.path.join(
                        self.data_root, "MIMIC-CXR", "images", dicom["dicom_id"]
                    )
                ],
            }
            report = dicom["findings"]

            # text cleansing
            report = clean_report_mimic_cxr(report)

            temp_dict["report"] = report
            temp_dict["task_type"] = self.task_type

            if dicom["chexbert"]:
                cls_labels = []
                for i, index in enumerate(dicom["chexbert"]):
                    cls_labels.append(
                        {
                            "name": CONDITIONS[i],
                            "label": CLASS_MAPPING[index].lower(),
                        }
                    )
                temp_dict["cls"] = cls_labels

                if temp_dict["report"] and len(temp_dict["report"]) > 4:
                    parsed_data.append(temp_dict)

        return parsed_data

    def build_prompts_and_chats(self, data, images):

        if self.task_type == "report_only":
            # pure image, report pair dataset
            template = self.select_template("report_only")
            prompts = []
            chats = [
                (
                    {"role": "user", "content": template[1]},
                    {
                        "role": "assistant",
                        "content": template[3].format(report=data["report"]),
                    },
                )
            ]
        elif self.task_type == "mrg":
            # report generation instruction
            template = self.select_template("mrg")

            prompts = [
                {"role": "system", "content": template[0]},
                {"role": "user", "content": template[1]},
            ]
            chats = [
                (
                    {"role": "user", "content": template[2]},
                    {
                        "role": "assistant",
                        "content": template[3].format(report=data["report"]),
                    },
                )
            ]

        elif self.task_type == "cot":
            multi_disease_template = self.select_template("multi_disease")
            cot_template = self.select_template("cot")

            findings = []
            positives = []
            for i in data["cls"]:
                name = i["name"].lower()
                if name != "no finding":
                    findings.append(name)
                    if i["label"] == "positive":
                        positives.append(name)

            findings = ", ".join(findings)
            positive_findings = ", ".join(positives)

            prompts = [
                {"role": "system", "content": multi_disease_template[0]},
                {"role": "user", "content": multi_disease_template[1]},
            ]

            if positives == []:
                # negative
                chats = [
                    (
                        {
                            "role": "user",
                            "content": multi_disease_template[2].format(
                                findings=findings
                            ),
                        },
                        {
                            "role": "assistant",
                            "content": multi_disease_template[4],
                        },
                    )
                ]
            else:
                # positive
                chats = [
                    (
                        {
                            "role": "user",
                            "content": multi_disease_template[2].format(
                                findings=findings
                            ),
                        },
                        {
                            "role": "assistant",
                            "content": multi_disease_template[3].format(
                                positive_findings=positive_findings
                            ),
                        },
                    ),
                ]
            chats.append(
                (
                    {"role": "user", "content": cot_template[2]},
                    {
                        "role": "assistant",
                        "content": cot_template[3].format(report=data["report"]),
                    },
                ),
            )

        elif self.task_type == "multi_disease":
            template = self.select_template(self.task_type)

            findings = []
            positives = []
            for i in data["cls"]:
                name = i["name"].lower()
                if name != "no finding":
                    findings.append(name)
                    if i["label"] == "positive":
                        positives.append(name)

            findings = ", ".join(findings)
            positive_findings = ", ".join(positives)

            prompts = [
                {"role": "system", "content": template[0]},
                {"role": "user", "content": template[1]},
            ]

            if positives == []:
                # negative
                chats = [
                    (
                        {
                            "role": "user",
                            "content": template[2].format(findings=findings),
                        },
                        {
                            "role": "assistant",
                            "content": template[4],
                        },
                    )
                ]
            else:
                # positive
                chats = [
                    (
                        {
                            "role": "user",
                            "content": template[2].format(findings=findings),
                        },
                        {
                            "role": "assistant",
                            "content": template[3].format(
                                positive_findings=positive_findings
                            ),
                        },
                    )
                ]
        return {"prompts": prompts, "chats": chats}
