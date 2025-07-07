import os

from tqdm import tqdm

from ..utils import clean_report_mimic_cxr, print_rank_0
from .base import BaseDataset
from .common import load_json_files, mscxr_duplicate


class DiseaseClassificationDataset(BaseDataset):
    def __init__(
        self,
        tokenizer,
        processors,
        max_length,
        data_root: str,
        **kwargs,
    ):
        super().__init__(tokenizer, processors, max_length, data_root, **kwargs)

        self.task_type = kwargs["task_type"]  # single_disease, multi_disease
        raw_data_lst = load_json_files(os.path.join(self.data_root, kwargs[self.split]))

        # remove mscxr from train dataset
        if "mscxr_duplicate" in kwargs and self.split == "train":
            mscxr_list = mscxr_duplicate(self.data_root, kwargs["mscxr_duplicate"])
            raw_data_lst = [i for i in raw_data_lst if i["image"] not in mscxr_list]

        self.dataset = self.load_data(raw_data_lst)

        print_rank_0(
            f"Load {self.__class__.__name__}, {self.task_type}, {self.split}. len: {len(self.dataset)}"
        )

    def load_data(self, raw_data):
        parsed_data = []
        for item in tqdm(raw_data):
            temp_dict = {
                "images": [os.path.join(self.data_root, item["image"])],
            }

            if self.task_type == "cot":
                # report text
                report = item["text"]
                report = clean_report_mimic_cxr(report)
                temp_dict["report"] = report

            cls_dicts = []
            for cls_dict in item["cls"]:
                if cls_dict["name"] != "lung":  # except lung
                    cls_dicts.append(cls_dict)
            temp_dict["cls"] = cls_dicts
            if temp_dict["cls"] == []:
                continue
            temp_dict["task_type"] = self.task_type

            parsed_data.append(temp_dict)

        return parsed_data

    def build_prompts_and_chats(self, data, images):
        if self.task_type == "single_disease":
            template = self.select_template(self.task_type)

            no_finding_template = self.select_template("single_disease_no_finding")

            prompts = [
                {"role": "system", "content": template[0]},
                {"role": "user", "content": template[1]},
            ]

            chats = []
            for cls_data in data["cls"]:
                finding = cls_data["name"].lower()
                label = cls_data["label"]

                if finding != "no finding":

                    if label == "positive":
                        answer = template[3]
                    elif label in ["negative", "blank"]:
                        answer = template[4]
                    elif label == "uncertain":
                        answer = template[5]

                    qa_pair = (
                        {
                            "role": "user",
                            "content": template[2].format(finding=finding),
                        },
                        {
                            "role": "assistant",
                            "content": answer,
                        },
                    )
                else:
                    # no finding

                    if label == "positive":
                        # No.
                        answer = template[4]
                    elif label == "negative":
                        # Yes.
                        answer = template[3]
                    elif label == "uncertain":
                        # Uncertain.
                        answer = template[5]

                    # skip no finding for blank
                    elif label == "blank":
                        continue

                    qa_pair = (
                        {"role": "user", "content": no_finding_template[2]},
                        {
                            "role": "assistant",
                            "content": answer,
                        },
                    )

                chats.append(qa_pair)

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
