import os

from tqdm import tqdm

from external.chexbert.f1chexbert_hf import CLASS_MAPPING, CONDITIONS

from ..utils import clean_report_mimic_cxr, print_rank_0
from .base import BaseDataset
from .common import load_json_files, mscxr_duplicate


class MimiccxrMultiImageDataset(BaseDataset):
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
        self.max_image_per_data = kwargs.get("max_image_per_data")
        raw_data_lst = load_json_files(os.path.join(self.data_root, kwargs[self.split]))

        # remove mscxr from train dataset
        if "mscxr_duplicate" in kwargs and self.split == "train":
            mscxr_list = mscxr_duplicate(self.data_root, kwargs["mscxr_duplicate"])

            filtered_data_lst = []
            for i in raw_data_lst:
                dicom_list = i["dicom_id"]
                exclude_item = False
                for image in dicom_list:
                    image = os.path.join("MIMIC-CXR", "images", image)
                    if image in mscxr_list:
                        exclude_item = True
                        break
                if not exclude_item:
                    filtered_data_lst.append(i)

            raw_data_lst = filtered_data_lst

        self.dataset = self.load_data(raw_data_lst)

        print_rank_0(
            f"Load {self.__class__.__name__}, {self.task_type}, {self.split}. len: {len(self.dataset)}"
        )

    def load_data(self, raw_data):
        parsed_data = []

        for study in tqdm(raw_data):
            temp_dict = {
                "images": [
                    os.path.join(self.data_root, "MIMIC-CXR", "images", image)
                    for image in study["dicom_id"]
                ],
            }

            # skip too many images per data
            if len(temp_dict["images"]) > self.max_image_per_data:
                continue

            report = study["findings"]

            # text cleansing
            report = clean_report_mimic_cxr(report)

            temp_dict["report"] = report
            temp_dict["task_type"] = self.task_type

            if study["chexbert"]:
                cls_labels = []
                for i, index in enumerate(study["chexbert"]):
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

        if self.task_type == "frontal_lateral":
            template = self.select_template(self.task_type)
            images_count = " ".join("<image>" for _ in data["images"])

            prompts = [
                {"role": "system", "content": template[0]},
                {"role": "user", "content": template[1].format(image=images_count)},
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

        elif self.task_type == "frontal_lateral_cot":
            template = self.select_template(self.task_type)
            images_count = " ".join("<image>" for _ in data["images"])

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
                {"role": "user", "content": template[1].format(images=images_count)},
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
                    ),
                ]
            chats.append(
                (
                    {"role": "user", "content": template[5]},
                    {
                        "role": "assistant",
                        "content": template[6].format(report=data["report"]),
                    },
                ),
            )

        return {"prompts": prompts, "chats": chats}
