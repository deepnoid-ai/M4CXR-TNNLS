import os

from tqdm import tqdm

from external.chexbert.f1chexbert_hf import CLASS_MAPPING, CONDITIONS

from ..utils import clean_report_mimic_cxr, print_rank_0
from .base import BaseDataset
from .common import load_json_files, mscxr_duplicate


class MimiccxrMultiStudyDataset(BaseDataset):
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
        self.history_legnth = kwargs.get("history_legnth")
        raw_data_lst = load_json_files(os.path.join(self.data_root, kwargs[self.split]))

        # remove mscxr from train dataset
        if "mscxr_duplicate" in kwargs and self.split == "train":
            mscxr_list = mscxr_duplicate(self.data_root, kwargs["mscxr_duplicate"])

            filtered_data_lst = []
            for i in raw_data_lst:
                exclude_item = False
                for study_history in i["study_history"]:
                    dicom_list = study_history["dicom_id"]
                    for image in dicom_list:
                        image = os.path.join("MIMIC-CXR", "images", image)
                        if image in mscxr_list:
                            exclude_item = True
                            break
                    if exclude_item:
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

        def overlapping_chunks(input_list, chunk_size):
            result = []
            for i in range(len(input_list) - chunk_size + 1):
                result.append(input_list[i : i + chunk_size])

            return result

        for data in tqdm(raw_data):
            history_chunks = overlapping_chunks(
                data["study_history"], self.history_legnth
            )

            for history_chunk in history_chunks:
                images = []
                images_history = []
                reports = []
                assert len(history_chunk) >= 2

                for study in history_chunk:
                    study_image = [
                        os.path.join(self.data_root, "MIMIC-CXR", "images", image)
                        for image in study["dicom_id"]
                    ]

                    report = study["findings"]

                    # text cleansing
                    report = clean_report_mimic_cxr(report)

                    if report and len(report) > 4:
                        last_study = study
                        reports.append(report)
                        images += study_image
                        images_history.append(study_image)

                if len(images_history) < 2:
                    continue

                if last_study["chexbert"]:

                    temp_dict = {}

                    cls_labels = []
                    for i, index in enumerate(last_study["chexbert"]):
                        cls_labels.append(
                            {
                                "name": CONDITIONS[i],
                                "label": CLASS_MAPPING[index].lower(),
                            }
                        )
                    temp_dict["cls"] = cls_labels

                    temp_dict["images_history"] = images_history
                    temp_dict["images"] = images
                    temp_dict["reports"] = reports
                    temp_dict["task_type"] = self.task_type

                    # skip too many images per data
                    if len(temp_dict["images"]) > self.max_image_per_data:
                        continue

                    parsed_data.append(temp_dict)

        return parsed_data

    def build_prompts_and_chats(self, data, images):

        if self.task_type == "history":

            template = self.select_template("history")

            prompts = []
            chats = []
            for i, (images, report) in enumerate(
                zip(data["images_history"], data["reports"])
            ):
                images_tokens = " ".join("<image>" for _ in images)

                if i == 0:
                    prompts += [
                        {"role": "system", "content": template[0]},
                        {
                            "role": "user",
                            "content": template[1].format(
                                prior_images=images_tokens, prior_findings=report
                            ),
                        },
                    ]

                else:
                    chats += [
                        (
                            {
                                "role": "user",
                                "content": template[2].format(images=images_tokens),
                            },
                            {
                                "role": "assistant",
                                "content": template[3].format(report=report),
                            },
                        )
                    ]

        elif self.task_type == "history_cot":
            template = self.select_template("history_cot")

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

            prompts = []
            chats = []
            for i, (images, report) in enumerate(
                zip(data["images_history"], data["reports"])
            ):
                images_tokens = " ".join("<image>" for _ in images)

                if i == 0:
                    prompts += [
                        {"role": "system", "content": template[0]},
                        {
                            "role": "user",
                            "content": template[1].format(
                                prior_images=images_tokens, prior_findings=report
                            ),
                        },
                    ]

                else:

                    if positives == []:
                        # negative
                        first_answer = template[4]
                    else:
                        # positive
                        first_answer = template[3].format(
                            positive_findings=positive_findings
                        )

                    chats += [
                        (
                            {
                                "role": "user",
                                "content": template[2].format(
                                    images=images_tokens, findings=findings
                                ),
                            },
                            {
                                "role": "assistant",
                                "content": first_answer,
                            },
                        ),
                        (
                            {"role": "user", "content": template[5]},
                            {
                                "role": "assistant",
                                "content": template[6].format(report=report),
                            },
                        ),
                    ]

        return {"prompts": prompts, "chats": chats}
