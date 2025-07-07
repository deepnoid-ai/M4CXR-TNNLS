# flake8: noqa

import csv
import os

from tqdm import tqdm

from common.utils import load_json
from exp.cxr_llm.data_utils.datasets.base import BaseDataset
from external.honeybee.utils import print_rank_0

from .common import mimic_cxr_test, mscxr_duplicate


class MimiccxrDiffVqaDataset(BaseDataset):
    def __init__(
        self,
        tokenizer,
        processors,
        max_length,
        data_root: str,
        **kwargs,
    ):
        super().__init__(tokenizer, processors, max_length, data_root, **kwargs)

        self.task_type = kwargs["task_type"]  # vqa, diff

        raw_data_lst = load_json(os.path.join(self.data_root, kwargs[self.split]))
        # remove mscxr from train dataset
        if "mscxr_duplicate" in kwargs and self.split == "train":
            mscxr_list = mscxr_duplicate(self.data_root, kwargs["mscxr_duplicate"])
            raw_data_lst = [
                i
                for i in raw_data_lst
                if not (
                    i["reference_image"] in mscxr_list and i["main_image"] in mscxr_list
                )
            ]

        # remove in MIMIC-CXR test dataset
        if "mimic_cxr_test" in kwargs and self.split == "train":
            mimic_test_list = mimic_cxr_test(self.data_root, kwargs["mimic_cxr_test"])

            raw_data_lst = [
                i
                for i in raw_data_lst
                if not (
                    os.path.basename(i["reference_image"]) in mimic_test_list
                    and os.path.basename(i["reference_image"]) in mimic_test_list
                )
            ]

        self.dataset = self.load_data(raw_data_lst)

        print_rank_0(
            f"Load {self.__class__.__name__}, {self.task_type}, {self.split}. len: {len(self.dataset)}"
        )

    def load_data(self, raw_data):
        """
        mimic_diff_vqa
        [
        {
            "reference_image": "MIMIC-CXR/images/e96cd900-31a079b1-623a5195-4874640a-963b7303.jpg",
            "main_image": "MIMIC-CXR/images/a960ea54-53d720dc-e613637c-738f6e79-03538f3f.jpg",
            "question_type": "difference",
            "question": "what has changed compared to the reference image?",
            "answer": "the main image is missing the finding of lung opacity than the reference image. "
        },
        ...
        ]

        """

        parsed_data = []
        for index, data in tqdm(enumerate(raw_data), total=len(raw_data)):
            # task is difference
            if self.task_type == "diff":

                reference_image = os.path.join(self.data_root, data["reference_image"])

                main_image = os.path.join(self.data_root, data["main_image"])

                temp_dict = {}
                temp_dict["images"] = [reference_image, main_image]
                temp_dict["task_type"] = self.task_type
                temp_dict["vqa"] = [data["question"], data["answer"]]

                parsed_data.append(temp_dict)

            elif self.task_type == "vqa":

                main_image = os.path.join(self.data_root, data["main_image"])

                temp_dict = {}
                temp_dict["images"] = [main_image]
                temp_dict["task_type"] = self.task_type
                temp_dict["vqa"] = [data["question"], data["answer"]]

                parsed_data.append(temp_dict)

        return parsed_data

    def build_prompts_and_chats(self, data, images):

        if self.task_type == "diff":
            template = self.select_template(self.task_type)

            prompts = [
                {"role": "system", "content": template[0]},
                {"role": "user", "content": template[1]},
            ]

            chats = [
                (
                    {
                        "role": "user",
                        "content": template[2].format(question=data["vqa"][0]),
                    },
                    {
                        "role": "assistant",
                        "content": template[3].format(answer=data["vqa"][1]),
                    },
                )
            ]

        elif self.task_type == "vqa":
            template = self.select_template(self.task_type)

            prompts = [
                {"role": "system", "content": template[0]},
                {
                    "role": "user",
                    "content": template[1].format(image="<image>"),
                },
            ]

            chats = [
                (
                    {
                        "role": "user",
                        "content": template[2].format(question=data["vqa"][0]),
                    },
                    {
                        "role": "assistant",
                        "content": template[3].format(answer=data["vqa"][1]),
                    },
                )
            ]

        return {"prompts": prompts, "chats": chats}
