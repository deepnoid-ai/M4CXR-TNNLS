import os

from tqdm import tqdm

from ..utils import print_rank_0
from .base import BaseDataset
from .common import load_json_files, mimic_cxr_test, mscxr_duplicate


class MimicVqaDataset(BaseDataset):
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
            raw_data_lst = [
                i
                for i in raw_data_lst
                if os.path.join("MIMIC-CXR/images/", i["image_id"] + ".jpg")
                not in mscxr_list
            ]

        # remove in MIMIC-CXR test dataset
        if "mimic_cxr_test" in kwargs and self.split == "train":
            mimic_test_list = mimic_cxr_test(self.data_root, kwargs["mimic_cxr_test"])

            raw_data_lst = [
                i
                for i in tqdm(raw_data_lst)
                if i["image_id"] + ".jpg" not in mimic_test_list
            ]

        self.dataset = self.load_data(raw_data_lst)

        print_rank_0(
            f"Load {self.__class__.__name__}, {self.task_type}, {self.split}. len: {len(self.dataset)}"
        )

    def load_data(self, raw_data):
        parsed_data = []
        for item in tqdm(raw_data):
            temp_dict = {
                "images": [
                    os.path.join(
                        self.data_root, "MIMIC-CXR", "images", item["image_id"] + ".jpg"
                    )
                ],
            }

            answer_list = item["answer"]
            if item["question"] and item["answer"]:
                if len(answer_list) == 1:
                    temp_dict["vqa"] = [item["question"], answer_list[0]]
                else:
                    answer_str = ""
                    for i in range(len(answer_list)):
                        if i == len(answer_list) - 1:
                            answer_str += answer_list[i]
                        else:
                            answer_str += answer_list[i] + ", "
                    temp_dict["vqa"] = [item["question"], f"{answer_str}"]
            else:
                continue
            temp_dict["task_type"] = self.task_type

            parsed_data.append(temp_dict)

        return parsed_data

    def build_prompts_and_chats(self, data, images):

        if self.task_type == "vqa":
            template = self.select_template(self.task_type)

            images_count = "<image>" * len(data["images"])

            prompts = [
                {"role": "system", "content": template[0]},
                {
                    "role": "user",
                    "content": template[1].format(image=images_count),
                },
            ]

            chats = []
            chat_pair = (
                {
                    "role": "user",
                    "content": template[2].format(question=data["vqa"][0]),
                },
                {
                    "role": "assistant",
                    "content": template[3].format(answer=data["vqa"][1] + "."),
                },
            )

            chats.append(chat_pair)

        return {"prompts": prompts, "chats": chats}
