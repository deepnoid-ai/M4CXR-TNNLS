import os

from tqdm import tqdm

from ..utils import print_rank_0
from .base import BaseDataset
from .common import load_json_files, normalize_bbox


class MscxrDataset(BaseDataset):
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

            if item["det"]:
                temp_dict["det"] = item["det"]

            temp_dict["task_type"] = self.task_type

            parsed_data.append(temp_dict)

        return parsed_data

    def build_prompts_and_chats(self, data, images):
        template = self.select_template(self.task_type)
        width = images["original_sizes"][0][1]
        height = images["original_sizes"][0][0]

        if self.task_type == "p_ground":

            question = []
            answer = []
            for i in data["det"]:
                question.append(i["name"])
                answer.append(i["label"])

            answer = answer[0]

            prompts = [
                {"role": "system", "content": template[0]},
                {"role": "user", "content": template[1]},
            ]

            chats = []

            norm_bbox = [
                normalize_bbox(*bbox, width, height, self.resize_type)
                for bbox in answer
            ]
            bbox_answer = ", ".join(norm_bbox)

            chat_pair = (
                {"role": "user", "content": template[2].format(phrase=question[0])},
                {
                    "role": "assistant",
                    "content": template[3].format(bbox=bbox_answer),
                },
            )

            chats.append(chat_pair)

        elif self.task_type == "grounded_p":
            template = self.select_template(self.task_type)

            question = []
            answer = []
            for i in data["det"]:
                question.append(i["label"])
                answer.append(i["name"])

            question = question[0]

            prompts = [
                {"role": "system", "content": template[0]},
                {"role": "user", "content": template[1]},
            ]

            chats = []

            norm_bbox = [
                normalize_bbox(*bbox, width, height, self.resize_type)
                for bbox in question
            ]

            bbox_question = ", ".join(norm_bbox)

            chat_pair = (
                {"role": "user", "content": template[2].format(bbox=bbox_question)},
                {"role": "assistant", "content": template[3].format(phrase=answer[0])},
            )

            chats.append(chat_pair)

        else:
            NotImplementedError
        return {"prompts": prompts, "chats": chats}
