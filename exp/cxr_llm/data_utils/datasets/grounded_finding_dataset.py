import os

from tqdm import tqdm

from ..utils import print_rank_0
from .base import BaseDataset
from .common import load_json_files, normalize_bbox


class GroundedFindingDataset(BaseDataset):
    def __init__(
        self,
        tokenizer,
        processors,
        max_length,
        data_root: str,
        **kwargs,
    ):
        super().__init__(tokenizer, processors, max_length, data_root, **kwargs)

        self.task_type = kwargs["task_type"]  # grounded_f
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

            # task type
            temp_dict["cls"] = item["cls"]
            if "det" in item:
                new_det_dicts = []
                for det_dict in item["det"]:
                    if det_dict["name"] != "lung":  # except lung
                        if det_dict["label"] == [[0, 0, 0, 0]]:
                            pass
                        else:  # positive only
                            if len(det_dict["label"]) > 1:  # biggest bbox
                                max_bbox = max(
                                    det_dict["label"],
                                    key=lambda x: (x[2] - x[0]) * (x[3] - x[1]),
                                )
                                det_dict["label"] = [max_bbox]
                            new_det_dicts.append(det_dict)
                if new_det_dicts == []:
                    continue
                temp_dict["det"] = new_det_dicts
            else:
                continue
            temp_dict["task_type"] = self.task_type

            parsed_data.append(temp_dict)

        return parsed_data

    def build_prompts_and_chats(self, data, images):
        # grounded finding
        template = self.select_template(self.task_type)

        prompts = [
            {"role": "system", "content": template[0]},
            {"role": "user", "content": template[1]},
        ]

        chats = []
        for det_data in data["det"]:
            det_name = det_data["name"].lower()
            det_name += "."
            det_label = det_data["label"]

            width = images["original_sizes"][0][1]
            height = images["original_sizes"][0][0]
            norm_bbox = [
                normalize_bbox(*bbox, width, height, self.resize_type)
                for bbox in det_label
            ]

            qa_pair = (
                {"role": "user", "content": template[2].format(bbox=norm_bbox[0])},
                {
                    "role": "assistant",
                    "content": template[3].format(finding=det_name),
                },
            )
            chats.append(qa_pair)
        return {"prompts": prompts, "chats": chats}
