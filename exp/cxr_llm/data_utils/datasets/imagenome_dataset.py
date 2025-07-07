import os

from tqdm import tqdm

from ..utils import print_rank_0
from .base import BaseDataset
from .common import load_json_files, mimic_cxr_test, mscxr_duplicate, normalize_bbox


class ImagenomeDataset(BaseDataset):
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
            raw_data_lst = [i for i in raw_data_lst if i["image"] not in mscxr_list]

        # remove in MIMIC-CXR test dataset
        if "mimic_cxr_test" in kwargs and self.split == "train":
            mimic_test_list = mimic_cxr_test(self.data_root, kwargs["mimic_cxr_test"])

            raw_data_lst = [
                i
                for i in tqdm(raw_data_lst)
                if os.path.basename(i["image"]) not in mimic_test_list
            ]

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
            # {region : bbox}
            region_bbox_info = item["bbox"]

            # {region : phrase}
            region_phrase_info = item["phrase"]

            region_bbox_phrase_info = []
            # {region : [bbox, phrase]}
            for region_bbox, region_phrase in zip(region_bbox_info, region_phrase_info):
                region_name_bbox = region_bbox["region"]
                region_bbox = region_bbox["bbox"]

                region_name_phrase = region_phrase["region"]
                region_phrase = region_phrase["phrase"]

                region_bbox_phrase_dict = {}

                if region_name_bbox == region_name_phrase:
                    region_bbox_phrase_dict[region_name_bbox] = [
                        region_bbox,
                        region_phrase,
                    ]
                    region_bbox_phrase_info.append(region_bbox_phrase_dict)

            temp_dict["region_phrase_bbox"] = region_bbox_phrase_info

            # region bbox phrases
            temp_dict["task_type"] = self.task_type

            parsed_data.append(temp_dict)

        return parsed_data

    def build_prompts_and_chats(self, data, images):
        if self.task_type == "grounded_p":
            """
            data['region_phrase_bbox'] = [{a : [bbox, phrase]}, {b : [bbox, phrase]}, ...]
            # {region_name : value=[[bbox],phrase]}

            grounded_p: Task of matching the phrase corresponding to the bounding box.
            a_ground, grounded_a: Task of matching the Name of the anatomical region corresponding to the bounding box (and vice versa).
            """
            template = self.select_template(self.task_type)

            prompts = [
                {"role": "system", "content": template[0]},
                {"role": "user", "content": template[1]},
            ]

            chats = []
            for region_phrase_bbox in data["region_phrase_bbox"]:
                for region_name, bbox_phrase in region_phrase_bbox.items():

                    if bbox_phrase[1] != "":

                        bbox = bbox_phrase[0]
                        width = images["original_sizes"][0][1]
                        height = images["original_sizes"][0][0]
                        norm_bbox = normalize_bbox(
                            *bbox, width, height, self.resize_type
                        )

                        qa_pair = (
                            {
                                "role": "user",
                                "content": template[2].format(bbox=norm_bbox),
                            },
                            {
                                "role": "assistant",
                                "content": template[3].format(phrase=bbox_phrase[1]),
                            },
                        )
                        chats.append(qa_pair)
        elif self.task_type == "p_ground":
            template = self.select_template(self.task_type)

            prompts = [
                {"role": "system", "content": template[0]},
                {"role": "user", "content": template[1]},
            ]

            chats = []
            for region_phrase_bbox in data["region_phrase_bbox"]:
                for region_name, bbox_phrase in region_phrase_bbox.items():

                    if bbox_phrase[1] != "":

                        bbox = bbox_phrase[0]
                        width = images["original_sizes"][0][1]
                        height = images["original_sizes"][0][0]
                        norm_bbox = normalize_bbox(
                            *bbox, width, height, self.resize_type
                        )

                        qa_pair = (
                            {
                                "role": "user",
                                "content": template[2].format(phrase=bbox_phrase[1]),
                            },
                            {
                                "role": "assistant",
                                "content": template[3].format(bbox=norm_bbox),
                            },
                        )
                        chats.append(qa_pair)

        elif self.task_type == "a_ground":
            template = self.select_template(self.task_type)

            prompts = [
                {"role": "system", "content": template[0]},
                {"role": "user", "content": template[1]},
            ]

            chats = []

            for region_phrase_bbox in data["region_phrase_bbox"]:
                for region_name, bbox_phrase in region_phrase_bbox.items():

                    bbox = bbox_phrase[0]
                    width = images["original_sizes"][0][1]
                    height = images["original_sizes"][0][0]
                    norm_bbox = normalize_bbox(*bbox, width, height, self.resize_type)

                    qa_pair = (
                        {
                            "role": "user",
                            "content": template[2].format(name=region_name),
                        },
                        {
                            "role": "assistant",
                            "content": template[3].format(bbox=norm_bbox),
                        },
                    )
                    chats.append(qa_pair)

        elif self.task_type == "grounded_a":
            template = self.select_template(self.task_type)

            prompts = [
                {"role": "system", "content": template[0]},
                {"role": "user", "content": template[1]},
            ]

            chats = []

            for region_phrase_bbox in data["region_phrase_bbox"]:
                for region_name, bbox_phrase in region_phrase_bbox.items():

                    bbox = bbox_phrase[0]
                    width = images["original_sizes"][0][1]
                    height = images["original_sizes"][0][0]
                    norm_bbox = normalize_bbox(*bbox, width, height, self.resize_type)

                    qa_pair = (
                        {
                            "role": "user",
                            "content": template[2].format(bbox=norm_bbox),
                        },
                        {
                            "role": "assistant",
                            "content": template[3].format(name=region_name),
                        },
                    )
                    chats.append(qa_pair)

        else:
            NotImplementedError

        return {"prompts": prompts, "chats": chats}
