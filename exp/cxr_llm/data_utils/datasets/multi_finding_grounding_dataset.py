import os

from tqdm import tqdm

from ..utils import print_rank_0
from .base import BaseDataset
from .common import load_json_files, normalize_bbox


class MultiFindingGroundingDataset(BaseDataset):
    def __init__(
        self,
        tokenizer,
        processors,
        max_length,
        data_root: str,
        **kwargs,
    ):
        super().__init__(tokenizer, processors, max_length, data_root, **kwargs)

        self.task_type = kwargs["task_type"]  # mf_ground
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
                det_dicts = []
                for det_dict in item["det"]:
                    if det_dict["name"] != "lung":  # except lung
                        det_dicts.append(det_dict)
                temp_dict["det"] = det_dicts
                if temp_dict["det"] == []:
                    continue
            else:
                continue
            temp_dict["task_type"] = self.task_type

            parsed_data.append(temp_dict)

        return parsed_data

    def build_prompts_and_chats(self, data, images):
        # multi finding grounding
        template = self.select_template(self.task_type)
        findings = []
        positives = []
        positive_bboxs = []
        for det_data in data["det"]:
            det_name = det_data["name"].lower()
            det_label = det_data["label"]
            findings.append(det_name)

            if det_data["label"] != [[0, 0, 0, 0]]:
                width = images["original_sizes"][0][1]
                height = images["original_sizes"][0][0]
                norm_bbox = [
                    normalize_bbox(*bbox, width, height, self.resize_type)
                    for bbox in det_label
                ]

                positives.append(det_name)
                positive_bboxs.append(norm_bbox)

        findings = ", ".join(findings)

        prompts = [
            {"role": "system", "content": template[0]},
            {"role": "user", "content": template[1]},
        ]

        if positives == []:
            # negative
            qa_pair = (
                {
                    "role": "user",
                    "content": template[2].format(findings=findings),
                },
                {"role": "assistant", "content": template[4]},
            )
        else:
            # positvives
            parts = []
            for i, bbox_list in enumerate(positive_bboxs):
                bbox_str = ", ".join(bbox_list)
                parts.append(f"{positives[i]} {bbox_str}")

            answer = ", ".join(parts)

            qa_pair = (
                {
                    "role": "user",
                    "content": template[2].format(findings=findings),
                },
                {
                    "role": "assistant",
                    "content": template[3].format(findings_bboxes=answer),
                },
            )

        return {"prompts": prompts, "chats": [qa_pair]}
