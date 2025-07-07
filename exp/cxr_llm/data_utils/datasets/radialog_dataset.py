import os

from tqdm import tqdm

from ..utils import clean_report_mimic_cxr, print_rank_0
from .base import BaseDataset
from .common import load_json_files, mimic_cxr_train, mscxr_duplicate


class RadialogDataset(BaseDataset):
    def __init__(
        self,
        tokenizer,
        processors,
        max_length,
        data_root: str,
        **kwargs,
    ):
        super().__init__(tokenizer, processors, max_length, data_root, **kwargs)

        raw_data_lst = load_json_files(os.path.join(self.data_root, kwargs[self.split]))

        # remove mscxr from train dataset
        if "mscxr_duplicate" in kwargs and self.split == "train":
            mscxr_list = mscxr_duplicate(self.data_root, kwargs["mscxr_duplicate"])
            # raw_data_lst에서 mscxr_list에 포함되지 않은 항목들만 남기기
            raw_data_lst = [
                i
                for i in tqdm(raw_data_lst)
                if os.path.join("MIMIC-CXR", "images", os.path.basename(i["image"]))
                not in mscxr_list
            ]

        # use only in MIMIC-CXR train dataset
        if "mimic_cxr_train" in kwargs and self.split == "train":
            mimic_train_list = mimic_cxr_train(
                self.data_root, kwargs["mimic_cxr_train"]
            )

            raw_data_lst = [
                i
                for i in tqdm(raw_data_lst)
                if os.path.basename(i["image"]) in mimic_train_list
            ]

        self.dataset = self.load_data(raw_data_lst)

        print_rank_0(
            f"Load {self.__class__.__name__}, {self.split}. len: {len(self.dataset)}"
        )

    def load_data(self, raw_data):
        """Data is a list where each item is similar to following
        {
            "image": "427446c1-881f5cce-85191ce1-91a58ba9-0a57d3f5.jpg",
            "conversations": [{
                "from": "human",
                "value": "<image>. What perspective is shown in this image?"
            },
            {
                "from": "gpt",
                "value": "The perspective shown in this image is PA."
            }
        ]
        },
        """

        mrg_template = self.select_template("mrg")

        parsed_data = []
        for item in tqdm(raw_data):
            # task_type RR,RG는 제외
            if item["task_type"] not in ["RG"]:
                item["image"] = item["image"].split("/")[-1]
                temp_dict = {
                    "images": [
                        os.path.join(
                            self.data_root, "MIMIC-CXR", "images", item["image"]
                        )
                    ],
                }

                # medical report generation
                # replace prompt with ours
                if (
                    "write the finding section of a chest x-ray radiology report for this X-ray image"
                    in item["conversations"][0]["value"]
                ):
                    item["conversations"][0]["value"] = mrg_template[2]

                    # cleansing report
                    item["conversations"][1]["value"] = clean_report_mimic_cxr(
                        item["conversations"][1]["value"]
                    )
                else:
                    # remove image token
                    item["conversations"][0]["value"] = item["conversations"][0][
                        "value"
                    ].replace("<image>. ", "")

                temp_dict["conversations"] = item["conversations"]

                temp_dict["task_type"] = "vqa"
                parsed_data.append(temp_dict)

        return parsed_data

    def build_prompts_and_chats(self, data, images):

        template = self.select_template("radialog_vqa")

        # templates
        prompts = [
            {"role": "system", "content": template[0]},
            {"role": "user", "content": template[1]},
        ]

        chats = []
        for i in range(len(data["conversations"]) // 2):
            chats.append(
                (
                    {
                        "role": "user",
                        "content": template[2].format(
                            question=data["conversations"][2 * i]["value"]
                        ),
                    },
                    {
                        "role": "assistant",
                        "content": template[3].format(
                            answer=data["conversations"][2 * i + 1]["value"]
                        ),
                    },
                )
            )

        return {"prompts": prompts, "chats": chats}
