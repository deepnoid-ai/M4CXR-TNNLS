import random

import numpy as np
import torch
from torch.utils.data.dataset import Dataset

from external.honeybee import utils

from ..templates import default_template


class BaseDataset(Dataset):
    """Base dataset class

    Data loading process:
        (offline) init -> load_data -> (finalize_data)
        (online) __getitem__ -> process_data -> preprocess_data ->
            image_processor -> build_text_from_data -> tokenizer
    """

    def __init__(
        self,
        tokenizer,
        processor,
        max_length,
        data_root,
        split,
        dset_name,
        resize_type,
        template_name,
        shuffle_multi_turn=False,
        **kwargs
    ):
        super().__init__()

        self.tokenizer = tokenizer
        self.processor = processor
        self.max_length = max_length
        self.data_root = data_root
        self.split = split
        self.dset_name = dset_name
        self.resize_type = resize_type

        if self.split == "train":
            self.shuffle_multi_turn = shuffle_multi_turn
        else:
            self.shuffle_multi_turn = False

        self.templates = eval(template_name)

    def get_dataset_name(self):
        return self.dset_name

    def select_template(self, key):
        # select 1st (default) template
        template = self.templates[key][0]

        # TODO: random template selection
        return template

    def __len__(self):
        return len(self.dataset)

    def load_data(self):
        """Load data files and parse data samples with dataset-specific parsing logics

        The result instruction text should follow the shared format example:
            'system message'
            Human: 'prompt'
            Human: <image>
            AI: 'answer'

        Required keys in result dictionary:
            'image': pull path of an image file
            'task_type': used for selecting a processor
            NOTE templatizer parse 'examples' into 'text'; only one or the other is required.
            'text': parsed instruction text like above example
            'examples': a list of examples for template-based instruction generation

        Return:
            Parsed data list
        """
        raise NotImplementedError()

    def preprocess_data(self, data):
        """perform pre-processing for the given data if required
        Args:
            data: datapoint given from self.dataset
        """
        return data

    def build_prompts_and_chats(self, data):
        """
        Apply a template to the data to construct prompts and chats.
        """
        return {"prompts": [], "chats": []}

    def build_text_from_chats(self, prompts, chats):
        "Convert prompts and chats into a single sequence of text."
        output_chats = []
        prompt = ""
        for contents in prompts:
            assert contents["role"] != "assistant"

            # vicuna support system prompt
            if self.tokenizer.chat_template_type == "vicuna":
                if contents["role"] == "system":
                    output_chats.append(contents)
                    continue

            prompt += contents["content"] + " "

        for i, chat in enumerate(chats):
            question = chat[0]
            answer = chat[1]

            # add system prompt and image token
            if i == 0:
                question["content"] = prompt + question["content"]

            output_chats.append(question)
            output_chats.append(answer)

        text = self.tokenizer.apply_chat_template(output_chats, tokenize=False)

        return text

    def process_images(self, data, processor):
        # SamImageProcessor style

        # Process Image if exists
        if "images" in data and len(data["images"]) > 0:
            image_urls = data["images"]
            pil_images = utils.load_images(image_urls)
            # [(height, width), ...]
            original_sizes = [(image.height, image.width) for image in pil_images]
            images = processor(pil_images)
            images["pixel_values"] = torch.FloatTensor(np.array(images["pixel_values"]))
            images["original_sizes"] = original_sizes
            images["image_path"] = image_urls
        else:
            images = None
        return images

    def process_data(self, data, processor):
        data = self.preprocess_data(data)

        # Process Image if exists
        images = self.process_images(data, processor)

        # Process Text
        # build prompts and chats
        outputs = self.build_prompts_and_chats(data, images)
        prompts = outputs["prompts"]
        chats = outputs["chats"]

        # shuffle multi-turn question-answer pair
        if self.shuffle_multi_turn:
            random.shuffle(chats)

        # prompts and chats to text
        text = self.build_text_from_chats(prompts, chats)
        text_input = self.tokenizer.encode_prompt(text, self.max_length)

        return {
            "image": images,
            "text_raw": text,
            "text": text_input,
            "task_type": data["task_type"],
            "dset_name": self.dset_name,
            "image_path": images["image_path"] if images is not None else "None",
        }

    def __getitem__(self, index):
        data = self.dataset[index]
        data = self.process_data(data, self.processor)
        return data
