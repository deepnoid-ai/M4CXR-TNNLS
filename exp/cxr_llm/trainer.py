import os
from collections import defaultdict
from typing import Dict, List, Optional, Union

import torch
from data_utils.templates import IGNORE_INDEX
from data_utils.utils import merge_consecutive_tokens
from torch import nn
from torch.nn.utils.rnn import pad_sequence

from common.trainer import BaseTrainer
from common.utils import (
    concat_with_splitter,
    deserialize,
    save_json,
    serialize,
    split_with_splitter,
)


class CXRLLMTrainer(BaseTrainer):
    def __init__(
        self,
        model=None,
        args=None,
        data_collator=None,
        train_dataset=None,
        eval_dataset=None,
        tokenizer=None,
        model_init=None,
        compute_metrics=None,
        callbacks=None,
        optimizers=(None, None),
        preprocess_logits_for_metrics=None,
        processor=None,
        cfg=None,
        **kwargs,
    ):
        self.processor = processor

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            model_init=model_init,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
            cfg=cfg,
        )

    def compute_loss(self, model, inputs, return_outputs=False):
        # forward pass
        outputs = model(**inputs)
        loss = outputs["loss"]

        return (loss, outputs) if return_outputs else loss

    def floating_point_ops(self, inputs):
        # error exception for text only dataset
        if ("pixel_values" in inputs) and (inputs.get("pixel_values") is None):
            inputs.pop("pixel_values")

        if hasattr(self.model, "floating_point_ops"):
            return self.model.floating_point_ops(inputs)
        else:
            return 0

    def inference_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, List]],
        ignore_keys: Optional[List[str]] = None,
        **kwargs,
    ):
        inputs = self._prepare_inputs(inputs)
        if ignore_keys is None:
            if hasattr(self.model, "config"):
                ignore_keys = getattr(
                    self.model.config, "keys_to_ignore_at_inference", []
                )
            else:
                ignore_keys = []

        batch_size = inputs["num_images"].size(0)
        image_indices = [0] + torch.cumsum(inputs["num_images"], dim=0).tolist()

        results = defaultdict(list)

        for i in range(batch_size):

            prompt = []
            outputs = []

            input_id = inputs["input_ids"][i]
            label_mask = inputs["labels"][i] != IGNORE_INDEX  # 1 if label, else 0
            input_text = IGNORE_INDEX * label_mask + input_id * (1 - label_mask.int())
            # make sure IGNORE_INDEX == splitter(-100)
            texts = split_with_splitter(input_text)

            # pixel values with num images per data
            pixel_values = (
                inputs["pixel_values"][image_indices[i] : image_indices[i + 1]]
                if image_indices[i] != image_indices[i + 1]
                else None
            )

            for text in texts:
                if (text == self.tokenizer.eos_token_id).all():
                    continue
                text = text.unsqueeze(0)
                prompt.append(text)
                input_ids = torch.cat(prompt, dim=1)

                output = model.generate(
                    pixel_values=pixel_values,
                    input_ids=input_ids,
                    **self.cfg["model"]["generate_config"],
                )  # [1 x seq_len]
                prompt.append(output)
                outputs.append(output[0])

            # concat outputs with splitter (-100)
            outputs = concat_with_splitter(outputs)

            results["outputs"].append(outputs)
            results["prompts"].append(input_text)
            results["labels"].append(inputs["labels"][i])
            results["image_paths"].append(
                torch.LongTensor(serialize(inputs["image_path"][i])).to(output.device)
            )

        outputs = {}
        for k, v in results.items():
            outputs[k] = pad_sequence(v, batch_first=True, padding_value=-100)

        # remove outputs to ignore
        for k in ignore_keys:
            if k in outputs:
                outputs.pop(k)

        return outputs

    def save_inference_outputs(
        self,
        outputs,
        output_dir: Optional[str] = None,
        checkpoint_name: Optional[str] = None,
        **kwargs,
    ):
        if output_dir is None:
            output_dir = os.path.join(self.args.output_dir, "inference")
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, f"{checkpoint_name}.json")

        results = {}

        inference_list = []
        predictions = outputs.predictions
        for output, prompt, label, image_path in zip(
            predictions["outputs"],
            predictions["prompts"],
            predictions["labels"],
            predictions["image_paths"],
        ):
            image_path = deserialize(image_path[image_path > -100])

            # split tensors with -100 for multi-turn
            prompts = split_with_splitter(torch.LongTensor(prompt))
            labels = split_with_splitter(torch.LongTensor(label))
            pred_texts = split_with_splitter(torch.LongTensor(output))

            # tensor to text
            prompt_list, label_list, output_list = [], [], []
            for p, l, o in zip(prompts, labels, pred_texts):
                prompt_list.append(self.decode_with_media_tokens(p))
                label_list.append(self.decode_with_media_tokens(l))
                output_list.append(self.decode_with_media_tokens(o))

            result = {
                "image": image_path,
                "prompts": prompt_list,
                "labels": label_list,
                "outputs": output_list,
            }
            inference_list.append(result)

        save_json(inference_list, save_path)

        results["inference_list"] = inference_list
        results["save_path"] = save_path

        return results

    def decode_with_media_tokens(self, tensor):
        for token_str, value in self.tokenizer.media_tokens.items():
            tensor = merge_consecutive_tokens(tensor, value)
            replace_tokens = self.tokenizer.encode(token_str, add_special_tokens=False)
            indices = (tensor == value).nonzero(as_tuple=True)[0]
            if indices.numel() > 0:
                slices = []

                # Start index for slicing
                start = 0
                # Iterate over the indices
                for idx in indices:
                    slices.append(tensor[start:idx])
                    # Append the replacement tensor to the list
                    slices.append(torch.tensor(replace_tokens))
                    # Update the start index for the next slice
                    start = idx + 1
                # Append the remaining slice of the tensor after the last -1
                slices.append(tensor[start:])
                # Concatenate all slices and replacement tensors
                tensor = torch.cat(slices)

        return self.tokenizer.decode(
            merge_consecutive_tokens(tensor, self.tokenizer.eos_token_id)
        )
