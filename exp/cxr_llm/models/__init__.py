import os
from typing import List

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoProcessor, AutoTokenizer, CLIPProcessor
from transformers.utils import logging

from external.honeybee.honeybee import HoneybeeConfig, HoneybeeForConditionalGeneration

logger = logging.get_logger(__name__)


class MllmConfig(HoneybeeConfig):

    def __init__(
        self,
        vision_config: dict,
        projector_config: dict,
        lm_config: dict,
        **kwargs,
    ):
        super().__init__(vision_config, projector_config, lm_config, **kwargs)


class MllmForConditionalGeneration(HoneybeeForConditionalGeneration):
    config_class = MllmConfig

    def __init__(self, config: MllmConfig):
        super().__init__(config)


def set_trainable_parameters(model, module_to_update: List[str] = None):
    if module_to_update is None:
        return

    if "vision_model" in module_to_update:
        for param in model.vision_model.parameters():
            param.requires_grad = True

    if "abstractor" in module_to_update:
        for param in model.abstractor.parameters():
            param.requires_grad = True

    if "language_model" in module_to_update:
        for param in model.language_model.parameters():
            param.requires_grad = True


def load_mllm(config, **kwargs):
    dtype = config["dtype"]
    if type(dtype) is str:
        dtype = eval(dtype)

    # prepare model
    logger.info("Build model...")

    # debug with gpt2
    if config["debug"]:
        config["model_config"]["lm_config"][
            "pretrained_lm_name_or_path"
        ] = "openai-community/gpt2"
        config["model_config"]["lm_config"][
            "pretrained_tokenizer_name_or_path"
        ] = "openai-community/gpt2"
        config["lora_config"]["target_modules"] = ".*language_model.*\\.(c_attn|c_proj)"

    # load pretrained model
    if config["pretrained_ckpt"]:
        logger.info(f"Load PreTrainedModel : {config['pretrained_ckpt']}")
        model = MllmForConditionalGeneration.from_pretrained(
            config["pretrained_ckpt"], torch_dtype=dtype
        )

    # initialize model with model config
    else:
        mllm_config = MllmConfig(**config["model_config"])
        model = MllmForConditionalGeneration(mllm_config)

        # set dtype of model
        model.to(dtype)

    # load adapter & trained parameters
    if config["adapter_ckpt"]:
        model = PeftModel.from_pretrained(
            model, config["adapter_ckpt"], is_trainable=True
        )

    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config["model_config"]["lm_config"]["pretrained_tokenizer_name_or_path"]
    )

    # image processor
    image_processor = AutoProcessor.from_pretrained(
        config["model_config"]["vision_config"]["pretrained_vision_name_or_path"]
    )
    if type(image_processor) is CLIPProcessor:
        image_processor = image_processor.image_processor

    return {
        "model": model,
        "tokenizer": tokenizer,
        "image_processor": image_processor,
    }


def apply_params_setting(config, model, output_dir):

    # apply lora if requested
    if config["lora_config"]["use_lora"]:
        assert (
            type(model) != PeftModel
        ), "model is already PeftModel with adapter. Please check the model configs"

        # LoRA
        peft_config = LoraConfig(
            target_modules=r"{}".format(config["lora_config"]["target_modules"]),
            inference_mode=config["lora_config"]["inference_mode"],
            r=config["lora_config"]["lora_r"],
            lora_alpha=config["lora_config"]["lora_alpha"],
            lora_dropout=config["lora_config"]["lora_dropout"],
            # additional trainable parameters
            modules_to_save=config["module_to_update"],
        )

        # save base model for load base model and adapter
        if not config["pretrained_ckpt"]:
            # save checkpoint as base model
            model.save_pretrained(os.path.join(output_dir, "base"))

        # obtain peft-applied model and set the base_model_name_or_path
        model = get_peft_model(model, peft_config)
    else:
        # Full training
        # if not use lora, manually freeze base model's layers
        # Then, we set trainable parameters
        for param in model.parameters():
            param.requires_grad = False
        # set trainable parameters
        set_trainable_parameters(model, config["module_to_update"])

    return model
