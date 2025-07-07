import numpy as np
import torch
from peft import PeftModel
from PIL import Image
from transformers import AutoProcessor, AutoTokenizer

from exp.cxr_llm.data_utils.templates import SYSTEM_MESSAGE
from exp.cxr_llm.models import MllmForConditionalGeneration
from exp.cxr_llm.models.tokenization_mllm import build_mllm_tokenizer


def load_images(image_paths: list[str] | str) -> list[Image.Image]:
    if isinstance(image_paths, str):
        image_paths = [image_paths]
    return [Image.open(image_path).convert("RGB") for image_path in image_paths]


def load_models(
    pretrained_model_path,
    adapter_path,
    image_processor_path,
    tokenizer_path,
    num_visual_tokens,
    chat_template,
    device,
    dtype,
):

    # base model
    model = MllmForConditionalGeneration.from_pretrained(
        pretrained_model_path,
        torch_dtype=dtype,
    ).to(device)

    # adapter
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)

    # image_processor
    image_processor = AutoProcessor.from_pretrained(image_processor_path)

    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    tokenizer = build_mllm_tokenizer(tokenizer, num_visual_tokens, chat_template)

    return {"model": model, "image_processor": image_processor, "tokenizer": tokenizer}


def chat(
    prompts,
    image_paths,
    max_length,
    generate_config,
    model,
    image_processor,
    tokenizer,
    system_prompt=SYSTEM_MESSAGE,
    verbose=True,
):

    image_token_count = sum(i.count("<image>") for i in prompts)

    if image_token_count > 0 and image_paths:
        assert (
            len(image_paths) == image_token_count
        ), "Please use the '<image>' token that matches the number of images."

    if image_paths:
        pil_images = load_images(image_paths)
        original_sizes = [(image.height, image.width) for image in pil_images]
        images = image_processor(pil_images)
        images["original_sizes"] = original_sizes
        pixel_values = torch.FloatTensor(np.array(images["pixel_values"])).to(
            model.device
        )
    else:
        pixel_values = None

    chats = []

    # vicuna support system prompt
    if tokenizer.chat_template_type == "vicuna":
        chats.append({"role": "system", "content": system_prompt})
    else:
        prompts[0] = system_prompt + " " + prompts[0]

    for prompt in prompts:
        chats.append({"role": "user", "content": prompt})

        text = tokenizer.apply_chat_template(
            chats, tokenize=False, add_generation_prompt=True
        )
        input_ids = (
            tokenizer.encode_prompt(text, max_length=max_length)["input_ids"]
            .unsqueeze(0)
            .to(model.device)
        )

        output = model.generate(
            pixel_values=pixel_values, input_ids=input_ids, **generate_config
        )

        chats.append(
            {
                "role": "assistant",
                "content": tokenizer.decode(output[0], skip_special_tokens=True),
            }
        )

        if verbose:
            print("=" * 40)
            print(f"Question : {chats[-2]['content']}")
            print(f"Answer : {chats[-1]['content']}")
            print("=" * 40)

    return chats


def generate_with_prompt(
    prompt_text,
    image_paths,
    max_length,
    generate_config,
    model,
    image_processor,
    tokenizer,
    verbose=True,
):
    image_token_count = prompt_text.count("<image>")

    if image_token_count > 0 and image_paths:
        assert (
            len(image_paths) == image_token_count
        ), "Please use the '<image>' token that matches the number of images."

    if image_paths:
        pil_images = load_images(image_paths)
        original_sizes = [(image.height, image.width) for image in pil_images]
        images = image_processor(pil_images)
        images["original_sizes"] = original_sizes
        pixel_values = torch.FloatTensor(np.array(images["pixel_values"])).to(
            model.device
        )
    else:
        pixel_values = None

    input_ids = (
        tokenizer.encode_prompt(prompt_text, max_length=max_length)["input_ids"]
        .unsqueeze(0)
        .to(model.device)
    )

    output = model.generate(
        pixel_values=pixel_values, input_ids=input_ids, **generate_config
    )

    chat_output = {
        "role": "assistant",
        "content": tokenizer.decode(output[0], skip_special_tokens=True),
    }

    if verbose:
        print("=" * 40)
        print(f"Prompt Text : {prompt_text}")
        print(f"Answer : {chat_output}")
        print("=" * 40)

    return chat_output


if __name__ == "__main__":

    models = load_models(
        pretrained_model_path="abstractor_pretraining/checkpoint-2000",
        adapter_path="instruction_tuning/checkpoint-40000",
        image_processor_path="microsoft/rad-dino",
        tokenizer_path="abstractor_pretraining/checkpoint-2000",
        num_visual_tokens=361,
        chat_template="mistral",
        device=torch.device("cuda"),
        dtype=torch.float32,
    )

    # image
    image_paths = [
        "datasets/MIMIC-CXR/images/58c1653b-5d091bf0-ce12c7b3-7f6c738d-cd6f4e1b.jpg",
        "datasets/MIMIC-CXR/images/58c1653b-5d091bf0-ce12c7b3-7f6c738d-cd6f4e1b.jpg",
    ]

    # Prompt list
    prompts = [
        "radilogy image : <image>, What is the view of this chest X-ray?",
        "follow-up image: <image> Provide a description of the findings in the radiology image.",
    ]

    max_length = 4095
    generate_config = {"max_new_tokens": 500, "do_sample": False}

    # chat with model
    chats = chat(
        prompts=prompts,
        image_paths=image_paths,
        max_length=max_length,
        generate_config=generate_config,
        **models,
    )
    print(chats)

    chats.append(
        {
            "role": "user",
            "content": "Explain this radiology report to the patient with easy language.",
        }
    )
    prompt_text = models["tokenizer"].apply_chat_template(
        chats, tokenize=False, add_generation_prompt=True
    )

    # generate with prompt
    chat_output = generate_with_prompt(
        prompt_text=prompt_text,
        image_paths=image_paths,
        max_length=max_length,
        generate_config=generate_config,
        **models,
    )

    chats.append(chat_output)
    print()
