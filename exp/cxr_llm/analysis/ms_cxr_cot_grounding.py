import os
import sys

import torch
from grounding_visualization import norm2origin_box
from PIL import Image, ImageDraw

from common.utils import load_json, save_json
from exp.cxr_llm.demo import chat, generate_with_prompt, load_models

# chexbert findings
findings = "Enlarged Cardiomediastinum, Cardiomegaly, Lung Opacity, Lung Lesion, Edema, Consolidation, Pneumonia, Atelectasis, Pneumothorax, Pleural Effusion, Pleural Other, Fracture, Support Devices"


def grounding_qa():
    prompt_list = []

    # multi-turn, 1. multi disease identification 2. report generation 3. explain easy language (Radialog)
    prompt_list.append(
        [
            f"radiology image: <image> Which of the following findings are present in the radiology image? Findings: {findings}",
            "Based on the previous conversation, provide a description of the findings in the radiology image.",
        ]
    )

    return prompt_list


def calculate_iou(gt_box, predict_box):
    # Unpack the coordinates
    x1_min, y1_min, x1_max, y1_max = gt_box
    x2_min, y2_min, x2_max, y2_max = predict_box

    # Calculate the coordinates of the intersection rectangle
    x_min_intersection = max(x1_min, x2_min)
    y_min_intersection = max(y1_min, y2_min)
    x_max_intersection = min(x1_max, x2_max)
    y_max_intersection = min(y1_max, y2_max)

    # Calculate the area of the intersection
    intersection_width = max(0, x_max_intersection - x_min_intersection)
    intersection_height = max(0, y_max_intersection - y_min_intersection)
    intersection_area = intersection_width * intersection_height

    # Calculate the area of both bounding boxes
    gt_area = (x1_max - x1_min) * (y1_max - y1_min)
    predict_area = (x2_max - x2_min) * (y2_max - y2_min)

    # Calculate the union area
    union_area = gt_area + predict_area - intersection_area

    # Calculate the IoU
    iou = intersection_area / union_area if union_area != 0 else 0

    return iou


if __name__ == "__main__":
    import datetime

    data_root = "datasets"
    save_dir = "qualitative_eval_grounding"

    now = "run-%s" % datetime.datetime.now().strftime("%m%d-%H%M%S")

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

    max_length = 4095
    generation_config = {"max_new_tokens": 512, "do_sample": False}

    # ms_cxr_test dataset
    test_data = load_json(
        os.path.join(
            data_root,
            "/share-data/advanced_tech/datasets/MS-CXR-0.1/preprocess/v2.0/test.json",
        )
    )

    results = []
    for data in test_data:
        image_paths = [os.path.join(data_root, data["image"])]

        # To compare boxes (debug)
        # image = Image.open(image_paths[-1]).convert("RGB")
        # original_size = (image.height, image.width)
        # height = original_size[0]
        # width = original_size[1]
        # resize_type = "shortest_edge"
        # fig_size = (10, 8)

        # generate
        result = {}
        result["input"] = {
            "image": data["image"],
            "findings": data["det"][-1]["name"],
            "det": data["det"][-1]["label"],
        }
        result["generation_config"] = generation_config
        result["chats"] = list()
        qa_list = grounding_qa()
        for prompts in qa_list:
            chats = chat(prompts, image_paths, max_length, generation_config, **models)

            result["chats"].append(chats)

        # grounding generate
        # input phrase in second answer
        phrase = ""
        chats.append(
            {
                "role": "user",
                "content": f"Provide the bounding box coordinate of the region this phrase describes: {phrase}",
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
            generate_config=generation_config,
            **models,
        )

        # box check
        # model_answer_bbox = norm2origin_box([63,58,95,100], width, height, resize_type)
        # calculate iou, (gt_box,predict_box)
        # iou = calculate_iou([1704,1669,2543,2633], [2010,2011,2290,2291])

        chats.append(chat_output)
        result["chats"] = list()
        result["chats"].append(chats)

        results.append(result)

    save_path = os.path.join(save_dir, f"results_{now}.json")
    os.makedirs(save_dir, exist_ok=True)
    save_json(results, save_path)
