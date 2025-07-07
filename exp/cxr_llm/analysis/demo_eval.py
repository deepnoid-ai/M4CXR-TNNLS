import os

import torch

from common.utils import load_json, save_json
from exp.cxr_llm.demo import chat, load_models
from external.chexbert.chexbert_labeler import CHEXBERT_CLASS

# chexbert findings
findings = "Enlarged Cardiomediastinum, Cardiomegaly, Lung Opacity, Lung Lesion, Edema, Consolidation, Pneumonia, Atelectasis, Pneumothorax, Pleural Effusion, Pleural Other, Fracture, Support Devices"


def question_after_cot():
    prompt_list = []

    # easy language
    prompt_list.append(
        [
            f"radiology image: <image> Which of the following findings are present in the radiology image? Findings: {findings}",
            "Based on the previous conversation, provide a description of the findings in the radiology image.",
            "Explain the description with easy language.",
        ]
    )

    # concise summary
    prompt_list.append(
        [
            f"radiology image: <image> Which of the following findings are present in the radiology image? Findings: {findings}",
            "Based on the previous conversation, provide a description of the findings in the radiology image.",
            "Summarize the description concisely.",
        ]
    )

    # three concise points
    prompt_list.append(
        [
            f"radiology image: <image> Which of the following findings are present in the radiology image? Findings: {findings}",
            "Based on the previous conversation, provide a description of the findings in the radiology image.",
            "Can you summarize the description in three concise points?",
        ]
    )

    # summary
    prompt_list.append(
        [
            f"radiology image: <image> Which of the following findings are present in the radiology image? Findings: {findings}",
            "Based on the previous conversation, provide a description of the findings in the radiology image.",
            "Provide a short summary of the most important lesion in this chest x-ray report.",
        ]
    )

    # summary in one sentence
    prompt_list.append(
        [
            f"radiology image: <image> Which of the following findings are present in the radiology image? Findings: {findings}",
            "Based on the previous conversation, provide a description of the findings in the radiology image.",
            "Summarize the description in one concise sentence.",
        ]
    )

    return prompt_list


if __name__ == "__main__":
    import datetime

    data_root = "datasets"
    save_dir = "qualitative_eval"

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
    generation_config = {"max_new_tokens": 512, "do_sample": True}

    # test dataset
    test_data = load_json(
        os.path.join(data_root, "MIMIC-CXR/preprocess/v8.0/official_testset.json")
    )

    num_test_samples = 2
    results = []

    for data in test_data[:num_test_samples]:
        image_paths = [os.path.join(data_root, "MIMIC-CXR", "images", data["dicom_id"])]
        result = {}
        result["input"] = {
            "study_id": data["study_id"],
            "dicom_id": data["dicom_id"],
            "subject_id": data["subject_id"],
            "view_position": data["view_position"],
            "findings": data["findings"],
            "chexbert": [
                CHEXBERT_CLASS[i]
                for i, label in enumerate(data["chexbert"])
                if label == 1
            ],
        }
        result["generation_config"] = generation_config
        result["chats"] = list()

        qa_list = question_after_cot()
        for prompts in qa_list:
            chats = chat(prompts, image_paths, max_length, generation_config, **models)
            result["chats"].append(chats)
        results.append(result)

    save_path = os.path.join(save_dir, f"results_{now}.json")
    os.makedirs(save_dir, exist_ok=True)
    save_json(results, save_path)
