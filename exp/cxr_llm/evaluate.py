import csv
import os
import re
import traceback

import numpy as np
import torch
from data_utils.utils import clean_report_mimic_cxr
from nltk.tokenize import wordpunct_tokenize
from pycocoevalcap.bleu.bleu import Bleu
from tqdm import tqdm

from common.utils import load_json, save_evaluation_result
from exp.cxr_llm.metrics import (
    clinical_efficacy,
    compute_clinical_efficacy_metrics,
    nlp_metrics,
)
from external.chexbert.f1chexbert_hf import CONDITIONS, CheXbertLabeler


def remove_special_tokens(
    sentence: str,
    tokens_list: list = [
        # mistral
        "<s>",
        "</s>",
        "[INST]",
        "[/INST]",
    ],
) -> str:
    for token in tokens_list:
        sentence = sentence.replace(token, "")
    return sentence.strip()


def compute_report_generation_metrics(
    pred_path,
    gt_path,
    task_prompt="provide a description of the findings in the radiology image",
    exclude_prompt=None,
    exclude_lateral=False,
    json_file=None,
):

    pred_json = load_json(pred_path)
    gt_json = load_json(gt_path)

    # remove lateral data
    if exclude_lateral:
        frontal_list = []
        for data in gt_json:
            if data["view_position"] in ["PA", "AP"]:
                frontal_list.append(data)
        gt_json = frontal_list

    # json to dict
    pred_dict = {}
    for data in pred_json:

        # multi-turn inference file
        for i, prompt in enumerate(data["prompts"]):
            if task_prompt in prompt and (
                exclude_prompt is None or exclude_prompt not in prompt
            ):
                image_file = (
                    data["image"][0] if type(data["image"]) is list else data["image"]
                )
                key = os.path.basename(image_file)
                assert (
                    key not in pred_dict
                ), "Too many questions for single image. Please check task_prompt or split json files"
                pred_dict[key] = data["outputs"][i]

    # pred, gt list
    pred_list = []
    gt_list = []
    nlp_gt_list = []
    for data in gt_json:
        key = os.path.basename(data["dicom_id"])

        assert key in pred_dict, f"{key} prediction result doesn't exist"

        pred_report = remove_special_tokens(pred_dict[key])
        pred_list.append(pred_report)

        gt_list.append(data["findings"])
        gt_report = clean_report_mimic_cxr(data["findings"])
        nlp_gt_list.append(gt_report)

    # save csv files for RadGraph F1
    if json_file is not None:
        csv_dir = "csv_frontal" if exclude_lateral else "csv"
        csv_dir = os.path.join(os.path.dirname(json_file), csv_dir)
        os.makedirs(csv_dir, exist_ok=True)

        # gt save
        csv_file = os.path.join(csv_dir, "gt.csv")
        with open(csv_file, "w") as f:
            wr = csv.writer(f)
            wr.writerow(["study_id", "report"])
            for i, report in enumerate(nlp_gt_list):
                wr.writerow([i, report.lower()])

        # pred save
        csv_file = os.path.join(csv_dir, "pred.csv")
        with open(csv_file, "w") as f:
            wr = csv.writer(f)
            wr.writerow(["study_id", "report"])
            for i, report in enumerate(pred_list):
                wr.writerow([i, report.lower()])

    result = {}

    # clinical efficacy
    labeler_model = CheXbertLabeler()
    ce_result = clinical_efficacy(pred_list, gt_list, labeler_model)
    result["clinical_efficacy"] = ce_result

    # nlp metrics
    nlp_result = nlp_metrics(pred_list, nlp_gt_list)
    result["nlp_metrics"] = nlp_result

    if json_file:
        save_evaluation_result(
            result,
            json_file,
            "MIMIC-CXR",
        )

    return result


def compute_vqa_metrics(pred_path, json_file=None, dataset_name="mimic-cxr-vqa"):

    pred_json = load_json(pred_path)

    # json to dict
    gt_list = [data["labels"][0] for data in pred_json]
    pred_list = [data["outputs"][0] for data in pred_json]

    # pred, gt list
    filtered_gt_list = []
    filtered_pred_list = []
    for gt_answer, pred_answer in zip(gt_list, pred_list):
        gt_text = remove_special_tokens(gt_answer).replace(".", "").lower()
        filtered_gt_list.append(gt_text)

        pred_text = remove_special_tokens(pred_answer).replace(".", "").lower()
        filtered_pred_list.append(pred_text)

    result = {}

    # accuracy
    assert len(filtered_pred_list) == len(filtered_gt_list)

    cnt = 0
    for gt_answer, pred_answer in zip(filtered_gt_list, filtered_pred_list):
        if gt_answer == pred_answer:
            cnt += 1

    accuracy = cnt / len(filtered_gt_list)
    result["accuracy"] = accuracy

    # BLEU-1 (macro average)

    def bleu_1(gt_answer, pred_answer):
        def postprocess(text):
            return " ".join(wordpunct_tokenize(text))

        gts = {0: [postprocess(gt_answer)]}
        preds = {0: [postprocess(pred_answer)]}

        score, _ = Bleu(1).compute_score(gts, preds, verbose=0)
        return score

    bleu_1_list = []

    for gt_answer, pred_answer in zip(filtered_gt_list, filtered_pred_list):
        bleu_1_list.append(bleu_1(gt_answer, pred_answer)[0])

    # average score on all test samples
    result["bleu_1"] = np.mean(bleu_1_list)

    # Recall

    def calculate_recall(ground_truth_tokens, generated_tokens):
        ground_truth_set = set(ground_truth_tokens)
        generated_set = set(generated_tokens)
        matched_tokens = ground_truth_set.intersection(generated_set)
        recall = len(matched_tokens) / len(ground_truth_set)
        return recall

    recall_list = []
    for gt_answer, pred_answer in zip(filtered_gt_list, filtered_pred_list):
        gt_tokens = wordpunct_tokenize(gt_answer)
        pred_tokens = wordpunct_tokenize(pred_answer)
        recall_list.append(calculate_recall(gt_tokens, pred_tokens))

    result["recall"] = np.mean(recall_list)

    if json_file:
        save_evaluation_result(
            result,
            json_file,
            dataset_name,
        )

    return result


def compute_multi_cls_metrics(
    pred_path,
    gt_path,
    task_prompt="Which of the following findings are present in the radiology image?",
    exclude_prompt=None,
    json_file=None,
):

    labeler_class = CONDITIONS

    pred_json = load_json(pred_path)
    gt_json = load_json(gt_path)

    # json to dict
    pred_dict = {}
    for data in pred_json:

        # multi-turn inference file
        for i, prompt in enumerate(data["prompts"]):
            if task_prompt in prompt and (
                exclude_prompt is None or exclude_prompt not in prompt
            ):
                # TODO: multi-image
                image_file = (
                    data["image"][0] if type(data["image"]) is list else data["image"]
                )
                key = os.path.basename(image_file)
                assert (
                    key not in pred_dict
                ), "Too many questions for single image. Please check task_prompt or split json files"
                pred_dict[key] = data["outputs"][i]

    # pred, gt list matching
    pred_list = []
    gt_list = []
    for data in gt_json:
        key = os.path.basename(data["dicom_id"])

        assert key in pred_dict, f"{key} prediction result doesn't exist"

        pred_list.append(pred_dict[key])
        gt_list.append(data["chexbert"])

    # make gt label list
    gt_labels = []
    for gt in gt_list:
        # positive 1, uncertain, blank, negative 0
        gt_labels.append([int(i == 1) for i in gt])

    # make pred label list
    pred_labels = []

    for pred in pred_list:
        label_container = [None] * len(labeler_class)

        # find label names (without no finding)
        for index, name in enumerate(labeler_class[:-1]):
            if name.lower() in pred:
                label_container[index] = 1
            else:
                label_container[index] = 0

        # make no finding prediction label
        # exclude no finding, support devices
        if sum(label_container[:-2]) >= 1:
            label_container[-1] = 0
        else:
            label_container[-1] = 1

        pred_labels.append(label_container)

    # list to tensor
    pred_labels, gt_labels = torch.LongTensor(pred_labels), torch.LongTensor(gt_labels)

    result = compute_clinical_efficacy_metrics(
        label_class=labeler_class, pred_labels=pred_labels, gt_labels=gt_labels
    )

    result_dict = {"clinical_efficacy": result}
    if json_file:
        save_evaluation_result(
            result_dict,
            json_file,
            "MIMIC-CXR",
        )

    return result_dict


def compute_grounding_metrics(
    pred_path,
    task_prompt="Provide the bounding box coordinate",
    json_file=None,
):
    pred_json = load_json(pred_path)

    ious = list()
    for data in pred_json:
        for i, prompt in enumerate(data["prompts"]):
            if task_prompt in prompt:
                gt_bbox = remove_special_tokens(data["labels"][i])
                pred_bbox = remove_special_tokens(data["outputs"][i])

                # check bounding box format
                if len(pred_bbox.split(",")) != 4:
                    iou = 0
                else:
                    b1_x1, b1_y1, b1_x2, b1_y2 = (
                        int(s) for s in re.sub(r"[\[\].]", "", gt_bbox).split(",")
                    )
                    b2_x1, b2_y1, b2_x2, b2_y2 = (
                        int(s) for s in re.sub(r"[\[\].]", "", pred_bbox).split(",")
                    )

                    inter_x1 = max(b1_x1, b2_x1)
                    inter_y1 = max(b1_y1, b2_y1)
                    inter_x2 = min(b1_x2, b2_x2)
                    inter_y2 = min(b1_y2, b2_y2)

                    intersection = max(inter_x2 - inter_x1, 0) * max(
                        inter_y2 - inter_y1, 0
                    )
                    b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
                    b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
                    union = b1_area + b2_area - intersection
                    iou = intersection / (union + 1e-16)
                ious.append(iou)

    result = {
        "mIoU": np.mean(ious),
        "Acc": len([iou for iou in ious if iou >= 0.5]) / len(ious),
    }

    if json_file:
        save_evaluation_result(
            result,
            json_file,
            "MS-CXR",
        )

    return result


def compute_all_metrics(gt_path, pred_path, json_prefix, output_dir, logger):

    try:
        logger.info("try compute multi_cls metric with MIMIC-CXR test dataset")
        compute_multi_cls_metrics(
            pred_path=pred_path,
            gt_path=gt_path,
            json_file=os.path.join(
                output_dir,
                f"{json_prefix}_multi_cls.json",
            ),
        )
    except Exception:
        logger.info("compute multi_cls metric error")
        logger.info(traceback.format_exc())

    try:
        logger.info("try compute report generation metric with MIMIC-CXR test dataset")

        compute_report_generation_metrics(
            pred_path=pred_path,
            gt_path=gt_path,
            json_file=os.path.join(
                output_dir,
                f"{json_prefix}_report.json",
            ),
        )
    except Exception:
        logger.info("compute report generation metric error")
        logger.info(traceback.format_exc())

    try:
        logger.info(
            "try compute report generation metric with MIMIC-CXR test dataset (PA & AP)"
        )

        compute_report_generation_metrics(
            pred_path=pred_path,
            gt_path=gt_path,
            exclude_lateral=True,
            json_file=os.path.join(
                output_dir,
                f"{json_prefix}_report_PA_AP.json",
            ),
        )
    except Exception:
        logger.info("compute report generation metric error")
        logger.info(traceback.format_exc())

    try:
        logger.info("try compute grounding metric with MS-CXR test dataset")
        compute_grounding_metrics(
            pred_path=pred_path,
            json_file=os.path.join(
                output_dir,
                f"{json_prefix}_p_grounding.json",
            ),
        )
    except Exception:
        logger.info("compute grounding metric error")
        logger.info(traceback.format_exc())


def compute_report_generation_metrics_multi_image(
    pred_path,
    gt_path,
    output_mode="cot",
    json_file=None,
):
    # gpt_mimic_frontal_lateral

    gt_data = load_json(gt_path)
    fl_data = load_json(pred_path)

    gt_fl_report = []
    fl_report = []

    for gt_dict in tqdm(gt_data):
        if "PA" == gt_dict["view_position"] or "AP" == gt_dict["view_position"]:
            for fl_dict in fl_data:
                fl_imgs = [os.path.basename(fl_img) for fl_img in fl_dict["image"]]
                if gt_dict["dicom_id"] in fl_imgs:
                    gt_fl_report.append(gt_dict["findings"])
                    if output_mode == "mrg":
                        fl_report.append(fl_dict["outputs"][0])  # 2461
                    elif output_mode == "cot":
                        fl_report.append(fl_dict["outputs"][1])  # 2461
                    break

    pred_list = []
    gt_list = []
    for gt_text, pred_text in zip(gt_fl_report, fl_report):
        pred_report = remove_special_tokens(pred_text)
        pred_list.append(pred_report)
        gt_list.append(gt_text)

    result = {}

    # clinical efficacy
    labeler_model = CheXbertLabeler()
    ce_result = clinical_efficacy(pred_list, gt_list, labeler_model)
    result["clinical_efficacy"] = ce_result

    save_evaluation_result(
        result,
        json_file,
        "MIMIC-CXR",
    )


def compute_report_generation_metrics_multi_study(
    pred_path,
    gt_path,
    fl_path,
    output_mode="cot",
    json_file=None,
):
    # gpt_mimic_history_multi
    gt_data = load_json(gt_path)
    hs_data = load_json(pred_path)
    fl_data = load_json(fl_path)

    gt_hs_report = []
    hs_report = []
    for gt_dict in tqdm(gt_data):
        match_bool = False
        if "PA" == gt_dict["view_position"] or "AP" == gt_dict["view_position"]:
            for hs_dict in hs_data:
                hs_prompt = hs_dict["prompts"][0]
                assert hs_prompt.count("<image>") == len(hs_dict["image"])
                current_prompt = hs_prompt.split("current radiology images: ")[-1]
                current_token_num = current_prompt.count("<image>")

                hs_imgs = [os.path.basename(hs_img) for hs_img in hs_dict["image"]]
                if gt_dict["dicom_id"] in hs_imgs[-1 * current_token_num :]:
                    gt_hs_report.append(gt_dict["findings"])
                    if output_mode == "mrg":
                        hs_report.append(hs_dict["outputs"][0])
                    elif output_mode == "cot":
                        hs_report.append(hs_dict["outputs"][1])
                    match_bool = True
                    break
            if match_bool is False:
                for fl_dict in fl_data:
                    fl_imgs = [os.path.basename(fl_img) for fl_img in fl_dict["image"]]
                    if gt_dict["dicom_id"] in fl_imgs:
                        gt_hs_report.append(gt_dict["findings"])
                        if output_mode == "mrg":
                            hs_report.append(fl_dict["outputs"][0])
                        elif output_mode == "cot":
                            hs_report.append(fl_dict["outputs"][1])

    pred_list = []
    gt_list = []
    for gt_text, pred_text in zip(gt_hs_report, hs_report):
        pred_report = remove_special_tokens(pred_text)
        pred_list.append(pred_report)
        gt_list.append(gt_text)

    result = {}

    # clinical efficacy
    labeler_model = CheXbertLabeler()
    ce_result = clinical_efficacy(pred_list, gt_list, labeler_model)
    result["clinical_efficacy"] = ce_result

    save_evaluation_result(
        result,
        json_file,
        "MIMIC-CXR",
    )


if __name__ == "__main__":
    # multi_cls
    pred_path = "checkpoint-40000.json"
    gt_path = "datasets/MIMIC-CXR/preprocess/v9.2/dicom_level/test.json"
    results = compute_multi_cls_metrics(
        pred_path=pred_path,
        gt_path=gt_path,
        json_file=os.path.join(os.path.dirname(pred_path), "multi_cls_new.json"),
    )

    # report generation (text)
    results = compute_report_generation_metrics(
        pred_path,
        gt_path,
        json_file=os.path.join(
            os.path.dirname(pred_path), "report_generation_all_view_new.json"
        ),
    )

    # report generation (text, PA&AP only)
    results = compute_report_generation_metrics(
        pred_path,
        gt_path,
        exclude_lateral=True,
        json_file=os.path.join(
            os.path.dirname(pred_path), "report_generation_frontal_new.json"
        ),
    )

    # # phrase grounding
    # pred_path = "checkpoint-240.json"
    # results = compute_grounding_metrics(pred_path)

    # # vqa
    # pred_path = "checkpoint-40000.json"
    # results = compute_vqa_metrics(pred_path)

    # gt_path = "datasets/MIMIC-CXR/preprocess/v9.2/dicom_level/test.json"
    # multi_image_path = "checkpoint-40000_multi_image.json"
    # compute_report_generation_metrics_multi_image(
    #     multi_image_path,
    #     gt_path,
    #     output_mode="cot",
    #     json_file=os.path.join(os.path.dirname(multi_image_path), "multi_image.json"),
    # )

    # multi_study_path = "checkpoint-40000_multi_study.json"
    # compute_report_generation_metrics_multi_study(
    #     multi_study_path,
    #     gt_path,
    #     multi_image_path,
    #     output_mode="cot",
    #     json_file=os.path.join(os.path.dirname(multi_study_path), "multi_study.json"),
    # )
