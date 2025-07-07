from collections import defaultdict

import numpy as np
import torch
from nltk.tokenize import wordpunct_tokenize
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from torchmetrics.functional.classification import (
    accuracy,
    binary_jaccard_index,
    f1_score,
    precision,
    recall,
    specificity,
)

from external.chexbert.f1chexbert_hf import CONDITIONS


def report_texts_to_labels(
    pred_list,
    gt_list,
    model,
):
    label_class = CONDITIONS
    # CheXbert:  {0: "Blank", 1: "Positive", 2: "Negative", 3: "Uncertain"}
    pred_labels_output = torch.LongTensor(model.get_label_from_list(pred_list)[1])
    gt_labels_output = torch.LongTensor(model.get_label_from_list(gt_list)[1])

    # Positive 1, rest 0
    pred_labels = (pred_labels_output == 1).long()
    gt_labels = (gt_labels_output == 1).long()

    # Positive, uncertain 1, rest 0
    pred_labels_uncertain = (
        (pred_labels_output == 1) + (pred_labels_output == 3)
    ).long()
    gt_labels_uncertain = ((gt_labels_output == 1) + (gt_labels_output == 3)).long()

    return (
        label_class,
        pred_labels,
        gt_labels,
        pred_labels_uncertain,
        gt_labels_uncertain,
    )


def clinical_efficacy(
    pred_list,
    gt_list,
    model,
):
    label_class, pred_labels, gt_labels, pred_labels_uncertain, gt_labels_uncertain = (
        report_texts_to_labels(pred_list, gt_list, model)
    )

    result = compute_clinical_efficacy_metrics(
        label_class, pred_labels, gt_labels, pred_labels_uncertain, gt_labels_uncertain
    )
    return result


def compute_clinical_efficacy_metrics(
    label_class,
    pred_labels,
    gt_labels,
    pred_labels_uncertain=None,
    gt_labels_uncertain=None,
):
    # compute classification metrics
    result = defaultdict(dict)
    binary_classification_metrics = [
        "f1_score",
        "recall",
        "precision",
        "specificity",
        "accuracy",
    ]

    # metric with 14 observations
    # uncertain negative
    metric_result = compute_report_classification_metrics(
        pred_labels,
        gt_labels,
        label_class,
        binary_classification_metrics,
    )
    result.update(metric_result)

    # only for CheXbert
    # uncertain positive
    if pred_labels_uncertain is not None and gt_labels_uncertain is not None:

        metric_result = compute_report_classification_metrics(
            pred_labels_uncertain,
            gt_labels_uncertain,
            label_class,
            binary_classification_metrics,
        )
        for k, v in metric_result.items():
            if "micro" in k or "macro" in k:
                result[k + "_uncertain"] = v

    return result


def compute_report_classification_metrics(
    pred_labels,
    gt_labels,
    label_class,
    binary_classification_metrics,
):
    result = defaultdict(dict)

    for i, label_name in enumerate(label_class):
        for cls_metric in binary_classification_metrics:
            result[label_name][cls_metric] = eval(cls_metric)(
                pred_labels[:, i], gt_labels[:, i], "binary"
            ).item()

    # macro average
    for cls_metric in binary_classification_metrics:
        scores = list()
        for i, label_name in enumerate(label_class):
            scores.append(result[label_name][cls_metric])
        result["macro_avg"][cls_metric] = np.mean(scores)

    # micro average
    for cls_metric in binary_classification_metrics:

        pred = pred_labels.flatten()
        gt = gt_labels.flatten()
        result["micro_avg"][cls_metric] = eval(cls_metric)(pred, gt, "binary").item()

    # example based average
    for cls_metric in binary_classification_metrics:
        example_result = list()
        for pred, gt in zip(pred_labels, gt_labels):
            example_result.append(eval(cls_metric)(pred, gt, "binary").item())
        result["example_based_avg"][cls_metric] = np.mean(example_result)

    # 5 observations
    # only for CheXbert
    observation_5 = [
        "Atelectasis",
        "Cardiomegaly",
        "Consolidation",
        "Edema",
        "Pleural Effusion",
    ]

    pred_list_5 = []
    gt_list_5 = []
    for i, label_name in enumerate(label_class):
        if label_name in observation_5:
            pred_list_5.append(pred_labels[:, i])
            gt_list_5.append(gt_labels[:, i])

    pred_list_5 = torch.cat(pred_list_5)
    gt_list_5 = torch.cat(gt_list_5)

    # micro average (5-obs)
    for cls_metric in binary_classification_metrics:
        result["micro_avg_5_obs"][cls_metric] = eval(cls_metric)(
            pred_list_5, gt_list_5, "binary"
        ).item()

    # macro average (5-obs)
    scores = list()
    for cls_metric in binary_classification_metrics:
        for i, label_name in enumerate(label_class):
            if label_name in observation_5:
                scores.append(result[label_name][cls_metric])
        result["macro_avg_5_obs"][cls_metric] = np.mean(scores)

    return result


def nlp_metrics(pred_list, gt_list):
    """
    pycocoevalcap NLP metrics
    (BLEU, Meteor, Rouge)

    Args:
        pred_list (List[str]): list of prediction text
        gt_list (List[str]): list of ground truth text

    Returns:
        result (dict): dictionary of metric
    """

    def postprocess(text):
        return " ".join(wordpunct_tokenize(text.lower()))

    result = dict()
    gts = {k: [postprocess(v)] for k, v in enumerate(gt_list)}
    res = {k: [postprocess(v)] for k, v in enumerate(pred_list)}
    scorers = [
        (Bleu(4), ["BLEU_1", "BLEU_2", "BLEU_3", "BLEU_4"]),
        (Meteor(), "METEOR"),
        (Rouge(), "ROUGE_L"),
    ]
    # Compute score for each metric
    for scorer, method in scorers:
        print("computing %s score..." % (scorer.method()))
        try:
            score, scores = scorer.compute_score(gts, res, verbose=0)
        except TypeError:
            score, scores = scorer.compute_score(gts, res)
        if type(method) is list:
            for sc, m in zip(score, method):
                result[m] = sc
        else:
            result[method] = score
    return result
