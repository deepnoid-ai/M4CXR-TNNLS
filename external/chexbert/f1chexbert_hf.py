import pandas as pd
import torch
from f1chexbert.f1chexbert import F1CheXbert, generate_attention_masks, tokenize
from tqdm import tqdm

# CONDITIONS is a list of all 14 medical observations
CONDITIONS = [
    "Enlarged Cardiomediastinum",
    "Cardiomegaly",
    "Lung Opacity",
    "Lung Lesion",
    "Edema",
    "Consolidation",
    "Pneumonia",
    "Atelectasis",
    "Pneumothorax",
    "Pleural Effusion",
    "Pleural Other",
    "Fracture",
    "Support Devices",
    "No Finding",
]
CLASS_MAPPING = {0: "Blank", 1: "Positive", 2: "Negative", 3: "Uncertain"}


class CheXbertLabeler(F1CheXbert):
    def get_label(self, report):
        impressions = pd.Series([report])
        out = tokenize(impressions, self.tokenizer)
        batch = torch.LongTensor([o for o in out])
        src_len = [b.shape[0] for b in batch]
        attn_mask = generate_attention_masks(batch, src_len, self.device)
        out = self.model(batch.to(self.device), attn_mask)
        out = [out[j].argmax(dim=1).item() for j in range(len(out))]
        return out

    def get_label_from_list(self, report_list):
        predictions_dict = []
        predictions = []

        for report in tqdm(report_list):
            out = self.get_label(report)

            result = {}
            for i, pred in enumerate(out):
                result[CONDITIONS[i]] = CLASS_MAPPING[pred]

            predictions_dict.append(result)
            predictions.append(out)

        return predictions_dict, predictions


if __name__ == "__main__":

    chexbert = CheXbertLabeler()

    report_list = [
        "calcification is present.",
        "nodule is seen.",
        "consolidation is seen.",
        "there is no evidence for the presence of pleural effusion.",
        "pneumothorax",
    ]

    chexbert.get_label_from_list(report_list)
