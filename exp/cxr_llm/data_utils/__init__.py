from torch.utils.data.dataset import ConcatDataset

from external.honeybee.utils import print_rank_0

from .datasets import load_dataset
from .multidata_wrapper import MultiDataset


def load_datasets(dataset_cfg, tokenizer, processors, train, inference):
    datasets = {}
    if train:
        datasets["train"] = datasets_provider(
            dataset_cfg,
            tokenizer=tokenizer,
            split="train",
            processors=processors,
        )
        datasets["eval"] = datasets_provider(
            dataset_cfg,
            tokenizer=tokenizer,
            split="eval",
            processors=processors,
        )

    if inference:
        datasets["test"] = datasets_provider(
            dataset_cfg,
            tokenizer=tokenizer,
            split="test",
            processors=processors,
        )
    return datasets


def datasets_provider(data_config, tokenizer, processors, split="train"):
    print_rank_0(f"> building {split} datasets for MLLM ...")

    # load datasets
    data_cfgs = data_config.get(f"{split}_dataset", None)
    if data_cfgs is not None:
        common_cfgs = data_config["common_config"]

        # if data_config.template is not set or data_config.template.name is None,
        # datasets do not use templatizer.
        datasets_lst = [
            load_dataset(
                dset_name,
                tokenizer,
                processors,
                data_root=data_config["data_root"],
                split=split,
                **common_cfgs,
                **data_config[dset_name],
            )
            for dset_name in data_cfgs
        ]

        if (
            len(datasets_lst) > 1
            and split == "train"
            and data_config.get("concat_train_dataset", None)
        ):
            print_rank_0(
                f"> Wrapping ConcatDataset, split : {split}... (#dataset={len(datasets_lst)})"
            )
            print_rank_0(
                "> Training without sampling weights.. Please check concat_train_dataset setting."
            )
            # wrap with ConcatDataset class
            dataset = ConcatDataset(datasets_lst)

        elif len(datasets_lst) > 1 and split == "train":
            print_rank_0(
                f"> Wrapping Multidataset, split : {split}... (#dataset={len(datasets_lst)})"
            )
            # wrap with Multidataset class
            dataset = MultiDataset(datasets_lst, **common_cfgs)

        elif len(datasets_lst) > 1 and split == "eval":
            print_rank_0(
                f"> Wrapping Dict, split : {split} ... (#dataset={len(datasets_lst)})"
            )

            # wrap with dict to evaluate separately
            dataset = {}
            for dset in datasets_lst:
                dataset[dset.dset_name] = dset

        elif len(datasets_lst) > 1 and split == "test":
            print_rank_0(
                f"> Wrapping ConcatDataset, split : {split} ... (#dataset={len(datasets_lst)})"
            )
            dataset = ConcatDataset(datasets_lst)
        else:
            print_rank_0("> Single dataset ...")
            dataset = datasets_lst[0]
        print_rank_0("> finished creating MLLM datasets ...")
    else:
        dataset = None

    return dataset
