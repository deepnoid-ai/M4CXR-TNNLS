import argparse
import json
import logging
import os
import pickle
import sys
from datetime import datetime
from typing import List

import numpy as np
import torch
import torch.distributed as dist
from omegaconf import OmegaConf


def now():
    from datetime import datetime

    return datetime.now().strftime("%Y%m%d%H%M")[:-1]


def update_nested_dict(original, updates):
    for key, value in updates.items():
        if key in original:
            if isinstance(value, dict) and isinstance(original[key], dict):
                update_nested_dict(original[key], value)
            else:
                original[key] = value
        else:
            original[key] = value


class Config:
    def __init__(self, args):
        self.config = OmegaConf.load(args.cfg_path)

        # convert to dict
        self.config = OmegaConf.to_container(self.config, resolve=True)

        # apply additional cfg list
        for cfg_name in args.add_cfg_list:
            add_cfg_path = os.path.join(
                os.path.dirname(args.cfg_path), "configs", cfg_name
            )

            # add .yaml
            if not cfg_name.endswith(".yaml"):
                add_cfg_path += ".yaml"

            # config load
            add_cfg = OmegaConf.load(add_cfg_path)
            add_cfg = OmegaConf.to_container(add_cfg, resolve=True)

            update_nested_dict(self.config, add_cfg)

        self.config.update({"args": vars(args)})

        # override user, name with args
        if self.config["args"].get("user"):
            self.config["experiment"]["user"] = self.config["args"].get("user")

        if self.config["args"].get("name"):
            self.config["experiment"]["name"] = self.config["args"].get("name")


class CustomPrefixFilter(logging.Filter):
    def filter(self, record):
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_level = record.levelname
        record.msg = f"[{current_time} {log_level}] {record.msg}"
        return True


class MainProcessFilter(logging.Filter):
    def filter(self, record):
        if dist.is_initialized():
            return dist.get_rank() == 0
        else:
            return True


def load_logger(logger=None, level=logging.INFO):
    if logger is None:
        logger = logging.getLogger("myLogger")

    logger.setLevel(level)
    logger.propagate = False

    # Set up a handler to output logs to the console
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(
        logging.Formatter("%(message)s")
    )  # Custom message format
    logger.addHandler(console_handler)

    # Add a custom filter
    logger.addFilter(CustomPrefixFilter())

    # Add a main process filter only in DDP environments
    logger.addFilter(MainProcessFilter())
    return logger


def set_logger_file(filepath, logger):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    # set logging file
    logger.addHandler(logging.FileHandler(filepath, mode="a"))

    # TODO: sys.stdout write file


def load_json(file_path, encoding="utf-8"):
    with open(file_path, "r", encoding=encoding) as f:
        return json.load(f)


def save_json(data, file_path, encoding="utf-8", indent=2):
    with open(file_path, "w", encoding=encoding) as f:
        json.dump(data, f, indent=indent)


def save_evaluation_result(result_dict, save_path, datatset_name):
    result = {f"{datatset_name}": result_dict}
    save_json(result, save_path)


def output_directory_setting(cfg, logger):
    # output directory settings
    cfg["train"]["output_dir"] = os.path.join(
        cfg["experiment"]["output_root_dir"],
        cfg["experiment"]["user"],
        cfg["experiment"]["name"],
    )
    set_logger_file(os.path.join(cfg["train"]["output_dir"], "output.log"), logger)
    logger.info(f"experiment output directory : {cfg['train']['output_dir']}")

    # skip report in debug mode
    if (cfg["args"]["no_report"]) or cfg["experiment"]["user"] == "debug":
        logger.info("skip report to wandb")
        cfg["train"]["report_to"] = "none"
    else:
        # wandb setting
        if cfg["train"]["report_to"] == "wandb":
            cfg["train"]["run_name"] = os.path.join(
                cfg["experiment"]["user"], cfg["experiment"]["name"]
            )
            os.environ["WANDB_PROJECT"] = cfg["experiment"]["project"]
            os.environ["WANDB_DIR"] = cfg["train"]["output_dir"]


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def serialize(string):
    return bytearray(pickle.dumps(string))


def deserialize(serialized_data):
    if type(serialized_data) is np.ndarray:
        serialized_data = serialized_data.tolist()
    return pickle.loads(bytes(serialized_data))


def split_with_splitter(tensor: torch.LongTensor):
    # Find the indices of values that are not -100.
    valid_indices = (tensor != -100).nonzero(as_tuple=True)[0]

    # Split the tensor based on non-continuous segments.
    splits = []
    if valid_indices.numel() == 0:
        return splits
    start = valid_indices[0]
    for i in range(1, len(valid_indices)):
        if valid_indices[i] != valid_indices[i - 1] + 1:
            splits.append(tensor[start : valid_indices[i - 1] + 1])
            start = valid_indices[i]

    # Add the last segment
    if start <= valid_indices[-1]:
        splits.append(tensor[start : valid_indices[-1] + 1])

    return splits


def concat_with_splitter(tensor_list: List[torch.LongTensor]):

    # Create a new list including -100 between tensors
    extended_tensor_list = []
    for tensor in tensor_list:
        extended_tensor_list.append(tensor)
        extended_tensor_list.append(torch.LongTensor([-100]).to(tensor.device))

    # Remove the last -100
    extended_tensor_list.pop()

    # Concatenate all tensors into one
    final_tensor = torch.cat(extended_tensor_list)
    return final_tensor
