import argparse
import os
import random
from functools import partial

from data_utils import load_datasets
from models import apply_params_setting, load_mllm
from models.tokenization_mllm import build_mllm_tokenizer
from torch.distributed.elastic.multiprocessing.errors import record
from transformers import EarlyStoppingCallback, TrainingArguments

from common.code_snapshot import code_snapshot
from common.trainer import logger
from common.utils import Config, output_directory_setting, str2bool
from exp.cxr_llm.data_utils.collate import batchify
from exp.cxr_llm.evaluate import compute_all_metrics
from exp.cxr_llm.trainer import CXRLLMTrainer


def parse_args():
    parser = argparse.ArgumentParser(description="CXR LLM Run")
    parser.add_argument(
        "--cfg-path",
        default=os.path.join(os.path.dirname(__file__), "config.yaml"),
        help="path to configuration file.",
    )
    parser.add_argument(
        "--add_cfg_list",
        default=["all_datasets", "mllm", "gh"],
        type=str,
        nargs="+",
        help="List of YAML files. The cfg will be overwritten in the given order.",
    )
    parser.add_argument(
        "--user",
        type=str,
        default=None,
        help="User. This argument, when provided, overrides any previous settings.",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Name of experiment. This argument, when provided, overrides any previous settings.",
    )
    parser.add_argument(
        "--train",
        default=True,
        type=str2bool,
        choices=[True, False],
        help="Set the train mode to True or False",
    )
    parser.add_argument(
        "--inference",
        default=True,
        type=str2bool,
        choices=[True, False],
        help="Set the inference mode to True or False",
    )
    parser.add_argument(
        "--compute_metric",
        default=True,
        type=str2bool,
        choices=[True, False],
        help="Compute metrics with inference results. Set to True or False",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="directory for saving inference results",
    )
    parser.add_argument("--no_report", action="store_true", help="Skip report to wandb")
    args = parser.parse_args()

    return args


@record
def main():
    args = parse_args()
    cfg = Config(args).config

    output_directory_setting(cfg, logger)
    snapshot_dir = code_snapshot(
        save_dir=os.path.join(cfg["train"]["output_dir"], "snapshot"), cfg=cfg
    )

    deepspeed = dict(cfg["deepspeed"]) if cfg.get("deepspeed") else None

    # resume from checkpoint with different random seed
    if (
        cfg["train"].get("ignore_data_skip")
        and cfg["experiment"]["resume_from_checkpoint"]
    ):
        cfg["train"]["seed"] = random.randint(0, cfg["train"]["seed"] - 1)
    training_args = TrainingArguments(**cfg["train"], deepspeed=deepspeed)

    # check metric_for_best_model
    if cfg["args"]["train"]:
        assert (
            cfg["train"]["metric_for_best_model"] == "eval_loss"
            and len(cfg["dataset"]["eval_dataset"]) == 1
        ) or (
            cfg["train"]["metric_for_best_model"]
            in ["eval_" + k + "_loss" for k in cfg["dataset"]["eval_dataset"]]
        ), "Please set 'metric_for_best_model' with one of eval_dataset names : eval_{dataset name}_loss"

    logger.info("model load")
    models = load_mllm(cfg["model"])
    models["model"] = apply_params_setting(
        cfg["model"], models["model"], cfg["train"]["output_dir"]
    )

    models["tokenizer"] = build_mllm_tokenizer(
        models["tokenizer"], **cfg["model"]["tokenizer_cfg"]
    )

    logger.info("datasets load")
    datasets = load_datasets(
        cfg["dataset"],
        tokenizer=models["tokenizer"],
        processors=models["image_processor"],
        train=cfg["args"]["train"],
        inference=cfg["args"]["inference"],
    )

    logger.info("construct trainer")

    _batchify = partial(
        batchify,
        tokenizer=models["tokenizer"],
        max_length=cfg["dataset"]["common_config"]["max_length"],
        use_trunc=True,
    )

    trainer = CXRLLMTrainer(
        model=models["model"],
        args=training_args,
        train_dataset=datasets.get("train"),
        eval_dataset=datasets.get("eval"),
        tokenizer=models["tokenizer"],
        data_collator=_batchify,
        cfg=cfg,
        # compute_metrics=compute_metrics,
        processor=models["image_processor"],
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=cfg["experiment"]["early_stopping_patience"]
            )
        ],
    )

    # run train
    if cfg["args"]["train"]:
        logger.info("run train")
        trainer.train(
            resume_from_checkpoint=cfg["experiment"]["resume_from_checkpoint"]
        )

    # run inference
    if cfg["args"]["inference"]:
        logger.info("run inference")

        # results save directory
        if cfg["args"]["output_dir"] is None:
            output_dir = os.path.join(
                cfg["train"]["output_dir"], "inference", os.path.basename(snapshot_dir)
            )
        else:
            output_dir = os.path.join(
                cfg["args"]["output_dir"], os.path.basename(snapshot_dir)
            )

        # checkpoint path
        if trainer.state.best_model_checkpoint is not None:
            checkpoint = trainer.state.best_model_checkpoint
        else:
            checkpoint = cfg["experiment"]["resume_from_checkpoint"]

        inference_outputs = trainer.inference(
            test_dataset=datasets.get("test"),
            output_dir=output_dir,
            resume_from_checkpoint=checkpoint,
        )

        # compute metrics (with main process)
        if cfg["args"]["compute_metric"] and trainer.is_world_process_zero():
            gt_path = os.path.join(
                cfg["dataset"]["data_root"],
                cfg["dataset"]["mimiccxr_single_image_cot_prompting"]["test"],
            )
            pred_path = inference_outputs.results["save_path"]
            json_prefix = (
                f"{cfg['experiment']['user']}_{cfg['experiment']['name']}".replace(
                    "/", "_"
                )
            )

            compute_all_metrics(
                gt_path=gt_path,
                pred_path=pred_path,
                json_prefix=json_prefix,
                output_dir=output_dir,
                logger=logger,
            )


if __name__ == "__main__":
    main()
