import logging
import math
import os
import time
from abc import ABCMeta, abstractmethod
from collections import defaultdict
from typing import Dict, List, NamedTuple, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from datasets import Dataset
from transformers import Trainer
from transformers.trainer import (
    MODEL_FOR_CAUSAL_LM_MAPPING_NAMES,
    DataLoader,
    EvalLoopOutput,
    EvalPrediction,
    IterableDatasetShard,
    _is_peft_model,
    deepspeed_init,
    deepspeed_load_checkpoint,
    denumpify_detensorize,
    find_batch_size,
    get_last_checkpoint,
    has_length,
    is_sagemaker_mp_enabled,
    is_torch_xla_available,
    logger,
    nested_concat,
    nested_numpify,
    speed_metrics,
    unwrap_model,
)

from common.utils import load_logger

if is_torch_xla_available():
    import torch_xla.core.xla_model as xm


class InferenceOutput(NamedTuple):
    predictions: Union[np.ndarray, Tuple[np.ndarray]]
    label_ids: Optional[Union[np.ndarray, Tuple[np.ndarray]]]
    metrics: Optional[Dict[str, float]]
    results: Optional[dict]


# load logger and add custom functions
logger = load_logger(logger=logger, level=logging.INFO)


class BaseTrainer(Trainer, metaclass=ABCMeta):
    def __init__(
        self,
        model=None,
        args=None,
        data_collator=None,
        train_dataset=None,
        eval_dataset=None,
        tokenizer=None,
        model_init=None,
        compute_metrics=None,
        callbacks=None,
        optimizers=(None, None),
        preprocess_logits_for_metrics=None,
        cfg=None,
        **kwargs,
    ):
        self.cfg = cfg
        self.logger = logger

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            model_init=model_init,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )

    @abstractmethod
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.

        Args:
            model (nn.Module): The model used for calculating the loss.
            inputs (Dict[str, Union[torch.Tensor, List]]): Batch inputs composed through the dataloader.
            return_outputs (bool, optional): Determines whether to return only the loss value or include other outputs as well. Defaults to False.

        Returns:
            loss (torch.Tensor): A tensor containing the loss value used for model training.
            outputs (dict): A dictionary containing various outputs, including model predictions.
        """
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        outputs = model(**inputs)
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            unwrapped_model = unwrap_model(model)
            if _is_peft_model(unwrapped_model):
                model_name = unwrapped_model.base_model.model._get_name()
            else:
                model_name = unwrapped_model._get_name()
            if model_name in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                loss = self.label_smoother(outputs, labels, shift_labels=True)
            else:
                loss = self.label_smoother(outputs, labels)
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        return (loss, outputs) if return_outputs else loss

    @abstractmethod
    def inference_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, List]],
        ignore_keys: Optional[List[str]] = None,
        **kwargs,
    ):
        """
        This function defines the computations performed in each process during `trainer.inference`.
        (trainer.inference -> trainer.inference_loop -> trainer.inference_step)
        Args:
            model (nn.Module): The model used for inference.
            inputs (Dict[str, Union[torch.Tensor, List]]): Batch inputs composed through the dataloader.
            ignore_keys (Optional[List[str]], optional): A list of keys to exclude from the return dictionary. Defaults to None.

        Returns:
            outputs (Dict[str, Union[torch.Tensor]]): A dictionary containing the prediction results and any inputs that need to be logged.
                                                      The predicted results for each instance in the mini-batch should be concatenated with a value of -100 to form a single tensor (LongTensor or FloatTensor).
                                                      If you need to pass string information (inputs), use the `serialize` function from `common.utils` to convert the string into an integer before passing it.
        """

        inputs = self._prepare_inputs(inputs)
        if ignore_keys is None:
            if hasattr(self.model, "config"):
                ignore_keys = getattr(
                    self.model.config, "keys_to_ignore_at_inference", []
                )
            else:
                ignore_keys = []
        outputs = {}

        # TODO: implement inference codes here

        # remove outputs to ignore
        for k in ignore_keys:
            if k in outputs:
                outputs.pop(k)

        return outputs

    @abstractmethod
    def save_inference_outputs(
        self,
        outputs,
        output_dir: Optional[str] = None,
        checkpoint_name: Optional[str] = None,
        **kwargs,
    ):
        """
        A function that takes the outputs from inference and saves them in the desired format (e.g., JSON).

        Args:
            outputs (InferenceOutput): An `InferenceOutput` object that contains all prediction results and input information in `outputs.predictions`.
            output_dir (Optional[str], optional): The directory where the inference results will be saved. Defaults to None.
            checkpoint_name (Optional[str], optional): The name of the checkpoint used for inference. Defaults to None.

        Returns:
            result (dict): The results returned as a dictionary.
        """
        if output_dir is None:
            output_dir = os.path.join(self.args.output_dir, "inference")

        output_dir = os.path.join(output_dir, checkpoint_name)
        os.makedirs(output_dir, exist_ok=True)
        results = {}

        # TODO: save outputs

        return results

    def inference(
        self,
        test_dataset: Dataset,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "test",
        output_dir: Optional[str] = None,
        resume_from_checkpoint: Union[bool, str] = None,
        **kwargs,
    ) -> InferenceOutput:
        """
        Run prediction and returns predictions and potential metrics.

        Depending on the dataset and your use case, your test dataset may contain labels. In that case, this method
        will also return metrics, like in `evaluate()`.

        Args:
            test_dataset (`Dataset`):
                Dataset to run the predictions on. If it is an `datasets.Dataset`, columns not accepted by the
                `model.forward()` method are automatically removed. Has to implement the method `__len__`
            ignore_keys (`List[str]`, *optional*):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.
            metric_key_prefix (`str`, *optional*, defaults to `"test"`):
                An optional prefix to be used as the metrics key prefix. For example the metrics "bleu" will be named
                "test_bleu" if the prefix is "test" (default)

        <Tip>

        If your predictions or labels have different sequence length (for instance because you're doing dynamic padding
        in a token classification task) the predictions will be padded (on the right) to allow for concatenation into
        one array. The padding index is -100.

        </Tip>

        Returns: *NamedTuple* A namedtuple with the following keys:

            - predictions (`np.ndarray`): The predictions on `test_dataset`.
            - label_ids (`np.ndarray`, *optional*): The labels (if the dataset contained some).
            - metrics (`Dict[str, float]`, *optional*): The potential dictionary of metrics (if the dataset contained
              labels).
        """

        # skip resume_from_checkpoint after train (self._load_best_model())
        if self.args.load_best_model_at_end and (
            resume_from_checkpoint == self.state.best_model_checkpoint
        ):
            logger.info(
                "Since best model has already been loaded, skip resume from checkpoint"
            )
            pass

        else:
            if resume_from_checkpoint is False:
                resume_from_checkpoint = None

            # Load potential model checkpoint
            if isinstance(resume_from_checkpoint, bool) and resume_from_checkpoint:
                resume_from_checkpoint = get_last_checkpoint(self.args.output_dir)
                if resume_from_checkpoint is None:
                    raise ValueError(
                        f"No valid checkpoint found in output directory ({self.args.output_dir})"
                    )

            if resume_from_checkpoint is not None:
                if (
                    not is_sagemaker_mp_enabled()
                    and not self.is_deepspeed_enabled
                    and not self.is_fsdp_enabled
                ):
                    self._load_from_checkpoint(resume_from_checkpoint)

        # inference 결과 파일 이름을 checkpoint로 지정
        if resume_from_checkpoint:
            checkpoint_name = os.path.basename(resume_from_checkpoint)
        else:
            checkpoint_name = f"checkpoint-{self.state.global_step}"

        # memory metrics - must set up as early as possible
        self._memory_tracker.start()

        test_dataloader = self.get_test_dataloader(test_dataset)
        start_time = time.time()

        output = self.inference_loop(
            test_dataloader,
            description="Inference",
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
            resume_from_checkpoint=resume_from_checkpoint,
            **kwargs,
        )
        total_batch_size = self.args.eval_batch_size * self.args.world_size
        if f"{metric_key_prefix}_jit_compilation_time" in output.metrics:
            start_time += output.metrics[f"{metric_key_prefix}_jit_compilation_time"]
        output.metrics.update(
            speed_metrics(
                metric_key_prefix,
                start_time,
                num_samples=output.num_samples,
                num_steps=math.ceil(output.num_samples / total_batch_size),
            )
        )

        self.log(output.metrics)

        self.control = self.callback_handler.on_predict(
            self.args, self.state, self.control, output.metrics
        )

        self._memory_tracker.stop_and_update_metrics(output.metrics)

        results = {}
        # check main process and save results
        if self.is_world_process_zero():
            results = self.save_inference_outputs(
                output, output_dir, checkpoint_name, **kwargs
            )

        return InferenceOutput(
            predictions=output.predictions,
            label_ids=output.label_ids,
            metrics=output.metrics,
            results=results,
        )

    def inference_loop(
        self,
        dataloader: DataLoader,
        description: str,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "test",
        **kwargs,
    ) -> EvalLoopOutput:
        """
        Prediction/evaluation loop, shared by `Trainer.evaluate()` and `Trainer.predict()`.

        Works both with or without labels.
        """
        args = self.args

        # if eval is called w/o train, handle model prep here
        if self.is_deepspeed_enabled and self.deepspeed is None:
            self.optimizer, self.lr_scheduler = deepspeed_init(
                self, num_training_steps=args.max_steps
            )

        model = self._wrap_model(self.model, training=False, dataloader=dataloader)

        if len(self.accelerator._models) == 0 and model is self.model:

            if self.is_deepspeed_enabled:
                # prepare using `accelerator` prepare
                if hasattr(self.lr_scheduler, "step"):
                    if self.use_apex:
                        model = self.accelerator.prepare(self.model)
                    else:
                        model, self.optimizer = self.accelerator.prepare(
                            self.model, self.optimizer
                        )
                else:
                    # to handle cases wherein we pass "DummyScheduler" such as when it is specified in DeepSpeed config.
                    model, self.optimizer, self.lr_scheduler = self.accelerator.prepare(
                        self.model, self.optimizer, self.lr_scheduler
                    )
            else:
                model = self.accelerator.prepare_model(model, evaluation_mode=True)

            if self.is_fsdp_enabled:
                self.model = model

            # for the rest of this function `model` is the outside model, whether it was wrapped or not
            if model is not self.model:
                self.model_wrapped = model

            # backward compatibility
            if self.is_deepspeed_enabled:
                self.deepspeed = self.model_wrapped

        # ckpt loading
        resume_from_checkpoint = kwargs.get("resume_from_checkpoint")
        if resume_from_checkpoint is not None:
            if self.is_deepspeed_enabled:
                deepspeed_load_checkpoint(
                    self.model_wrapped, resume_from_checkpoint, self.logger
                )
            elif is_sagemaker_mp_enabled() or self.is_fsdp_enabled:
                self._load_from_checkpoint(resume_from_checkpoint, self.model_wrapped)

        # if full fp16 or bf16 eval is wanted and this ``evaluation`` or ``predict`` isn't called
        # while ``train`` is running, cast it to the right dtype first and then put on device
        if not self.is_in_train:
            if args.fp16_full_eval:
                model = model.to(dtype=torch.float16, device=args.device)
            elif args.bf16_full_eval:
                model = model.to(dtype=torch.bfloat16, device=args.device)

            # fp32 inference applied in automatic mixed precision case
            elif not self.is_deepspeed_enabled:
                model = model.to(dtype=torch.float32, device=args.device)

        batch_size = self.args.eval_batch_size

        logger.info(f"***** Running {description} *****")
        if has_length(dataloader):
            logger.info(f"  Num examples = {self.num_examples(dataloader)}")
        else:
            logger.info("  Num examples: Unknown")
        logger.info(f"  Batch size = {batch_size}")

        model.eval()

        self.callback_handler.eval_dataloader = dataloader
        # Do this before wrapping.
        eval_dataset = getattr(dataloader, "dataset", None)

        if args.past_index >= 0:
            self._past = None

        # Initialize containers
        # losses/preds/labels on GPU/TPU (accumulated for eval_accumulation_steps)
        outputs_host = defaultdict(lambda: None)

        # losses/preds/labels on CPU (final containers)
        all_outputs = defaultdict(lambda: None)
        # Will be useful when we have an iterable dataset so don't know its length.

        observed_num_examples = 0
        # Main evaluation loop
        for step, inputs in enumerate(dataloader):
            # Update the observed num examples
            observed_batch_size = find_batch_size(inputs)
            if observed_batch_size is not None:
                observed_num_examples += observed_batch_size
                # For batch samplers, batch_size is not known by the dataloader in advance.
                if batch_size is None:
                    batch_size = observed_batch_size

            # Prediction step
            outputs = self.inference_step(
                model, inputs, ignore_keys=ignore_keys, **kwargs
            )
            main_input_name = getattr(self.model, "main_input_name", "input_ids")

            if is_torch_xla_available():
                xm.mark_step()

            # Update containers on host
            for k, v in outputs.items():
                v = self.accelerator.pad_across_processes(v, dim=1, pad_index=-100)
                v = self.gather_function((v))

                outputs_host[k] = (
                    v
                    if outputs_host[k] is None
                    else nested_concat(outputs_host[k], v, padding_index=-100)
                )

            self.control = self.callback_handler.on_prediction_step(
                args, self.state, self.control
            )

            # Gather all tensors and put them back on the CPU if we have done enough accumulation steps.
            if (
                args.eval_accumulation_steps is not None
                and (step + 1) % args.eval_accumulation_steps == 0
            ):
                for k, v in outputs.items():
                    if outputs_host[k] is not None:
                        v = nested_numpify(outputs_host[k])
                        all_outputs[k] = (
                            v
                            if all_outputs[k] is None
                            else nested_concat(all_outputs[k], v, padding_index=-100)
                        )

                # Set back to None to begin a new accumulation
                for k, v in outputs_host.items():
                    v = None

        # After all calls to `.gather_function`, reset to `gather_for_metrics`:
        self.gather_function = self.accelerator.gather_for_metrics
        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of the evaluation loop
            delattr(self, "_past")

        # Gather all remaining tensors and put them back on the CPU
        for k, v in outputs.items():
            if outputs_host[k] is not None:
                v = nested_numpify(outputs_host[k])
                all_outputs[k] = (
                    v
                    if all_outputs[k] is None
                    else nested_concat(all_outputs[k], v, padding_index=-100)
                )

        # Number of samples
        if has_length(eval_dataset):
            num_samples = len(eval_dataset)
        # The instance check is weird and does not actually check for the type, but whether the dataset has the right
        # methods. Therefore we need to make sure it also has the attribute.
        elif (
            isinstance(eval_dataset, IterableDatasetShard)
            and getattr(eval_dataset, "num_examples", 0) > 0
        ):
            num_samples = eval_dataset.num_examples
        else:
            if has_length(dataloader):
                num_samples = self.num_examples(dataloader)
            else:  # both len(dataloader.dataset) and len(dataloader) fail
                num_samples = observed_num_examples
        if num_samples == 0 and observed_num_examples > 0:
            num_samples = observed_num_examples

        # Metrics!
        all_preds = all_outputs.get("outputs")
        all_labels = all_outputs.get("labels")
        all_inputs = (
            all_outputs.get(main_input_name)
            if args.include_inputs_for_metrics
            else None
        )

        if (
            self.compute_metrics is not None
            and all_preds is not None
            and all_labels is not None
        ):
            if args.include_inputs_for_metrics:
                metrics = self.compute_metrics(
                    EvalPrediction(
                        predictions=all_preds, label_ids=all_labels, inputs=all_inputs
                    )
                )
            else:
                metrics = self.compute_metrics(
                    EvalPrediction(predictions=all_preds, label_ids=all_labels)
                )
        else:
            metrics = {}

        # To be JSON-serializable, we need to remove numpy types or zero-d tensors
        metrics = denumpify_detensorize(metrics)

        if hasattr(self, "jit_compilation_time"):
            metrics[f"{metric_key_prefix}_jit_compilation_time"] = (
                self.jit_compilation_time
            )

        # Prefix all keys with metric_key_prefix + '_'
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

        return EvalLoopOutput(
            predictions=all_outputs,
            label_ids=all_labels,
            metrics=metrics,
            num_samples=num_samples,
        )
