from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from transformers.modeling_outputs import BaseModelOutput


class VisualEncoderMixin:
    """VisualEncoderMixin is an abstract class for visual encoders."""

    def get_dtype(self):
        """dtype of visual encoder"""
        raise NotImplementedError()

    def get_num_tokens(self) -> int:
        """The number of ouptut tokens. Mostly, num_patches (without cls token)"""
        raise NotImplementedError()

    def has_cls_token(self) -> bool:
        """Whether the encoder has cls token or not. Default: True"""
        return True

    def freeze_blocks(self, n: int):
        """Freeze the first n blocks of the encoder"""
        raise NotImplementedError()

    def postprocess_for_projector(self, visual_features):
        """Perform any post-processing (e.g., cls feature removal) for visual features
        before submitting to projector.
        """
        if self.has_cls_token():
            # defaultly, we assume cls token is located at 0-index if exists.
            visual_features = (
                visual_features[:, 1:]
                if visual_features.ndim == 3
                else visual_features[:, :, 1:]
            )
        return visual_features


@dataclass
class VisionModelOutput(BaseModelOutput):
    """
    Base class for vision model's outputs that also contains image embeddings of the pooling of the last hidden states.

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
    """

    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
