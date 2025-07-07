from transformers import Dinov2Model

from .visual_encoder_mixin import VisualEncoderMixin


class CustomDinov2Model(Dinov2Model, VisualEncoderMixin):
    def get_dtype(self):
        return self.embeddings.patch_embeddings.projection.weight.dtype

    def get_num_tokens(self):
        return self.embeddings.patch_embeddings.num_patches
