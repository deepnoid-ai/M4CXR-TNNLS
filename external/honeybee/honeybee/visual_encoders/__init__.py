from ... import utils
from ...pipeline.utils import check_local_file
from .clip import CustomCLIP
from .dinov2 import CustomDinov2Model


def build_encoder(config):
    with utils.transformers_log_level(40):  # 40 == logging.ERROR
        if config.encoder_type == "openai.clip":
            vm_local_files_only, vm_file_name = check_local_file(
                config.pretrained_vision_name_or_path
            )
            model = CustomCLIP.from_pretrained(
                vm_file_name,
                local_files_only=vm_local_files_only,
            )
        elif config.encoder_type == "dinov2":
            vm_local_files_only, vm_file_name = check_local_file(
                config.pretrained_vision_name_or_path
            )
            model = CustomDinov2Model.from_pretrained(
                vm_file_name,
                local_files_only=vm_local_files_only,
            )
        else:
            raise NotImplementedError()

    return model
