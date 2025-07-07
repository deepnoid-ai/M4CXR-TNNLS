from .abnormality_detetection_dataset import AbnormalityDetectionDataset
from .disease_classification_dataset import DiseaseClassificationDataset
from .finding_grounding_dataset import FindingGroundingDataset
from .grounded_finding_dataset import GroundedFindingDataset
from .grounded_organ_dataset import GroundedOrganDataset
from .imagenome_dataset import ImagenomeDataset
from .mimic_vqa_dataset import MimicVqaDataset
from .mimiccxr_diff_vqa import MimiccxrDiffVqaDataset
from .mimiccxr_multi_image import MimiccxrMultiImageDataset
from .mimiccxr_multi_study import MimiccxrMultiStudyDataset
from .mimiccxr_single_image import MimiccxrSingleImageDataset
from .mscxr_dataset import MscxrDataset
from .multi_finding_grounding_dataset import MultiFindingGroundingDataset
from .organ_grounding_dataset import OrganGroundingDataset
from .radialog_dataset import RadialogDataset
from .slake_dataset import SlakeDataset

DATASET_CLASS_LIST = [
    MimiccxrSingleImageDataset,
    MimiccxrMultiImageDataset,
    MimiccxrMultiStudyDataset,
    MimiccxrDiffVqaDataset,
    MimicVqaDataset,
    ImagenomeDataset,
    MscxrDataset,
    MimicVqaDataset,
    ImagenomeDataset,
    DiseaseClassificationDataset,
    FindingGroundingDataset,
    GroundedFindingDataset,
    MultiFindingGroundingDataset,
    AbnormalityDetectionDataset,
    OrganGroundingDataset,
    GroundedOrganDataset,
    ImagenomeDataset,
    RadialogDataset,
    SlakeDataset,
]
DATASET_CLASS_DICT = {c.__name__: c for c in DATASET_CLASS_LIST}


def load_dataset(
    dset_name, tokenizer, processors, max_length, class_name: str, **kwargs
):
    dataset_class = DATASET_CLASS_DICT[class_name]
    dset = dataset_class(
        tokenizer, processors, max_length, dset_name=dset_name, **kwargs
    )

    return dset
