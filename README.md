# M4CXR: Exploring Multi-task Potentials of Multi-modal Large Language Models for Chest X-ray Interpretation

### Environment
```bash
pip install -r requirements.txt
```

### Data Preparation
Download datasets and preprocess with codes.
(TBA)

### Pretraining
```bash
PYTHONPATH=. torchrun --nproc_per_node=2 --nnodes=1 exp/cxr_llm/run.py --add_cfg_list mrg amp_bf16 pre_training_abstractor paths
```

### Visual instruction tuning

```bash
PYTHONPATH=. torchrun --nproc_per_node=2 --nnodes=1 exp/cxr_llm/run.py --add_cfg_list mrg iu vqa amp_bf16 instruction_tuning paths
```

### LICENSE
[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)
