import os

__author__ = "Maomao Li, Ge Yuan, Cairong Wang, Zhian Liu, Yong Zhang, Yongwei Nie, Jue Wang, Dong Xu"
__version__ = "0.0.1"
__description__ = "A Face Swapping and Editing Framework Based on StyleGAN Latent Space"

REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
DATASETS_ROOT = os.environ.get(
    "E4S2024_DATASETS_ROOT", "/apdcephfs/share_1290939/zhianliu/datasets"
)
PRETRAINED_ROOT = os.environ.get(
    "PRETRAINED_ROOT", os.path.join(REPO_ROOT, "pretrained")
)
PRETRAINED_SHARE_ROOT = os.environ.get(
    "PRETRAINED_SHARE_ROOT", "/apdcephfs/share_1290939/zhianliu/pretrained_models"
)
TDDFA_ROOT = os.environ.get("TDDFA_ROOT", os.path.join(REPO_ROOT, "TDDFA_V2"))
TMP_ROOT = os.environ.get("TMP_ROOT", os.path.join(REPO_ROOT, "tmp"))
# This needs documentation
SHARE_PY_ROOT = os.environ.get(
    "SHARE_ROOT", "/apdcephfs/share_1290939/zhianliu/py_projects/"
)
SHARE_MODELS_ROOT = os.environ.get(
    "SHARE_MODELS_ROOT", "/apdcephfs_cq2/share_1290939/branchwang/pretrained_models"
)

__all__ = [
    "REPO_ROOT",
    "DATASETS_ROOT",
    "PRETRAINED_ROOT",
    "PRETRAINED_SHARE_ROOT",
    "TDDFA_ROOT",
    "TMP_ROOT",
    "SHARE_PY_ROOT",
    "SHARE_MODELS_ROOT",
]
