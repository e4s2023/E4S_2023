import os

REPO_ROOT = os.path.dirname(__file__)
PRETRAINED_ROOT = os.environ.get("PRETRAINED_ROOT", os.path.join(REPO_ROOT, "pretrained"))
TDDFA_ROOT = os.environ.get("TDDFA_ROOT", os.path.join(REPO_ROOT, "TDDFA_V2"))
TMP_ROOT = os.environ.get("TMP_ROOT", os.path.join(REPO_ROOT, "tmp"))
# This needs documentation
SHARE_PY_ROOT = os.environ.get("SHARE_ROOT", "/apdcephfs/share_1290939/zhianliu/py_projects/")
SHARE_MODELS_ROOT = os.environ.get("SHARE_MODELS_ROOT", "/apdcephfs_cq2/share_1290939/branchwang/pretrained_models")

__all__ = ["PRETRAINED_ROOT", "REPO_ROOT", "TDDFA_ROOT", "TMP_ROOT", "SHARE_PY_ROOT", "SHARE_MODELS_ROOT"]
