from pathlib import Path


KAGGLE_INPUT_ROOT = "/kaggle/input"
KAGGLE_WORKING_ROOT = "/kaggle/working"


def _normalize(path_value):
    return str(path_value).replace("\\", "/")


def ensure_writable_path(path_value):
    """Remap Kaggle read-only input paths to /kaggle/working."""
    path_obj = Path(path_value)
    normalized = _normalize(path_obj)

    if normalized == KAGGLE_INPUT_ROOT or normalized.startswith(f"{KAGGLE_INPUT_ROOT}/"):
        relative = normalized[len(KAGGLE_INPUT_ROOT) :].lstrip("/")
        return Path(KAGGLE_WORKING_ROOT) / relative if relative else Path(KAGGLE_WORKING_ROOT)

    return path_obj


def resolve_output_base(default_base):
    """Choose a writable base on Kaggle while preserving local behavior."""
    default_path = Path(default_base)
    normalized = _normalize(default_path)

    if Path(KAGGLE_WORKING_ROOT).exists() or normalized.startswith(f"{KAGGLE_INPUT_ROOT}/"):
        return Path(KAGGLE_WORKING_ROOT)

    return default_path
