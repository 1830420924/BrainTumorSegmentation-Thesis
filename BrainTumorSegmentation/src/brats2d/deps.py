from __future__ import annotations


def require_nibabel():
    try:
        import nibabel as nib
    except ImportError as exc:
        raise RuntimeError(
            "nibabel is required but is not installed in the current environment.\n"
            "Install it with:\n"
            "  conda run -n pytorch_gpu pip install nibabel"
        ) from exc
    return nib
