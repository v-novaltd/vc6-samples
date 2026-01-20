# SPDX-License-Identifier: BSD-3-Clause-Clear
# Copyright (c) 2025 V-Nova International Ltd.
from pathlib import Path
import logging

import global_vars

LOGGER = logging.getLogger("DEBUG_DUMP")
_CUPY_AVAILABLE = None


def dump_decoded_images(codec_name, decoded, names=None, sizes=None, prefix=None):
    if not global_vars.DEBUG_DUMP_IMAGES:
        return
    output_dir = Path(global_vars.DEBUG_DUMP_DIR) / codec_name
    output_dir.mkdir(parents=True, exist_ok=True)

    images = _to_numpy_batch(decoded)
    if not images:
        LOGGER.warning("No decodable images for %s; skipping debug dump.", codec_name)
        return

    for idx, image in enumerate(images):
        shape = sizes[idx] if sizes and idx < len(sizes) else None
        name = names[idx] if names and idx < len(names) else f"img_{idx:04d}"
        filename = f"{prefix}_{name}.png" if prefix else f"{name}.png"
        try:
            normalized = _normalize_image_array(image, shape)
        except ValueError as exc:
            LOGGER.warning("Skipping %s image %s: %s", codec_name, name, exc)
            continue
        _save_png(output_dir / filename, normalized)


def _save_png(path, image_array):
    from PIL import Image

    Image.fromarray(image_array).save(path)


def _to_numpy_batch(decoded):
    if isinstance(decoded, (list, tuple)):
        images = []
        for item in decoded:
            array = _to_numpy_array(item)
            if array is None:
                continue
            images.append(array)
        return images

    array = _to_numpy_array(decoded)
    if array is None:
        return []
    if array.ndim == 4:
        return [array[i] for i in range(array.shape[0])]
    return [array]


def _to_numpy_array(obj):
    import numpy as np

    if hasattr(obj, "as_cpu"):
        cpu_obj = obj.as_cpu()
        if hasattr(cpu_obj, "as_array"):
            return cpu_obj.as_array()
        if hasattr(cpu_obj, "asnumpy"):
            return cpu_obj.asnumpy()
        if hasattr(cpu_obj, "numpy"):
            return cpu_obj.numpy()
        return np.asarray(cpu_obj)

    if hasattr(obj, "cpu") and hasattr(obj, "numpy"):
        return obj.cpu().numpy()

    if hasattr(obj, "__dlpack__"):
        try:
            from torch.utils import dlpack
        except ImportError:
            return None
        return dlpack.from_dlpack(obj).cpu().numpy()

    if hasattr(obj, "__cuda_array_interface__"):
        global _CUPY_AVAILABLE
        if _CUPY_AVAILABLE is None:
            try:
                import cupy
            except ImportError:
                _CUPY_AVAILABLE = False
                LOGGER.warning("CuPy is required for CUDA debug dumps.")
                return None
            _CUPY_AVAILABLE = True
        if not _CUPY_AVAILABLE:
            return None
        import cupy
        return cupy.asarray(obj).get()

    if hasattr(obj, "__array__"):
        return np.asarray(obj)

    return None


def _normalize_image_array(image_array, shape):
    import numpy as np

    if image_array is None:
        raise ValueError("empty image array")

    array = image_array
    if array.ndim == 1 and shape and shape[0] > 0 and shape[1] > 0:
        width, height = shape
        array = array.reshape(height, width, 3)

    if array.ndim == 3 and array.shape[0] in (1, 3, 4) and array.shape[-1] not in (1, 3, 4):
        array = np.moveaxis(array, 0, -1)

    if array.ndim not in (2, 3):
        raise ValueError(f"unexpected image array shape: {array.shape}")

    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)

    return array
