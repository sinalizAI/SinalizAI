


import time
from pathlib import Path

import cv2
import numpy as np
import torch


_imshow = cv2.imshow


def imread(filename: str, flags: int = cv2.IMREAD_COLOR):
    
    file_bytes = np.fromfile(filename, np.uint8)
    if filename.endswith((".tiff", ".tif")):
        success, frames = cv2.imdecodemulti(file_bytes, cv2.IMREAD_UNCHANGED)
        if success:

            return frames[0] if len(frames) == 1 and frames[0].ndim == 3 else np.stack(frames, axis=2)
        return None
    else:
        return cv2.imdecode(file_bytes, flags)


def imwrite(filename: str, img: np.ndarray, params=None):
    
    try:
        cv2.imencode(Path(filename).suffix, img, params)[1].tofile(filename)
        return True
    except Exception:
        return False


def imshow(winname: str, mat: np.ndarray):
    
    _imshow(winname.encode("unicode_escape").decode(), mat)



_torch_load = torch.load
_torch_save = torch.save


def torch_load(*args, **kwargs):
    
    from ultralytics.utils.torch_utils import TORCH_1_13

    if TORCH_1_13 and "weights_only" not in kwargs:
        kwargs["weights_only"] = False

    return _torch_load(*args, **kwargs)


def torch_save(*args, **kwargs):
    
    for i in range(4):
        try:
            return _torch_save(*args, **kwargs)
        except RuntimeError as e:
            if i == 3:
                raise e
            time.sleep((2**i) / 2)
