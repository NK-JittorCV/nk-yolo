# NK-YOLO 🚀 AGPL-3.0 License
# Refer to https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/patches.py

"""Monkey patches to update/extend functionality of existing functions."""

import time
from pathlib import Path

import cv2
import numpy as np
import jittor as jt

# OpenCV Multilanguage-friendly functions ------------------------------------------------------------------------------
_imshow = cv2.imshow  # copy to avoid recursion errors


def imread(filename: str, flags: int = cv2.IMREAD_COLOR):
    """
    Read an image from a file.

    Args:
        filename (str): Path to the file to read.
        flags (int, optional): Flag that can take values of cv2.IMREAD_*. Defaults to cv2.IMREAD_COLOR.

    Returns:
        (np.ndarray): The read image.
    """
    return cv2.imdecode(np.fromfile(filename, np.uint8), flags)


def imwrite(filename: str, img: np.ndarray, params=None):
    """
    Write an image to a file.

    Args:
        filename (str): Path to the file to write.
        img (np.ndarray): Image to write.
        params (list of ints, optional): Additional parameters. See OpenCV documentation.

    Returns:
        (bool): True if the file was written, False otherwise.
    """
    try:
        cv2.imencode(Path(filename).suffix, img, params)[1].tofile(filename)
        return True
    except Exception:
        return False


def imshow(winname: str, mat: np.ndarray):
    """
    Displays an image in the specified window.

    Args:
        winname (str): Name of the window.
        mat (np.ndarray): Image to be shown.
    """
    _imshow(winname.encode("unicode_escape").decode(), mat)


# Jittor functions ----------------------------------------------------------------------------------------------------
_jittor_load = jt.load  # copy to avoid recursion errors
_jittor_save = jt.save


def jittor_load(*args, **kwargs):
    """
    Load a Jittor model with updated arguments to avoid warnings.

    This function wraps jt.load and adds the 'weights_only' argument for Jittor to prevent warnings.

    Args:
        *args (Any): Variable length argument list to pass to jt.load.
        **kwargs (Any): Arbitrary keyword arguments to pass to jt.load.

    Returns:
        (Any): The loaded Jittor object.

    Note:
        For Jittor versions 2.0 and above, this function automatically sets 'weights_only=False'
        if the argument is not provided, to avoid deprecation warnings.
    """

    return _jittor_load(*args, **kwargs)


def jittor_save(*args, **kwargs):
    """
    Optionally use dill to serialize lambda functions where pickle does not, adding robustness with 3 retries and
    exponential standoff in case of save failure.

    Args:
        *args (tuple): Positional arguments to pass to jt.save.
        **kwargs (Any): Keyword arguments to pass to jt.save.
    """
    for i in range(4):  # 3 retries
        try:
            return _jittor_save(*args, **kwargs)
        except RuntimeError as e:  # unable to save, possibly waiting for device to flush or antivirus scan
            if i == 3:
                raise e
            time.sleep((2**i) / 2)  # exponential standoff: 0.5s, 1.0s, 2.0s
