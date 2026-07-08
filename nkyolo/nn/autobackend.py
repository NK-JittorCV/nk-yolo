# NK-YOLO 🚀, AGPL-3.0 license
# Refer to https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/autobackend.py

import json
from pathlib import Path

import numpy as np
import jittor as jt
import jittor.nn as nn

from nkyolo.utils import LOGGER, ROOT, yaml_load
from nkyolo.utils.checks import check_suffix, check_yaml
from nkyolo.utils.downloads import attempt_download_asset, is_url


def check_class_names(names):
    """
    Check class names.

    Map imagenet class codes to human-readable names if required. Convert lists to dicts.
    """
    if isinstance(names, list):  # names is a list
        names = dict(enumerate(names))  # convert to dict
    if isinstance(names, dict):
        # Convert 1) string keys to int, i.e. '0' to 0, and non-string values to strings, i.e. True to 'True'
        names = {int(k): str(v) for k, v in names.items()}
        n = len(names)
        if max(names.keys()) >= n:
            raise KeyError(
                f"{n}-class dataset requires class indices 0-{n - 1}, but you have invalid class indices "
                f"{min(names.keys())}-{max(names.keys())} defined in your dataset YAML."
            )
        if isinstance(names[0], str) and names[0].startswith("n0"):  # imagenet class codes, i.e. 'n01440764'
            names_map = yaml_load(ROOT / "cfg/datasets/ImageNet.yaml")["map"]  # human-readable names
            names = {k: names_map[v] for k, v in names.items()}
    return names


def default_class_names(data=None):
    """Applies default class names to an input YAML file or returns numerical class names."""
    if data:
        data_yaml = check_yaml(data, hard=False)
        if data_yaml:
            names = yaml_load(data_yaml).get("names")
            if names:
                return names
    return {i: f"class{i}" for i in range(999)}  # return default if above errors


class AutoBackend(nn.Module):
    """
    Handles dynamic backend selection for running inference using NK-YOLO models with Jittor support.

    The AutoBackend class is designed to provide an abstraction layer for various inference engines. It supports a wide
    range of formats, each with specific naming conventions as outlined below:

        Supported Formats and Naming Conventions:
            | Format                | File Suffix      |
            |-----------------------|------------------|
            | JittorPickle          | *.pkl            |
            | jtScript              | *.jtscript      |
            | Torch Weights         | *.pt, *.pth      | (converted to Jittor on load)
            
        TODO: Support for other formats (ONNX, OpenVINO, TensorRT, CoreML, TensorFlow, PaddlePaddle, NCNN) is planned.

    This class offers dynamic backend switching capabilities based on the input model format, making it easier to deploy
    models across various platforms. Primary backend is Jittor for optimal performance.
    """

    @jt.no_grad()
    def __init__(
        self,
        weights="yolo11n.pt",
        device=jt.cpu,
        dnn=False,
        data=None,
        fp16=False,
        batch=1,
        fuse=True,
        verbose=True,
    ):
        """
        Initialize the AutoBackend for inference with Jittor support.

        Args:
            weights (str): Path to the model weights file. Defaults to 'yolo11n.pt'. Supports Jittor (.pkl) and Torch weights (.pt, .pth) via conversion.
            device (jt.device): Jittor device to run the model on. Defaults to jt.cpu.
            dnn (bool): Not currently supported. Reserved for future ONNX OpenCV DNN support.
            data (str | Path | optional): Path to the additional data.yaml file containing class names. Optional.
            fp16 (bool): Enable half-precision inference. Supported on Jittor. Defaults to False.
            batch (int): Batch-size to assume for inference.
            fuse (bool): Fuse Conv2D + BatchNorm layers for optimization. Defaults to True.
            verbose (bool): Enable verbose logging. Defaults to True.
        """
        super().__init__()
        self.names = {}
        self.transforms = None
        w = str(weights[0] if isinstance(weights, list) else weights)
        nn_module = isinstance(weights, jt.nn.Module)
        # _model_type() may return a short list (supported-only) or full list of flags.
        model_types = list(self._model_type(w))
        if len(model_types) == 15:
            (
                pt,
                pkl,
                jit,
                onnx,
                xml,
                engine,
                coreml,
                saved_model,
                pb,
                tflite,
                edgetpu,
                tfjs,
                paddle,
                ncnn,
                triton,
            ) = model_types
        elif len(model_types) == 5:
            pt, pkl, jit, torch_pt, triton = model_types
            pt = pt or torch_pt  # treat .pt/.pth weights as PT inputs for conversion
            onnx = xml = engine = coreml = saved_model = pb = tflite = edgetpu = tfjs = paddle = ncnn = False
        else:
            # Fallback: pad to full length for compatibility
            model_types = model_types + [False] * (15 - len(model_types))
            (
                pt,
                pkl,
                jit,
                onnx,
                xml,
                engine,
                coreml,
                saved_model,
                pb,
                tflite,
                edgetpu,
                tfjs,
                paddle,
                ncnn,
                triton,
            ) = model_types[:15]
        # Only pt, pkl, jit are supported; others are for format detection only
        fp16 &= pt or pkl or jit or nn_module  # FP16
        nhwc = False  # Jittor uses BCHW format
        stride = 32  # default stride
        model, metadata, task, kpt_shape = None, None, None, None

        # Download if not local
        if not (pt or pkl or jit or nn_module):
            w = attempt_download_asset(w)

        # In-memory Jittor model
        if nn_module:
            model = weights
            if fuse:
                model = model.fuse(verbose=verbose)
            stride = max(int(np.max(model.stride.numpy())), 32)
            names = model.names  # get class names
            self.model = model  # explicitly assign for to(), cpu(), cuda(), half()
            pt = True

        # Jittor (.pt)
        elif pt:
            from nkyolo.nn.tasks import attempt_load_weights

            model = attempt_load_weights(
                weights if isinstance(weights, list) else w, device=device, inplace=True, fuse=fuse
            )
            stride = max(int(np.max(model.stride.numpy())), 32)
            names = model.names  # get class names
            self.model = model  # explicitly assign for to(), cpu(), cuda(), half()

        # JittorPickle (.pkl)
        elif pkl:
            from nkyolo.nn.tasks import attempt_load_weights

            model = attempt_load_weights(
                weights if isinstance(weights, list) else w, device=device, inplace=True, fuse=fuse
            )
            stride = max(int(np.max(model.stride.numpy())), 32)
            names = model.names  # get class names
            self.model = model  # explicitly assign for to(), cpu(), cuda(), half()

        # jtScript
        elif jit:
            LOGGER.info(f"Loading {w} for jtScript inference...")
            extra_files = {"config.txt": ""}  # model metadata
            model = jt.jit.load(w, _extra_files=extra_files, map_location=device)
            if extra_files["config.txt"]:  # load metadata dict
                metadata = json.loads(extra_files["config.txt"], object_hook=lambda x: dict(x.items()))

        # TODO: Other format support (ONNX, OpenVINO, TensorRT, CoreML, TensorFlow, PaddlePaddle, NCNN, Triton)
        elif dnn or onnx or xml or engine or coreml or saved_model or pb or tflite or edgetpu or tfjs or paddle or ncnn or triton:
            raise NotImplementedError(
                f"Model format '{w}' is not yet supported. Currently only Jittor (.pkl), Torch weights (.pt/.pth, converted), and jtScript (.jtscript) formats are supported. "
                "TODO: Add support for other formats."
            )

        # Any other format (unsupported)
        else:
            from nkyolo.engine.exporter import export_formats

            raise TypeError(
                f"model='{w}' is not a supported model format. NK-YOLO supports: {export_formats()['Format']}\n"
                f"See https://docs.jittoryolo.com/modes/predict for help."
            )

        # Convert model weights to match the inference dtype: execute() feeds
        # FP16 inputs when fp16 is set, and mixed fp16-input/fp32-weight convs
        # have no cudnn algorithm (hard failure in algorithm selection).
        # NOTE: jittor Modules expose float32(), not torch's float().
        if model is not None and hasattr(model, "half"):
            model.half() if fp16 else model.float32()

        # Load external metadata YAML
        if isinstance(metadata, (str, Path)) and Path(metadata).exists():
            metadata = yaml_load(metadata)
        if metadata and isinstance(metadata, dict):
            for k, v in metadata.items():
                if k in {"stride", "batch"}:
                    metadata[k] = int(v)
                elif k in {"imgsz", "names", "kpt_shape"} and isinstance(v, str):
                    metadata[k] = eval(v)
            stride = metadata["stride"]
            task = metadata["task"]
            batch = metadata["batch"]
            imgsz = metadata["imgsz"]
            names = metadata["names"]
            kpt_shape = metadata.get("kpt_shape")
        elif not (pt or pkl or jit or nn_module):
            LOGGER.warning(f"WARNING ⚠️ Metadata not found for 'model={weights}'")

        # Check names
        if "names" not in locals():  # names missing
            names = default_class_names(data)
        names = check_class_names(names)

        # Disable gradients
        if pt or pkl:
            for p in model.parameters():
                p.requires_grad = False

        # Avoid self-referential attribute that breaks jittor's Module.dfs()
        locals_dict = {k: v for k, v in locals().items() if k != "self"}
        self.__dict__.update(locals_dict)  # assign all variables to self
        self.__dict__.pop("self", None)

    def execute(self, im, augment=False, visualize=False, embed=None):
        """
        Runs inference on the YOLOv8 MultiBackend model.

        Args:
            im (jt.Tensor): The image tensor to perform inference on.
            augment (bool): whether to perform data augmentation during inference, defaults to False
            visualize (bool): whether to visualize the output predictions, defaults to False
            embed (list, optional): A list of feature vectors/embeddings to return.

        Returns:
            (tuple): Tuple containing the raw output tensor, and processed output for visualization (if visualize=True)
        """
        if self.fp16 and im.dtype != jt.float16:
            im = im.half()  # to FP16
        if self.nhwc:
            im = im.permute(0, 2, 3, 1)  # jt BCHW to numpy BHWC shape(1,320,192,3)

        # Jittor (.pkl) or in-memory Jittor model
        if self.pt or self.pkl or self.nn_module:
            y = self.model(im, augment=augment, visualize=visualize, embed=embed)

        # jtScript
        elif self.jit:
            y = self.model(im)

        # TODO: Other format support
        else:
            raise NotImplementedError(
                "This model format is not yet supported. Currently only Jittor (.pkl), PT/PTH weights (converted), and jtScript (.jtscript) formats are supported."
            )

        if isinstance(y, (list, tuple)):
            if len(self.names) == 999 and (self.task == "segment" or len(y) == 2):  # segments and names not defined
                ip, ib = (0, 1) if len(y[0].shape) == 4 else (1, 0)  # index of protos, boxes
                nc = y[ib].shape[1] - y[ip].shape[3] - 4  # y = (1, 160, 160, 32), (1, 116, 8400)
                self.names = {i: f"class{i}" for i in range(nc)}
            return self.from_numpy(y[0]) if len(y) == 1 else [self.from_numpy(x) for x in y]
        else:
            return self.from_numpy(y)

    def from_numpy(self, x):
        """
        Convert a numpy array to a tensor.

        Args:
            x (np.ndarray): The array to be converted.

        Returns:
            (jt.Tensor): The converted tensor
        """
        return jt.tensor(x).to(self.device) if isinstance(x, np.ndarray) else x

    def warmup(self, imgsz=(1, 3, 640, 640)):
        """
        Warm up the model by running one forward pass with a dummy input.

        Args:
            imgsz (tuple): The shape of the dummy input tensor in the format (batch_size, channels, height, width)
        """

        warmup_types = self.pt, self.pkl, self.jit, self.nn_module
        if any(warmup_types):
            im = jt.empty(*imgsz, dtype=jt.half if self.fp16 else jt.float)  # input
            for _ in range(2 if self.jit else 1):
                self.execute(im)  # warmup

    @staticmethod
    def _model_type(p="path/to/model.pt"):
        """
        Takes a path to a model file and returns the model type. Possible types are pkl (JittorPickle),
        jit (jtScript), or PT/PTH weights (converted to Jittor).

        Args:
            p: path to the model file. Defaults to path/to/model.pt

        Examples:
            >>> model = AutoBackend(weights="path/to/model.pt")
            >>> model_type = model._model_type()  # returns model type flags
        """

        if not is_url(p) and not isinstance(p, str):
            check_suffix(p, export_formats()["Suffix"])
        suffix = Path(p).suffix.lower()
        pkl = suffix == ".pkl"
        jit = suffix == ".jtscript"
        torch_pt = suffix in {".pt", ".pth"}
        pt = False  # Jittor checkpoints use .pkl

        if pkl or jit or torch_pt:
            triton = False
        else:
            from urllib.parse import urlsplit

            url = urlsplit(p)
            triton = bool(url.netloc) and bool(url.path) and url.scheme in {"http", "grpc"}

        return [pt, pkl, jit, torch_pt, triton]


def export_formats():
    """NK-YOLO YOLO export formats (Jittor runtime, optional PT/PTH weight conversion)."""
    x = [
        ["JittorPickle", "pkl", ".pkl", True, True],
        ["jtScript", "jtscript", ".jtscript", True, True],
        ["PTWeights", "pt", ".pt", True, True],  # PT weights converted to Jittor
        ["PTHWeights", "pth", ".pth", True, True],  # PT state dict converted to Jittor
        # TODO: Add support for the following formats
        # ["ONNX", "onnx", ".onnx", True, True],
        # ["OpenVINO", "openvino", "_openvino_model", True, False],
        # ["TensorRT", "engine", ".engine", False, True],
        # ["CoreML", "coreml", ".mlpackage", True, False],
        # ["TensorFlow SavedModel", "saved_model", "_saved_model", True, True],
        # ["TensorFlow GraphDef", "pb", ".pb", True, True],
        # ["TensorFlow Lite", "tflite", ".tflite", True, False],
        # ["TensorFlow Edge TPU", "edgetpu", "_edgetpu.tflite", True, False],
        # ["TensorFlow.js", "tfjs", "_web_model", True, False],
        # ["PaddlePaddle", "paddle", "_paddle_model", True, True],
        # ["NCNN", "ncnn", "_ncnn_model", True, True],
    ]
    return dict(zip(["Format", "Argument", "Suffix", "CPU", "GPU"], zip(*x)))
