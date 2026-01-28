# NK-YOLO 🚀 AGPL-3.0 License

"""
Export utilities for NK-YOLO models.
"""

import pandas as pd


def export_formats():
    """NK-YOLO YOLO export formats."""
    x = [
        ["JittorPickle", "pkl", ".pkl", True, True],
        ["jtScript", "jtscript", ".jtscript", True, True],
        ["PyTorch", "torch", ".pt", True, True],  # PyTorch support
        ["PyTorchPTH", "torch", ".pth", True, True],
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
    return pd.DataFrame(x, columns=["Format", "Argument", "Suffix", "CPU", "GPU"])


def gd_outputs(gd):
    """TensorFlow GraphDef model output node names."""
    name_list, input_list = [], []
    for node in gd.node:  # tensorflow.core.framework.node_def_pb2.NodeDef
        name_list.append(node.name)
        input_list.extend(node.input)
    return sorted(f"{x}:0" for x in list(set(name_list) - set(input_list)) if not x.startswith("NoOp"))

