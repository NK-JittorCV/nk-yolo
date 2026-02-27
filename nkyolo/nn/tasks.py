# NK-YOLO 🚀, AGPL-3.0 license
# Refer to https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/tasks.py

import contextlib
import importlib.util
import pickle
import re
import types
from copy import deepcopy
from pathlib import Path

import numpy as np
import jittor as jt
from jittor import nn

from nkyolo.nn.modules import (
    AIFI,
    C1,
    C2,
    C2PSA,
    C3,
    C3TR,
    ELAN1,
    OBB,
    PSA,
    SPP,
    SPPELAN,
    SPPF,
    AConv,
    ADown,
    Bottleneck,
    BottleneckCSP,
    C2f,
    C2fAttn,
    C2fCIB,
    C2fPSA,
    C3Ghost,
    C3k2,
    C3x,
    CBFuse,
    CBLinear,
    Classify,
    Concat,
    Conv,
    Conv2,
    ConvTranspose,
    Detect,
    DWConv,
    DWConvTranspose2d,
    Focus,
    GhostBottleneck,
    GhostConv,
    HGBlock,
    HGStem,
    ImagePoolingAttn,
    Index,
    Pose,
    RepC3,
    RepConv,
    RepNCSPELAN4,
    RepVGGDW,
    ResNetLayer,
    RTDETRDecoder,
    SCDown,
    Segment,
    TorchVision,
    WorldDetect,
    v10Detect,
    MSBlock,
    A2C2f
)
from nkyolo.utils import DEFAULT_CFG_DICT, DEFAULT_CFG_KEYS, LOGGER, yaml_load
from nkyolo.utils.checks import check_requirements, check_suffix, check_yaml
from nkyolo.utils.loss import (
    E2EDetectLoss,
    v8ClassificationLoss,
    v8DetectionLoss,
)
from nkyolo.utils.ops import make_divisible
from nkyolo.utils.plotting import feature_visualization
from nkyolo.utils.jittor_utils import (
    fuse_conv_and_bn,
    fuse_deconv_and_bn,
    initialize_weights,
    model_info,
    scale_img,
    state_dict_to_jittor,
    time_sync,
)

_TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
if _TORCH_AVAILABLE:
    import torch



class BaseModel(nn.Module):
    """The BaseModel class serves as a base class for all the models in the NK-YOLO family."""

    def __init__(self):
        super().__init__()
        self.criterion = None
        self.clip_model = None
        self.yaml_file = ""
        self.names = {}
        self.transforms = None
        self.supports_flops = True

    def execute(self, x, *args, **kwargs):
        """
        Perform execute pass of the model for either training or inference.

        If x is a dict, calculates and returns the loss for training. Otherwise, returns predictions for inference.

        Args:
            x (jt.Var | dict): Input tensor for inference, or dict with image tensor and labels for training.
            *args (Any): Variable length argument list.
            **kwargs (Any): Arbitrary keyword arguments.

        Returns:
            (jt.Var): Loss if x is a dict (training), or network predictions (inference).
        """
        if isinstance(x, dict):  # for cases of training and validating while training.
            return self.loss(x, *args, **kwargs)
        return self.predict(x, *args, **kwargs)

    def predict(self, x, profile=False, visualize=False, augment=False, embed=None):
        """
        Perform a execute pass through the network.

        Args:
            x (jt.Var): The input tensor to the model.
            profile (bool):  Print the computation time of each layer if True, defaults to False.
            visualize (bool): Save the feature maps of the model if True, defaults to False.
            augment (bool): Augment image during prediction, defaults to False.
            embed (list, optional): A list of feature vectors/embeddings to return.

        Returns:
            (jt.Var): The last output of the model.
        """
        if augment:
            return self._predict_augment(x)
        return self._predict_once(x, profile, visualize, embed)

    def _predict_once(self, x, profile=False, visualize=False, embed=None):
        """
        Perform a execute pass through the network.

        Args:
            x (jt.Var): The input tensor to the model.
            profile (bool):  Print the computation time of each layer if True, defaults to False.
            visualize (bool): Save the feature maps of the model if True, defaults to False.
            embed (list, optional): A list of feature vectors/embeddings to return.

        Returns:
            (jt.Var): The last output of the model.
        """
        y, dt, embeddings = [], [], []  # outputs
        embed = frozenset(embed) if embed is not None else {-1}
        max_idx = max(embed)
        for m in self.model:
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
            if profile:
                self._profile_one_layer(m, x, dt)
            x = m(x)  # run
            y.append(x if m.i in self.save else None)  # save output
            if visualize:
                feature_visualization(x, m.type, m.i, save_dir=visualize)
            if m.i in embed:
                embeddings.append(nn.functional.adaptive_avg_pool2d(x, (1, 1)).squeeze(-1).squeeze(-1))  # flatten
                if m.i == max_idx:
                    return jt.unbind(jt.concat(embeddings, 1), dim=0)
        return x

    def _predict_augment(self, x):
        """Perform augmentations on input image x and return augmented inference."""
        LOGGER.warning(
            f"WARNING ⚠️ {self.__class__.__name__} does not support 'augment=True' prediction. "
            f"Reverting to single-scale prediction."
        )
        return self._predict_once(x)

    def _profile_one_layer(self, m, x, dt):
        """
        Profile the computation time and FLOPs of a single layer of the model on a given input. Appends the results to
        the provided list.

        Args:
            m (nn.Module): The layer to be profiled.
            x (jt.Var): The input data to the layer.
            dt (list): A list to store the computation time of the layer.

        Returns:
            None
        """
        c = m == self.model[-1] and isinstance(x, list)  # is final layer list, copy input as inplace fix
        # Calculate FLOPs
        from nkyolo.utils.jittor_profile import profile_layer_with_fallback
        flops = profile_layer_with_fallback(m, inputs=[x.copy() if c else x])
        t = time_sync()
        for _ in range(10):
            m(x.copy() if c else x)
        dt.append((time_sync() - t) * 100)
        if m == self.model[0]:
            LOGGER.info(f"{'time (ms)':>10s} {'GFLOPs':>10s} {'params':>10s}  module")
        LOGGER.info(f"{dt[-1]:10.2f} {flops:10.2f} {m.np:10.0f}  {m.type}")
        if c:
            LOGGER.info(f"{sum(dt):10.2f} {'-':>10s} {'-':>10s}  Total")

    def fuse(self, verbose=True):
        """
        Fuse the `Conv2d()` and `BatchNorm2d()` layers of the model into a single layer, in order to improve the
        computation efficiency.

        Returns:
            (nn.Module): The fused model is returned.
        """
        if not self.is_fused():
            for m in self.model.modules():
                if isinstance(m, (Conv, Conv2, DWConv)):
                    if isinstance(m, Conv2):
                        m.fuse_convs()
                    m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
                    delattr(m, "bn")  # remove batchnorm
                    m.execute = m.execute_fuse  # update execute
                if isinstance(m, ConvTranspose):
                    m.conv_transpose = fuse_deconv_and_bn(m.conv_transpose, m.bn)
                    delattr(m, "bn")  # remove batchnorm
                    m.execute = m.execute_fuse  # update execute
                if isinstance(m, RepConv):
                    m.fuse_convs()
                    m.execute = m.execute_fuse  # update execute
                if isinstance(m, RepVGGDW):
                    m.fuse()
                    m.execute = m.execute_fuse
            self.info(verbose=verbose)

        return self

    def is_fused(self, thresh=10):
        """
        Check if the model has less than a certain threshold of BatchNorm layers.

        Args:
            thresh (int, optional): The threshold number of BatchNorm layers. Default is 10.

        Returns:
            (bool): True if the number of BatchNorm layers in the model is less than the threshold, False otherwise.
        """
        bn = tuple(v for k, v in nn.__dict__.items() if "Norm" in k)  # normalization layers, i.e. BatchNorm2d()
        return sum(isinstance(v, bn) for v in self.modules()) < thresh  # True if < 'thresh' BatchNorm layers in model

    def info(self, detailed=False, verbose=True, imgsz=640):
        """
        Prints model information.

        Args:
            detailed (bool): if True, prints out detailed information about the model. Defaults to False
            verbose (bool): if True, prints out the model information. Defaults to False
            imgsz (int): the size of the image that the model will be trained on. Defaults to 640
        """
        return model_info(self, detailed=detailed, verbose=verbose, imgsz=imgsz)

    def _apply(self, fn):
        """
        Applies a function to all the tensors in the model that are not parameters or registered buffers.

        Args:
            fn (function): the function to apply to the model

        Returns:
            (BaseModel): An updated BaseModel object.
        """
        self = super()._apply(fn)
        m = self.model[-1]  # Detect()
        if isinstance(m, (Detect, WorldDetect, v10Detect)):  # includes all Detect subclasses
            m.stride = fn(m.stride)
            m.stride.stop_grad()  # stride is a configuration parameter, not a trainable weight
            m.anchors = fn(m.anchors)
            m.strides = fn(m.strides)
            m.anchors.stop_grad()
            m.strides.stop_grad()
            m.stride.persistent = False
            m.anchors.persistent = False
            m.strides.persistent = False
        return self

    def load(self, weights, verbose=True):
        """
        Load weights into the model (Jittor version).

        Args:
            weights (dict | nn.Module): Checkpoint dict or Jittor model.
            verbose (bool): Whether to log transfer progress.
        """
        
        # 1. Extract weight state_dict
        if isinstance(weights, dict):
            model_data = weights.get("model", weights)
        else:
            model_data = weights

        if isinstance(model_data, nn.Module):
            csd = model_data.state_dict()
        elif isinstance(model_data, dict):
            csd = model_data
        else:
            raise TypeError(f"Unsupported weights type: {type(model_data)}")

        csd = state_dict_to_jittor(csd)
        droppable_suffixes = (".anchors", ".strides")
        csd = {k: v for k, v in csd.items() if not k.endswith(droppable_suffixes)}

        # 2. Filter out keys with shape mismatch
        model_sd = self.state_dict()
        updated_csd = {
            k: v for k, v in csd.items()
            if k in model_sd and tuple(model_sd[k].shape) == tuple(v.shape)
        }

        # 3. Load
        self.load_state_dict(updated_csd)
        len_updated = len(updated_csd)

        # 4. Handle first conv channel mismatch
        first_conv = "model.0.conv.weight"
        if first_conv not in updated_csd and first_conv in model_sd:
            c1, c2, h, w = model_sd[first_conv].shape
            cc1, cc2, ch, cw = csd[first_conv].shape
            if ch == h and cw == w:
                c1, c2 = min(c1, cc1), min(c2, cc2)
                model_sd[first_conv][:c1, :c2] = csd[first_conv][:c1, :c2]
                len_updated += 1

        if verbose:
            LOGGER.info(
                f"Transferred {len_updated}/{len(self.model.state_dict())} "
                f"items from pretrained weights"
            )

    def loss(self, batch, preds=None):
        """
        Compute loss.

        Args:
            batch (dict): Batch to compute loss on
            preds (jt.Var | List[jt.Var]): Predictions.
        """
        if self.criterion is None:
            self.criterion = self.init_criterion()

        preds = self.execute(batch["img"]) if preds is None else preds
        return self.criterion(preds, batch)

    def init_criterion(self):
        """Initialize the loss criterion for the BaseModel."""
        raise NotImplementedError("compute_loss() needs to be implemented by task heads")

    def clone(self):
        """Return a detached copy of the model."""
        raise NotImplementedError("clone() must be implemented by derived model classes.")


class DetectionModel(BaseModel):
    """YOLO detection model (supports YOLOv5, v8, v9, v10, v11, etc.)."""

    def __init__(self, cfg, ch=3, nc=None, verbose=True):  # model, input channels, number of classes
        """Initialize the YOLO detection model with the given config and parameters.
        
        Args:
            cfg (str | dict): Model configuration file path (supports YOLOv5, v8, v9, v10, v11, etc.) or config dict.
            ch (int): Number of input channels. Default is 3 (RGB).
            nc (int, optional): Number of classes. If None, uses value from config.
            verbose (bool): Whether to print model information. Default is True.
        """
        super().__init__()
        self.yaml = cfg if isinstance(cfg, dict) else yaml_model_load(cfg)  # cfg dict
        self.yaml_file = self.yaml.get("yaml_file", "")
        if self.yaml["backbone"][0][2] == "Silence":
            LOGGER.warning(
                "WARNING ⚠️ YOLOv9 `Silence` module is deprecated in favor of nn.Identity. "
                "Please delete local *.pt file and re-download the latest model checkpoint."
            )
            self.yaml["backbone"][0][2] = "nn.Identity"

        # Define model
        ch = self.yaml["ch"] = self.yaml.get("ch", ch)  # input channels
        if nc and nc != self.yaml["nc"]:
            LOGGER.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
            self.yaml["nc"] = nc  # override YAML value
        self.model, self.save = parse_model(deepcopy(self.yaml), ch=ch, verbose=verbose)  # model, savelist
        self.names = {i: f"{i}" for i in range(self.yaml["nc"])}  # default names dict
        self.inplace = self.yaml.get("inplace", True)
        self.end2end = self.model[-1].end2end

        # Build strides
        m = self.model[-1]  # Detect()
        if isinstance(m, (Detect, WorldDetect, v10Detect)):  # all detection head types
            s = 640  # 2x min stride
            m.inplace = self.inplace

            def _execute(x):
                """Performs a forward pass through the model for stride calculation."""
                # Forward through backbone and neck to get feature maps
                y = []  # outputs
                for module in self.model[:-1]:  # excluding the head part
                    if module.f != -1:  # if not from previous layer
                        x = y[module.f] if isinstance(module.f, int) else [x if j == -1 else y[j] for j in module.f]
                    x = module(x)  # run
                    y.append(x if module.i in self.save else None)  # save output
                
                # Get feature maps that will be fed to the head
                head_input = [y[j] for j in m.f]
                
                # Execute head to get feature maps
                # In training mode, Detect.execute returns feature maps directly
                # In eval mode, it returns (inference_output, feature_maps)
                head_output = m(head_input)
                
                # Handle different return formats from head
                if isinstance(head_output, (list, tuple)) and len(head_output) == 2:
                    # Eval mode: (inference, features) - return feature maps
                    return head_output[1]
                elif isinstance(head_output, list):
                    # Training mode: feature maps list
                    return head_output
                else:
                    # Fallback: use head input (feature maps before head processing)
                    return head_input

            m.stride = jt.Var([s / x.shape[-2] for x in _execute(jt.zeros(1, ch, s, s))])  # execute
            m.stride.stop_grad()  # stride is a configuration parameter, not a trainable weight
            self.stride = m.stride
            m.bias_init()  # only run once
        else:
            self.stride = jt.Var([32])  # default stride for i.e. RTDETR
            self.stride.stop_grad()  # stride is a configuration parameter, not a trainable weight

        # Init weights, biases
        initialize_weights(self)
        if verbose:
            self.info()
            LOGGER.info("")

    def clone(self):
        """Clone the detection model using its YAML config and weights."""
        model = self.__class__(
            cfg=deepcopy(self.yaml),
            ch=self.yaml.get("ch", 3),
            nc=self.yaml.get("nc"),
            verbose=False,
        )
        model.load_state_dict(self.state_dict())
        model.names = deepcopy(self.names)
        model.inplace = self.inplace
        return model

    def _predict_augment(self, x):
        """Perform augmentations on input image x and return augmented inference and train outputs."""
        if self.end2end or self.__class__.__name__ != "DetectionModel":
            LOGGER.warning("WARNING ⚠️ Model does not support 'augment=True', reverting to single-scale prediction.")
            return self._predict_once(x)
        img_size = x.shape[-2:]  # height, width
        s = [1, 0.83, 0.67]  # scales
        f = [None, 3, None]  # flips (2-ud, 3-lr)
        y = []  # outputs
        for si, fi in zip(s, f):
            xi = scale_img(x.flip(fi) if fi else x, si, gs=int(self.stride.max()))
            yi = super().predict(xi)  # execute
            # Handle different output formats: (pred, train) tuple or just pred
            yi = yi[0] if isinstance(yi, (list, tuple)) else yi
            yi = self._descale_pred(yi, fi, si, img_size)
            y.append(yi)
        y = self._clip_augmented(y)  # clip augmented tails
        return jt.concat(y, -1), None  # augmented inference, train

    @staticmethod
    def _descale_pred(p, flips, scale, img_size, dim=1):
        """De-scale predictions following augmented inference (inverse operation)."""
        p[:, :4] /= scale  # de-scale
        x, y, wh, cls = p.split((1, 1, 2, p.shape[dim] - 4), dim)
        if flips == 2:
            y = img_size[0] - y  # de-flip ud
        elif flips == 3:
            x = img_size[1] - x  # de-flip lr
        return jt.concat((x, y, wh, cls), dim)

    def _clip_augmented(self, y):
        """Clip YOLO augmented inference tails."""
        nl = self.model[-1].nl  # number of detection layers (P3-P5)
        g = sum(4**x for x in range(nl))  # grid points
        e = 1  # exclude layer count
        i = (y[0].shape[-1] // g) * sum(4**x for x in range(e))  # indices
        y[0] = y[0][..., :-i]  # large
        i = (y[-1].shape[-1] // g) * sum(4 ** (nl - 1 - x) for x in range(e))  # indices
        y[-1] = y[-1][..., i:]  # small
        return y

    def init_criterion(self):
        """Initialize the loss criterion for the DetectionModel."""
        return E2EDetectLoss(self) if self.end2end else v8DetectionLoss(self)


class OBBModel(DetectionModel):
    """YOLO Oriented Bounding Box (OBB) model (supports YOLOv5, v8, v9, v10, v11, etc.).
    
    TODO: This model is not yet implemented. Only DetectionModel is currently supported.
    """
    # TODO: Not implemented yet - OBB task support

    def __init__(self, cfg="yolov8n-obb.yaml", ch=3, nc=None, verbose=True):
        """Initialize YOLO OBB model with given config and parameters."""
        raise NotImplementedError("OBBModel is not implemented yet. Only DetectionModel is currently supported.")

    def init_criterion(self):
        """Initialize the loss criterion for the model."""
        raise NotImplementedError("OBBModel is not implemented yet. Only DetectionModel is currently supported.")


class SegmentationModel(DetectionModel):
    """YOLO segmentation model (supports YOLOv5, v8, v9, v10, v11, etc.).
    
    TODO: This model is not yet implemented. Only DetectionModel is currently supported.
    """
    # TODO: Not implemented yet - Segmentation task support

    def __init__(self, cfg="yolov8n-seg.yaml", ch=3, nc=None, verbose=True):
        """Initialize YOLO segmentation model with given config and parameters."""
        raise NotImplementedError("SegmentationModel is not implemented yet. Only DetectionModel is currently supported.")

    def init_criterion(self):
        """Initialize the loss criterion for the SegmentationModel."""
        raise NotImplementedError("SegmentationModel is not implemented yet. Only DetectionModel is currently supported.")


class PoseModel(DetectionModel):
    """YOLO pose estimation model (supports YOLOv5, v8, v9, v10, v11, etc.).
    
    TODO: This model is not yet implemented. Only DetectionModel is currently supported.
    """
    # TODO: Not implemented yet - Pose estimation task support

    def __init__(self, cfg="yolov8n-pose.yaml", ch=3, nc=None, data_kpt_shape=(None, None), verbose=True):
        """Initialize YOLO Pose model."""
        raise NotImplementedError("PoseModel is not implemented yet. Only DetectionModel is currently supported.")

    def init_criterion(self):
        """Initialize the loss criterion for the PoseModel."""
        raise NotImplementedError("PoseModel is not implemented yet. Only DetectionModel is currently supported.")


class ClassificationModel(BaseModel):
    """YOLO classification model (supports YOLOv5, v8, v9, v10, v11, etc.).
    
    TODO: This model is not yet implemented. Only DetectionModel is currently supported.
    """
    # TODO: Not implemented yet - Classification task support

    def __init__(self, cfg="yolov8n-cls.yaml", ch=3, nc=None, verbose=True):
        """Init ClassificationModel with YAML, channels, number of classes, verbose flag."""
        raise NotImplementedError("ClassificationModel is not implemented yet. Only DetectionModel is currently supported.")

    def _from_yaml(self, cfg, ch, nc, verbose):
        """Set YOLO model configurations and define the model architecture."""
        self.yaml = cfg if isinstance(cfg, dict) else yaml_model_load(cfg)  # cfg dict

        # Define model
        ch = self.yaml["ch"] = self.yaml.get("ch", ch)  # input channels
        if nc and nc != self.yaml["nc"]:
            LOGGER.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
            self.yaml["nc"] = nc  # override YAML value
        elif not nc and not self.yaml.get("nc", None):
            raise ValueError("nc not specified. Must specify nc in model.yaml or function arguments.")
        self.model, self.save = parse_model(deepcopy(self.yaml), ch=ch, verbose=verbose)  # model, savelist
        self.stride = jt.Var([1])  # no stride constraints
        self.stride.stop_grad()  # stride is a configuration parameter, not a trainable weight
        self.names = {i: f"{i}" for i in range(self.yaml["nc"])}  # default names dict
        self.info()

    @staticmethod
    def reshape_outputs(model, nc):
        """Update a TorchVision classification model to class count 'n' if required."""
        base = model.model if isinstance(model, ClassificationModel) else model
        name, m = list(base.named_children())[-1]  # last module
        if isinstance(m, Classify):  # YOLO Classify() head
            if m.linear.out_features != nc:
                m.linear = nn.Linear(m.linear.in_features, nc)
        elif isinstance(m, nn.Linear):  # ResNet, EfficientNet
            if m.out_features != nc:
                setattr(model, name, nn.Linear(m.in_features, nc))
        elif isinstance(m, nn.Sequential):
            types = [type(x) for x in m]
            if nn.Linear in types:
                i = len(types) - 1 - types[::-1].index(nn.Linear)  # last nn.Linear index
                if m[i].out_features != nc:
                    m[i] = nn.Linear(m[i].in_features, nc)
            elif nn.Conv2d in types:
                i = len(types) - 1 - types[::-1].index(nn.Conv2d)  # last nn.Conv2d index
                if m[i].out_channels != nc:
                    m[i] = nn.Conv2d(m[i].in_channels, nc, m[i].kernel_size, m[i].stride, bias=m[i].bias is not None)

    def init_criterion(self):
        """Initialize the loss criterion for the ClassificationModel."""
        return v8ClassificationLoss()


class RTDETRDetectionModel(DetectionModel):
    """
    RTDETR (Real-time DEtection and Tracking using Transformers) Detection Model class.

    TODO: This model is not yet implemented. Only DetectionModel is currently supported.
    
    This class is responsible for constructing the RTDETR architecture, defining loss functions, and facilitating both
    the training and inference processes. RTDETR is an object detection and tracking model that extends from the
    DetectionModel base class.

    Attributes:
        cfg (str): The configuration file path or preset string. Default is 'rtdetr-l.yaml'.
        ch (int): Number of input channels. Default is 3 (RGB).
        nc (int, optional): Number of classes for object detection. Default is None.
        verbose (bool): Specifies if summary statistics are shown during initialization. Default is True.

    Methods:
        init_criterion: Initializes the criterion used for loss calculation.
        loss: Computes and returns the loss during training.
        predict: Performs a execute pass through the network and returns the output.
    """
    # TODO: Not implemented yet - RTDETR task support

    def __init__(self, cfg="rtdetr-l.yaml", ch=3, nc=None, verbose=True):
        """
        Initialize the RTDETRDetectionModel.

        Args:
            cfg (str): Configuration file name or path.
            ch (int): Number of input channels.
            nc (int, optional): Number of classes. Defaults to None.
            verbose (bool, optional): Print additional information during initialization. Defaults to True.
        """
        raise NotImplementedError("RTDETRDetectionModel is not implemented yet. Only DetectionModel is currently supported.")

    def init_criterion(self):
        """Initialize the loss criterion for the RTDETRDetectionModel."""
        from nkyolo.models.utils.loss import RTDETRDetectionLoss

        return RTDETRDetectionLoss(nc=self.nc, use_vfl=True)

    def loss(self, batch, preds=None):
        """
        Compute the loss for the given batch of data.

        Args:
            batch (dict): Dictionary containing image and label data.
            preds (jt.Var, optional): Precomputed model predictions. Defaults to None.

        Returns:
            (tuple): A tuple containing the total loss and main three losses in a tensor.
        """
        if self.criterion is None:
            self.criterion = self.init_criterion()

        img = batch["img"]
        # NOTE: preprocess gt_bbox and gt_labels to list.
        bs = len(img)
        batch_idx = batch["batch_idx"]
        gt_groups = [(batch_idx == i).sum().item() for i in range(bs)]
        targets = {
            "cls": batch["cls"].to(device=img.device, dtype=jt.int64).view(-1),
            "bboxes": batch["bboxes"].to(device=img.device),
            "batch_idx": batch_idx.to(device=img.device, dtype=jt.int64).view(-1),
            "gt_groups": gt_groups,
        }


        preds = self.predict(img, batch=targets) if preds is None else preds
        dec_bboxes, dec_scores, enc_bboxes, enc_scores, dn_meta = preds if self.training else preds[1]
        if dn_meta is None:
            dn_bboxes, dn_scores = None, None
        else:
            dn_bboxes, dec_bboxes = jt.split(dec_bboxes, dn_meta["dn_num_split"], dim=2)
            dn_scores, dec_scores = jt.split(dec_scores, dn_meta["dn_num_split"], dim=2)

        dec_bboxes = jt.concat([enc_bboxes.unsqueeze(0), dec_bboxes])  # (7, bs, 300, 4)
        dec_scores = jt.concat([enc_scores.unsqueeze(0), dec_scores])

        loss = self.criterion(
            (dec_bboxes, dec_scores), targets, dn_bboxes=dn_bboxes, dn_scores=dn_scores, dn_meta=dn_meta
        )
        # NOTE: There are like 12 losses in RTDETR, backward with all losses but only show the main three losses.
        return sum(loss.values()), jt.Var(
            [loss[k].detach() for k in ["loss_giou", "loss_class", "loss_bbox"]], device=img.device
        )

    def predict(self, x, profile=False, visualize=False, batch=None, augment=False, embed=None):
        """
        Perform a execute pass through the model.

        Args:
            x (jt.Var): The input tensor.
            profile (bool, optional): If True, profile the computation time for each layer. Defaults to False.
            visualize (bool, optional): If True, save feature maps for visualization. Defaults to False.
            batch (dict, optional): Ground truth data for evaluation. Defaults to None.
            augment (bool, optional): If True, perform data augmentation during inference. Defaults to False.
            embed (list, optional): A list of feature vectors/embeddings to return.

        Returns:
            (jt.Var): Model's output tensor.
        """
        y, dt, embeddings = [], [], []  # outputs
        embed = frozenset(embed) if embed is not None else {-1}
        max_idx = max(embed)
        for m in self.model[:-1]:  # excluding the head part
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
            if profile:
                self._profile_one_layer(m, x, dt)
            x = m(x)  # run
            y.append(x if m.i in self.save else None)  # save output
            if visualize:
                feature_visualization(x, m.type, m.i, save_dir=visualize)
            if m.i in embed:
                embeddings.append(nn.functional.adaptive_avg_pool2d(x, (1, 1)).squeeze(-1).squeeze(-1))  # flatten
                if m.i == max_idx:
                    return jt.unbind(jt.concat(embeddings, 1), dim=0)
        head = self.model[-1]
        x = head([y[j] for j in head.f], batch)  # head inference
        return x


class WorldModel(DetectionModel):
    """YOLO World Model (supports YOLOv5, v8, v9, v10, v11, etc.).
    
    TODO: This model is not yet implemented. Only DetectionModel is currently supported.
    """
    # TODO: Not implemented yet - YOLO-World task support

    def __init__(self, cfg="yolov8s-world.yaml", ch=3, nc=None, verbose=True):
        """Initialize YOLO World model with given config and parameters."""
        raise NotImplementedError("WorldModel is not implemented yet. Only DetectionModel is currently supported.")

    def set_classes(self, text, batch=80, cache_clip_model=True):
        """Set classes in advance so that model could do offline-inference without clip model."""
        if importlib.util.find_spec("clip") is None:
            check_requirements("git+https://github.com/ultralytics/CLIP.git")
        import clip

        if self.clip_model is None and cache_clip_model:
            self.clip_model = clip.load("ViT-B/32")[0]
        model = self.clip_model if cache_clip_model else clip.load("ViT-B/32")[0]
        device = next(model.parameters()).device
        text_token = clip.tokenize(text).to(device)
        txt_feats = [model.encode_text(token).detach() for token in text_token.split(batch)]
        txt_feats = txt_feats[0] if len(txt_feats) == 1 else jt.concat(txt_feats, dim=0)
        txt_feats = txt_feats / txt_feats.norm(p=2, dim=-1, keepdim=True)
        self.txt_feats = txt_feats.reshape(-1, len(text), txt_feats.shape[-1])
        self.model[-1].nc = len(text)

    def predict(self, x, profile=False, visualize=False, txt_feats=None, augment=False, embed=None):
        """
        Perform a execute pass through the model.

        Args:
            x (jt.Var): The input tensor.
            profile (bool, optional): If True, profile the computation time for each layer. Defaults to False.
            visualize (bool, optional): If True, save feature maps for visualization. Defaults to False.
            txt_feats (jt.Var): The text features, use it if it's given. Defaults to None.
            augment (bool, optional): If True, perform data augmentation during inference. Defaults to False.
            embed (list, optional): A list of feature vectors/embeddings to return.

        Returns:
            (jt.Var): Model's output tensor.
        """
        txt_feats = (self.txt_feats if txt_feats is None else txt_feats).to(device=x.device, dtype=x.dtype)
        if len(txt_feats) != len(x):
            txt_feats = txt_feats.repeat(len(x), 1, 1)
        ori_txt_feats = txt_feats.clone()
        y, dt, embeddings = [], [], []  # outputs
        embed = frozenset(embed) if embed is not None else {-1}
        max_idx = max(embed)
        for m in self.model:  # excluding the head part
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
            if profile:
                self._profile_one_layer(m, x, dt)
            if isinstance(m, C2fAttn):
                x = m(x, txt_feats)
            elif isinstance(m, WorldDetect):
                x = m(x, ori_txt_feats)
            elif isinstance(m, ImagePoolingAttn):
                txt_feats = m(x, txt_feats)
            else:
                x = m(x)  # run

            y.append(x if m.i in self.save else None)  # save output
            if visualize:
                feature_visualization(x, m.type, m.i, save_dir=visualize)
            if m.i in embed:
                embeddings.append(nn.functional.adaptive_avg_pool2d(x, (1, 1)).squeeze(-1).squeeze(-1))  # flatten
                if m.i == max_idx:
                    return jt.unbind(jt.concat(embeddings, 1), dim=0)
        return x

    def loss(self, batch, preds=None):
        """
        Compute loss.

        Args:
            batch (dict): Batch to compute loss on.
            preds (jt.Var | List[jt.Var]): Predictions.
        """
        if self.criterion is None:
            self.criterion = self.init_criterion()

        if preds is None:
            preds = self.execute(batch["img"], txt_feats=batch["txt_feats"])
        return self.criterion(preds, batch)


class Ensemble(nn.ModuleList):
    """Ensemble of models."""

    def __init__(self):
        """Initialize an ensemble of models."""
        super().__init__()

    def execute(self, x, augment=False, profile=False, visualize=False):
        """Function generates the YOLO network's final layer."""
        y = [module(x, augment, profile, visualize)[0] for module in self]
        y = jt.concat(y, 2)  # nms ensemble, y shape(B, HW, C)
        return y, None  # inference, train output


# Functions ------------------------------------------------------------------------------------------------------------


@contextlib.contextmanager
def temporary_modules(modules=None, attributes=None):
    """
    Context manager for temporarily adding or modifying modules in Python's module cache (`sys.modules`).

    This function can be used to change the module paths during runtime. It's useful when refactoring code,
    where you've moved a module from one location to another, but you still want to support the old import
    paths for backwards compatibility.

    Args:
        modules (dict, optional): A dictionary mapping old module paths to new module paths.
        attributes (dict, optional): A dictionary mapping old module attributes to new module attributes.

    Example:
        ```python
        with temporary_modules({"old.module": "new.module"}, {"old.module.attribute": "new.module.attribute"}):
            import old.module  # this will now import new.module
            from old.module import attribute  # this will now import new.module.attribute
        ```

    Note:
        The changes are only in effect inside the context manager and are undone once the context manager exits.
        Be aware that directly manipulating `sys.modules` can lead to unpredictable results, especially in larger
        applications or libraries. Use this function with caution.
    """
    if modules is None:
        modules = {}
    if attributes is None:
        attributes = {}
    import sys
    from importlib import import_module

    # Set attributes in sys.modules under their old name
    for old, new in attributes.items():
        old_module, old_attr = old.rsplit(".", 1)
        new_module, new_attr = new.rsplit(".", 1)
        setattr(import_module(old_module), old_attr, getattr(import_module(new_module), new_attr))

    # Set modules in sys.modules under their old name
    for old, new in modules.items():
        sys.modules[old] = import_module(new)

    yield
    # Remove the temporary module paths
    for old in modules:
        if old in sys.modules:
            del sys.modules[old]


class SafeClass:
    """A placeholder class to replace unknown classes during unpickling."""

    def __init__(self, *args, **kwargs):
        """Initialize SafeClass instance, ignoring all arguments."""
        pass

    def __call__(self, *args, **kwargs):
        """Run SafeClass instance, ignoring all arguments."""
        pass


class SafeUnpickler(pickle.Unpickler):
    """Custom Unpickler that replaces unknown classes with SafeClass."""

    def find_class(self, module, name):
        """Attempt to find a class, returning SafeClass if not among safe modules."""
        safe_modules = (
            "jittor",
            "jt",
            "collections",
            "collections.abc",
            "builtins",
            "math",
            "numpy",
            # Add other modules considered safe
        )
        if module in safe_modules:
            return super().find_class(module, name)
        else:
            return SafeClass


def _sidecar_yaml_path(weight):
    path = Path(weight)
    for ext in (".yaml", ".yml"):
        candidate = path.with_suffix(ext)
        if candidate.exists():
            return str(candidate)
    return ""


def _normalize_yaml_config(yaml_config, args):
    if isinstance(yaml_config, str):
        yaml_config = yaml_model_load(yaml_config)
    if not isinstance(yaml_config, dict):
        raise TypeError(f"model_yaml must be a dict or str, got {type(yaml_config)}: {yaml_config}")
    yaml_config = deepcopy(yaml_config)
    if "ch" not in yaml_config:
        yaml_config["ch"] = 3
    if "nc" not in yaml_config:
        yaml_config["nc"] = args.get("nc", 80) if args else 80
    if "scales" in yaml_config and isinstance(yaml_config["scales"], dict):
        scale = yaml_config.get("scale")
        if not scale and args:
            scale = args.get("scale")
        if not scale:
            scale = next(iter(yaml_config["scales"].keys()))
        yaml_config["scale"] = scale
    return yaml_config


def _build_model_from_yaml(task, yaml_config):
    if task == "detect":
        return DetectionModel(cfg=yaml_config, verbose=False)
    raise NotImplementedError(f"Task '{task}' is not implemented yet.")


def _load_state_dict_strict(model, state_dict):
    def _state_shape(val):
        if isinstance(val, jt.Var):
            return tuple(val.shape)
        if _TORCH_AVAILABLE and isinstance(val, torch.Tensor):
            return tuple(val.shape)
        if isinstance(val, np.ndarray):
            return tuple(val.shape)
        return None

    model_sd = model.state_dict()
    droppable_suffixes = (".anchors", ".strides")
    cleaned = {}
    mismatched = []

    for k, v in state_dict.items():
        if k.endswith(droppable_suffixes):
            continue
        if k not in model_sd:
            # Drop unexpected keys (not used by current model definition).
            continue
        ms = _state_shape(model_sd[k])
        vs = _state_shape(v)
        if ms is not None and vs is not None and ms != vs:
            mismatched.append((k, ms, vs))
            continue
        cleaned[k] = v

    if mismatched:
        mismatch_str = ", ".join(f"{k}: {ms} vs {vs}" for k, ms, vs in mismatched[:5])
        raise RuntimeError(
            f"State dict shape mismatch for {len(mismatched)} keys. Examples: {mismatch_str}"
        )

    model.load_state_dict(cleaned)


def _to_numpy(obj):
    if isinstance(obj, dict):
        return {k: _to_numpy(v) for k, v in obj.items()}
    if isinstance(obj, jt.Var):
        return obj.numpy()
    if _TORCH_AVAILABLE and isinstance(obj, torch.Tensor):
        return obj.detach().cpu().numpy()
    if isinstance(obj, np.ndarray):
        return obj
    return obj


def _torch_safe_load(weight):
    if not _TORCH_AVAILABLE:
        raise ModuleNotFoundError("torch is required to load .pt/.pth weights. Please install torch.")
    from nkyolo.utils.downloads import attempt_download_asset

    file = attempt_download_asset(weight)
    return torch.load(file, map_location="cpu"), file


def _extract_torch_state(ckpt):
    if isinstance(ckpt, torch.nn.Module):
        return ckpt.state_dict(), ckpt.yaml, {}
    if not isinstance(ckpt, dict):
        raise TypeError(f"Unsupported torch checkpoint type: {type(ckpt)}")
    train_args = ckpt.get("train_args", {})
    model_yaml = ckpt.get("model_yaml") or ckpt.get("yaml")
    model_obj = ckpt.get("ema") or ckpt.get("model")
    state_source = None
    if isinstance(model_obj, torch.nn.Module):
        model_yaml = model_obj.yaml
        state_source = model_obj.state_dict()
    elif isinstance(model_obj, dict):
        state_source = model_obj
    elif isinstance(ckpt.get("state_dict"), dict):
        state_source = ckpt.get("state_dict")
    elif isinstance(ckpt.get("model_state_dict"), dict):
        state_source = ckpt.get("model_state_dict")
    if state_source is None:
        raise ValueError("No state_dict found in torch checkpoint.")
    return state_source, model_yaml, train_args

def jittor_safe_load(weight, safe_only=False):
    """
    Load a Jittor checkpoint saved in .pkl format.

    Args:
        weight (str): Path to the Jittor checkpoint (.pkl).
        safe_only (bool): If True, replace unknown classes with SafeClass during loading.

    Returns:
        ckpt (dict): Loaded checkpoint.
        file (str): Resolved file path.
    """
    from nkyolo.utils.downloads import attempt_download_asset

    check_suffix(file=weight, suffix=(".pkl",))
    file = attempt_download_asset(weight)
    with temporary_modules(
        modules={
            "nkyolo.yolo.utils": "nkyolo.utils",
            "nkyolo.yolo.v8": "nkyolo.models.yolo",
            "nkyolo.yolo.data": "nkyolo.data",
        },
        attributes={
            "nkyolo.nn.modules.block.Silence": "jittor.nn.Identity",  # YOLOv9e
            "nkyolo.nn.tasks.YOLOv10DetectionModel": "nkyolo.nn.tasks.DetectionModel",  # YOLOv10
            "nkyolo.utils.loss.v10DetectLoss": "nkyolo.utils.loss.E2EDetectLoss",  # YOLOv10
        },
    ):
        if safe_only:
            safe_pickle = types.ModuleType("safe_pickle")
            safe_pickle.Unpickler = SafeUnpickler
            safe_pickle.load = lambda file_obj: SafeUnpickler(file_obj).load()
            with open(file, "rb") as f:
                ckpt = jt.load(f, pickle_module=safe_pickle)
        else:
            ckpt = jt.load(file)

    if isinstance(ckpt, nn.Module):
        ckpt = {"model": ckpt.state_dict()}
    if not isinstance(ckpt, dict):
        raise TypeError(
            f"Checkpoint '{weight}' is not a valid dictionary. Use model.save('filename.pkl') for NK-YOLO checkpoints."
        )

    protected_keys = {"model_yaml", "train_args", "train_metrics", "train_results", "names", "nc"}
    protected_data = {k: ckpt.pop(k) for k in protected_keys if k in ckpt}
    ckpt = _to_numpy(ckpt)
    ckpt.update(protected_data)
    return ckpt, file

def _load_weight_entry(weight, device=None, inplace=True, fuse=False):
    if isinstance(weight, nn.Module):
        model = weight
        ckpt = {}
        weight_path = ""
        args = DEFAULT_CFG_DICT
    else:
        weight_path = str(weight)
        suffix = Path(weight_path).suffix.lower()
        if suffix == ".pkl":
            ckpt, weight_path = jittor_safe_load(weight_path)
            args = {**DEFAULT_CFG_DICT, **ckpt.get("train_args", {})}
            yaml_config = ckpt.get("model_yaml")
            if yaml_config is None:
                raise ValueError(f"Checkpoint '{weight_path}' is missing model_yaml. Re-save as .pkl with NK-YOLO.")
            yaml_config = _normalize_yaml_config(yaml_config, args)
            model = _build_model_from_yaml(args.get("task", "detect"), yaml_config)
            state_source = ckpt.get("ema") or ckpt.get("model")
            if state_source is None:
                raise KeyError(f"Checkpoint '{weight_path}' has no model/ema weights.")
            jittor_state = state_dict_to_jittor(state_source)
            _load_state_dict_strict(model, jittor_state)
        elif suffix in {".pt", ".pth"}:
            if not _TORCH_AVAILABLE and suffix == ".pt":
                raise ModuleNotFoundError("torch is required to load .pt weights. Please install torch.")
            if not _TORCH_AVAILABLE and suffix == ".pth":
                sidecar = _sidecar_yaml_path(weight_path)
                if not sidecar:
                    raise ModuleNotFoundError(
                        "torch is required to load .pth weights without a sidecar YAML. "
                        "Provide model.yaml next to the .pth file."
                    )
                args = DEFAULT_CFG_DICT
                yaml_config = _normalize_yaml_config(sidecar, args)
                model = _build_model_from_yaml(args.get("task", "detect"), yaml_config)
                model.load(weight_path)  # Jittor native loader for torch .pth state dict
                ckpt = {"model_yaml": yaml_config, "train_args": args}
            else:
                ckpt, weight_path = _torch_safe_load(weight_path)
                state_source, model_yaml, train_args = _extract_torch_state(ckpt)
                if not model_yaml:
                    sidecar = _sidecar_yaml_path(weight_path)
                    if sidecar:
                        model_yaml = sidecar
                if not model_yaml:
                    raise ValueError(f"Torch checkpoint '{weight_path}' is missing model_yaml/yaml.")
                args = {**DEFAULT_CFG_DICT, **train_args}
                yaml_config = _normalize_yaml_config(model_yaml, args)
                model = _build_model_from_yaml(args.get("task", "detect"), yaml_config)
                jittor_state = state_dict_to_jittor(state_source)
                _load_state_dict_strict(model, jittor_state)
                ckpt = {"model_yaml": model_yaml, "train_args": train_args}
        else:
            raise ValueError(f"Unsupported weights suffix '{suffix}'. Use .pkl (Jittor) or .pt/.pth (PyTorch).")

    model.args = {k: v for k, v in args.items() if k in DEFAULT_CFG_KEYS}
    model.pt_path = weight_path
    model.task = guess_model_task(model)
    model.inplace = inplace
    if isinstance(ckpt, dict):
        if "names" in ckpt and ckpt["names"] is not None:
            model.names = ckpt["names"]
        if "nc" in ckpt and ckpt["nc"] is not None:
            model.nc = ckpt["nc"]
    if device is not None and _TORCH_AVAILABLE:
        # Only torch models support .to()
        if isinstance(model, torch.nn.Module):
            model.to(device)
    if fuse:
        model = model.fuse()
    model.eval()
    return model, ckpt


def attempt_load_weights(weights, device=None, inplace=True, fuse=False):
    """Loads an ensemble of models weights=[a,b,c] or a single model weights=[a] or weights=a."""
    ensemble = Ensemble()
    for w in weights if isinstance(weights, list) else [weights]:
        model, _ = _load_weight_entry(w, device=device, inplace=inplace, fuse=fuse)
        ensemble.append(model)

    if len(ensemble) == 1:
        return ensemble[-1]

    LOGGER.info(f"Ensemble created with {weights}\n")
    ensemble.names = ensemble[0].names
    ensemble.nc = ensemble[0].nc
    ensemble.yaml = ensemble[0].yaml
    ensemble.stride = ensemble[int(jt.argmax(jt.Var([m.stride.max() for m in ensemble])))].stride
    assert all(ensemble[0].nc == m.nc for m in ensemble), f"Models differ in class counts {[m.nc for m in ensemble]}"
    return ensemble


def attempt_load_one_weight(weight, device=None, inplace=True, fuse=False):
    """Loads a single model weights."""
    return _load_weight_entry(weight, device=device, inplace=inplace, fuse=fuse)


def parse_model(d, ch, verbose=True):  # model_dict, input_channels(3)
    """Parse a YOLO model.yaml dictionary into a Jittor model."""
    import ast

    # Args
    legacy = True  # backward compatibility for v3/v5/v8/v9 models
    max_channels = float("inf")
    nc, act, scales = (d.get(x) for x in ("nc", "activation", "scales"))
    depth, width, kpt_shape = (d.get(x, 1.0) for x in ("depth_multiple", "width_multiple", "kpt_shape"))
    if scales:
        scale = d.get("scale")
        if not scale:
            scale = tuple(scales.keys())[0]
            LOGGER.warning(
                f"WARNING ⚠️ no model scale passed. Assuming scale='{scale}' from scales dict. "
                f"Available scales: {list(scales.keys())}. This may cause shape mismatch if incorrect."
            )
        if scale not in scales:
            available = list(scales.keys())
            LOGGER.error(
                f"ERROR ❌ scale '{scale}' not found in scales dict. Available scales: {available}. "
                f"Using first available scale '{available[0]}' instead."
            )
            scale = available[0]
        depth, width, max_channels = scales[scale]
        LOGGER.info(f"Using scale '{scale}': width_multiple={width}, depth_multiple={depth}, max_channels={max_channels}")

        activation_mapping = {
            'SiLU': nn.SiLU,
            'silu': nn.SiLU,
            'ReLU': nn.ReLU,
            'relu': nn.ReLU,
            'LeakyReLU': nn.LeakyReLU,
            'leakyrelu': nn.LeakyReLU,

        }
        
        if 'act' in d:
            act = d['act']
            if isinstance(act, str):
                if act in activation_mapping:
                    Conv.default_act = activation_mapping[act]()
                else:
                    if act not in nn.__dict__:
                        raise ValueError(f"Activation function '{act}' not found in jittor.nn")
                    Conv.default_act = getattr(nn, act)()
            else:
                Conv.default_act = act

    if verbose:
        LOGGER.info(f"\n{'':>3}{'from':>20}{'n':>3}{'params':>10}  {'module':<45}{'arguments':<30}")
    ch = [ch]
    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
    for i, (f, n, m, args) in enumerate(d["backbone"] + d["head"]):  # from, number, module, args
        m = getattr(jt.nn, m[3:]) if "nn." in m else globals()[m]  # get module
        for j, a in enumerate(args):
            if isinstance(a, str):
                with contextlib.suppress(ValueError):
                    args[j] = locals()[a] if a in locals() else ast.literal_eval(a)
        n = n_ = max(round(n * depth), 1) if n > 1 else n  # depth gain
        if m in {
            Classify,
            Conv,
            ConvTranspose,
            GhostConv,
            Bottleneck,
            GhostBottleneck,
            SPP,
            SPPF,
            C2fPSA,
            C2PSA,
            DWConv,
            Focus,
            BottleneckCSP,
            C1,
            C2,
            C2f,
            C3k2,
            RepNCSPELAN4,
            ELAN1,
            ADown,
            AConv,
            SPPELAN,
            C2fAttn,
            C3,
            C3TR,
            C3Ghost,
            nn.ConvTranspose2d,
            DWConvTranspose2d,
            C3x,
            RepC3,
            PSA,
            SCDown,
            C2fCIB,
            MSBlock,
            A2C2f
        }:
            if m in {MSBlock} and isinstance(f, list):
                c1, c2 = ch[f[-1]], args[0]
            else:
                c1, c2 = ch[f], args[0]
            if c2 != nc:  # if c2 not equal to number of classes (i.e. for Classify() output)
                c2 = make_divisible(min(c2, max_channels) * width, 8)
            if m is C2fAttn:
                args[1] = make_divisible(min(args[1], max_channels // 2) * width, 8)  # embed channels
                args[2] = int(
                    max(round(min(args[2], max_channels // 2 // 32)) * width, 1) if args[2] > 1 else args[2]
                )  # num heads

            args = [c1, c2, *args[1:]]
            if m in {
                BottleneckCSP,
                C1,
                C2,
                C2f,
                C3k2,
                C2fAttn,
                C3,
                C3TR,
                C3Ghost,
                C3x,
                RepC3,
                C2fPSA,
                C2fCIB,
                C2PSA,
                A2C2f
            }:
                args.insert(2, n)  # number of repeats
                n = 1
            if m is C3k2:  # for M/L/X sizes
                legacy = False
                if scale in "mlx":
                    args[3] = True
            if m is A2C2f:
                legacy = False
                if scale in "lx":  # for L/X sizes
                    args.extend((True, 1.2))
        elif m is AIFI:
            args = [ch[f], *args]
        elif m in {HGStem, HGBlock}:
            c1, cm, c2 = ch[f], args[0], args[1]
            args = [c1, cm, c2, *args[2:]]
            if m is HGBlock:
                args.insert(4, n)  # number of repeats
                n = 1
        elif m is ResNetLayer:
            c2 = args[1] if args[3] else args[1] * 4
        elif m is nn.BatchNorm2d:
            args = [ch[f]]
        elif m is Concat:
            c2 = sum(ch[x] for x in f)
        elif m in {Detect, WorldDetect, Segment, Pose, OBB, ImagePoolingAttn, v10Detect}:
            args.append([ch[x] for x in f])
            if m is Segment:  # or m is YOLOESegment (when implemented)
                args[2] = make_divisible(min(args[2], max_channels) * width, 8)
            if m in {Detect, Segment, Pose, OBB}:  # Note: YOLOEDetect and YOLOESegment would also set legacy when implemented
                m.legacy = legacy
        elif m is RTDETRDecoder:  # special case, channels arg must be passed in index 1
            args.insert(1, [ch[x] for x in f])
        elif m in {CBLinear, TorchVision, Index}:
            c2 = args[0]
            c1 = ch[f]
            args = [c1, c2, *args[1:]]
        elif m is CBFuse:
            c2 = ch[f[-1]]
        else:
            c2 = ch[f]
        m_ = nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)  # module
        t = str(m)[8:-2].replace("__main__.", "")  # module type
        m_.np = sum(x.numel() for x in m_.parameters())  # number params
        m_.i, m_.f, m_.type = i, f, t  # attach index, 'from' index, type
        if verbose:
            LOGGER.info(f"{i:>3}{str(f):>20}{n_:>3}{m_.np:10.0f}  {t:<45}{str(args):<30}")  # print
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        layers.append(m_)
        if i == 0:
            ch = []
        ch.append(c2)
    return nn.Sequential(*layers), sorted(save)


def yaml_model_load(path):
    """Load a YOLO model from a YAML file (supports YOLOv5, v8, v9, v10, v11, etc.)."""
    path = Path(path)
    # Handle P6 models (yolov5n6, yolov8n6, yolov9n6, yolov10n6, yolov11n6, etc.)
    if path.stem in (f"yolov{d}{x}6" for x in "nsmlx" for d in (5, 8, 9, 10, 11)):
        new_stem = re.sub(r"(\d+)([nslmx])6(.+)?$", r"\1\2-p6\3", path.stem)
        LOGGER.warning(f"WARNING ⚠️ NK-YOLO P6 models now use -p6 suffix. Renaming {path.stem} to {new_stem}.")
        path = path.with_name(new_stem + path.suffix)

    # Unify model paths (e.g., yolov8x.yaml -> yolov8.yaml, yolov11n.yaml -> yolov11.yaml)
    unified_path = re.sub(r"(\d+)([nslmx])(.+)?$", r"\1\3", str(path))
    yaml_file = check_yaml(unified_path, hard=False) or check_yaml(path)
    d = yaml_load(yaml_file)  # model dict
    d["scale"] = guess_model_scale(path)
    d["yaml_file"] = str(path)
    return d


def guess_model_scale(model_path):
    """
    Extract the size character n, s, m, l, or x of the model's scale from the model path.
    Supports YOLOv5, v8, v9, v10, v11, and other versions.

    Args:
        model_path (str | Path): The path to the YOLO model's YAML file.

    Returns:
        (str): The size character of the model's scale (n, s, m, l, or x).
    """
    # Match patterns like yolov8n, yolov11s, yolo-e-m, etc.
    match = re.search(r"yolo(e-)?[v]?\d+([nslmx])", Path(model_path).stem)
    return match.group(2) if match else ""


def guess_model_task(model):
    """
    Guess the task of a Jittor model from its architecture or configuration.

    Args:
        model (nn.Module | dict): Jittor model or model configuration in YAML format.

    Returns:
        (str): Task of the model ('detect', 'segment', 'classify', 'pose').

    Raises:
        SyntaxError: If the task of the model could not be determined.
    """

    def cfg2task(cfg):
        """Guess from YAML dictionary."""
        m = cfg["head"][-1][-2].lower()  # output module name
        if m in {"classify", "classifier", "cls", "fc"}:
            return "classify"
        if "detect" in m:
            return "detect"
        if "segment" in m:
            return "segment"
        if m == "pose":
            return "pose"
        if m == "obb":
            return "obb"
        return "detect"  # Default: use detect task

    # Guess from model cfg
    if isinstance(model, dict):
        with contextlib.suppress(Exception):
            return cfg2task(model)
    # Guess from Jittor model
    if isinstance(model, nn.Module):  # Jittor model
        for x in "model.args", "model.model.args", "model.model.model.args":
            with contextlib.suppress(Exception):
                return eval(x)["task"]
        for x in "model.yaml", "model.model.yaml", "model.model.model.yaml":
            with contextlib.suppress(Exception):
                return cfg2task(eval(x))
        for m in model.modules():
            if isinstance(m, Segment):
                return "segment"
            elif isinstance(m, Classify):
                return "classify"
            elif isinstance(m, Pose):
                return "pose"
            elif isinstance(m, OBB):
                return "obb"
            elif isinstance(m, (Detect, WorldDetect, v10Detect)):
                return "detect"
            # Note: YOLOEDetect and YOLOESegment are not yet implemented
            # elif isinstance(m, (YOLOEDetect, YOLOESegment)):
            #     return "detect" if isinstance(m, YOLOEDetect) else "segment"

    # Guess from model filename
    if isinstance(model, (str, Path)):
        model = Path(model)
        if "-seg" in model.stem or "segment" in model.parts:
            return "segment"
        elif "-cls" in model.stem or "classify" in model.parts:
            return "classify"
        elif "-pose" in model.stem or "pose" in model.parts:
            return "pose"
        elif "-obb" in model.stem or "obb" in model.parts:
            return "obb"
        elif "detect" in model.parts:
            return "detect"

    # Unable to determine task from model
    LOGGER.warning(
        "WARNING ⚠️ Unable to automatically guess model task, assuming 'task=detect'. "
        "Note: Only 'detect' task is currently supported. Other tasks (segment, classify, pose, obb) are TODO."
    )
    return "detect"  # assume detect (only supported task)
