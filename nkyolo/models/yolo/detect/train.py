# NK-YOLO 🚀 AGPL-3.0 License
# Refer to https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/yolo/detect/train.py

import math
import random
from copy import copy
import jittor as jt
import numpy as np
import jittor.nn as nn

from nkyolo.data import build_dataloader, build_yolo_dataset
from nkyolo.engine.trainer import BaseTrainer
from nkyolo.models import yolo
from nkyolo.nn.tasks import DetectionModel
from nkyolo.utils import LOGGER, RANK, colorstr
from nkyolo.utils.plotting import plot_images, plot_labels, plot_results

class DetectionTrainer(BaseTrainer):
    """
    A class extending the BaseTrainer class for training based on a detection model.

    Example:
        ```python
        from nkyolo.models.yolo.detect import DetectionTrainer

        args = dict(model="yolo11n.pkl", data="coco8.yaml", epochs=3)
        trainer = DetectionTrainer(overrides=args)
        trainer.train()
        ```
    """

    def build_dataset(self, img_path, mode="train", batch=None):
        """Build YOLO Dataset.
        
        Args:
            img_path (str): Path to folder containing images.
            mode (str): Dataset mode, either "train" or "val".
            batch (int, optional): Batch size, used for rectangular batching configuration.
        
        Returns:
            YOLODataset: Configured YOLO dataset instance.
        """
        return build_yolo_dataset(self.args, img_path, batch, self.data, mode=mode, rect=mode == "val", stride=self.model.stride.max())

    def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode="train"):
        """Construct and return dataloader.
        
        Args:
            dataset_path (str): Path to dataset directory.
            batch_size (int): Number of samples per batch.
            rank (int): Process rank for distributed training.
            mode (str): Dataset mode, either "train" or "val".
        
        Returns:
            InfiniteDataLoader: Configured data loader.
        """
        assert mode in {"train", "val"}, f"Mode must be 'train' or 'val', not {mode}."
        dataset = self.build_dataset(dataset_path, mode, batch_size)
        shuffle = mode == "train"
        if getattr(dataset, "rect", False) and shuffle:
            LOGGER.warning("WARNING ⚠️ 'rect=True' is incompatible with DataLoader shuffle, setting shuffle=False")
            shuffle = False
        # Support dedicated validation workers; fallback to train workers.
        workers = self.args.workers
        if mode == "val":
            val_workers = int(getattr(self.args, "val_workers", -1) or -1)
            if val_workers >= 0:
                workers = val_workers
        return build_dataloader(dataset, batch_size, workers, shuffle, rank, buffer_size=None)

    def preprocess_batch(self, batch):
        """Preprocesses batch by scaling to [0, 1] and optionally applying multi-scale training.
        
        Args:
            batch (dict): Batch dictionary containing "img" key.
        
        Returns:
            dict: Preprocessed batch.
        """
        img = batch["img"].to(self.device, non_blocking=True)
        # Align input dtype with model parameters to avoid mixed-precision conv errors.
        params_iter = iter(self.model.parameters())
        first_param = next(params_iter, None)
        param_dtype = first_param.dtype if first_param is not None else None
        if param_dtype is not None and "float16" in str(param_dtype):
            img = img.half()
        else:
            img = img.float()
        batch["img"] = img / 255
        if self.args.multi_scale:
            imgs = batch["img"]
            sz = (
                random.randrange(int(self.args.imgsz * 0.5), int(self.args.imgsz * 1.5 + self.stride))
                // self.stride
                * self.stride
            )
            sf = sz / max(imgs.shape[2:])
            if sf != 1:
                ns = [
                    math.ceil(x * sf / self.stride) * self.stride for x in imgs.shape[2:]
                ]
                imgs = nn.functional.interpolate(imgs, size=ns, mode="bilinear", align_corners=False)
            batch["img"] = imgs
        return batch

    def set_model_attributes(self):
        """Set model attributes from data configuration and training arguments."""
        # nl = de_parallel(self.model).model[-1].nl  # number of detection layers (to scale hyps)
        # self.args.box *= 3 / nl  # scale to layers
        # self.args.cls *= self.data["nc"] / 80 * 3 / nl  # scale to classes and layers
        # self.args.cls *= (self.args.imgsz / 640) ** 2 * 3 / nl  # scale to image size and layers
        self.model.nc = self.data["nc"]  # attach number of classes to model
        self.model.names = self.data["names"]  # attach class names to model
        self.model.args = self.args  # attach hyperparameters to model
        # TODO: self.model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device) * nc

    def get_model(self, cfg=None, weights=None, verbose=True):
        """Return a YOLO detection model.
        
        Args:
            cfg (str, optional): Path to model configuration file.
            weights (str, optional): Path to pretrained weights file.
            verbose (bool): Whether to print model information.
        
        Returns:
            DetectionModel: Configured YOLO detection model.
        """
        model = DetectionModel(cfg, nc=self.data["nc"], verbose=verbose and RANK in {-1, 0})
        if weights:
            model.load(weights)
        return model

    def get_validator(self):
        """Returns a DetectionValidator for YOLO model validation.
        
        Returns:
            DetectionValidator: Configured validator instance.
        """
        self.loss_names = "box_loss", "cls_loss", "dfl_loss"
        return yolo.detect.DetectionValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )

    def label_loss_items(self, loss_items=None, prefix="train"):
        """Returns a loss dict with labelled training loss items tensor.
        
        Not needed for classification but necessary for segmentation & detection.
        
        Args:
            loss_items (list, optional): List of loss tensor values.
            prefix (str): Prefix to add to loss names (e.g., "train" or "val").
        
        Returns:
            dict or list: Dictionary mapping loss names to values, or list of loss names.
        """
        keys = [f"{prefix}/{x}" for x in self.loss_names]
        if loss_items is not None:
            loss_items = [round(float(x), 5) for x in loss_items]  # convert tensors to 5 decimal place floats
            return dict(zip(keys, loss_items))
        else:
            return keys

    def progress_string(self):
        """Returns formatted string header for training progress display.
        
        Returns:
            str: Formatted string with column headers (Epoch, GPU_mem (if available), losses, Instances, Size).
        """
        # Check if memory is available
        memory_str = self._get_memory_str()
        has_memory = bool(memory_str)
        
        headers = ["Epoch"]
        if has_memory:
            headers.append("GPU_mem")
        headers.extend(self.loss_names)
        headers.extend(["Instances", "Size"])

        header = " ".join(f"{h:>11}" for h in headers)
        return "\n" + colorstr("bold", header)

    def plot_training_samples(self, batch, ni):
        """Plots training samples with annotations.
        
        Args:
            batch (dict): Batch dictionary containing images, labels, and annotations.
            ni (int): Batch number or iteration index for file naming.
        """
        cls_tensor = jt.array(batch["cls"]).squeeze(-1)
        bboxes_tensor = jt.array(batch["bboxes"])
        plot_images(
            images=batch["img"],
            batch_idx=batch["batch_idx"],
            cls=cls_tensor,
            bboxes=bboxes_tensor,
            paths=batch["im_file"],
            fname=self.save_dir / f"train_batch{ni}.jpg",
            on_plot=self.on_plot,
        )

    def plot_metrics(self):
        """Plots metrics from CSV log file (saved as results.png)."""
        plot_results(file=self.csv, on_plot=self.on_plot)

    def plot_training_labels(self):
        """Create visualization plots of training label distributions."""
        boxes = np.concatenate([lb["bboxes"] for lb in self.train_loader.dataset.labels], 0)
        cls = np.concatenate([lb["cls"] for lb in self.train_loader.dataset.labels], 0)
        plot_labels(boxes, cls.squeeze(), names=self.data["names"], save_dir=self.save_dir, on_plot=self.on_plot)

    def auto_batch(self):
        """Calculate optimal batch size based on model memory requirements.
        
        Returns:
            int: Optimal batch size that fits in available memory.
        """
        train_dataset = self.build_dataset(self.trainset, mode="train", batch=16)
        # Multiply by 4 for mosaic augmentation
        max_num_obj = max(len(label["cls"]) for label in train_dataset.labels) * 4
        return super().auto_batch(max_num_obj)
