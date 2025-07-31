# NK-YOLO 🚀 AGPL-3.0 License
# Refer to https://github.com/ultralytics/ultralytics/blob/main/ultralytics/data/build.py

import os
import random
from pathlib import Path

import numpy as np
import jittor as jt
from jittor.dataset import Dataset
from PIL import Image

from nkyolo.data.dataset import YOLODataset
from nkyolo.data.loaders import (
    LOADERS,
    LoadImagesAndVideos,
    LoadPilAndNumpy,
    LoadScreenshots,
    LoadStreams,
    LoadTensor,
    SourceTypes,
    autocast_list,
)
from nkyolo.data.utils import IMG_FORMATS, PIN_MEMORY, VID_FORMATS
from nkyolo.utils import colorstr
from nkyolo.utils.checks import check_file


class InfiniteDataset(Dataset):
    """Dataset wrapper that repeats forever."""
    
    def __init__(self, dataset):
        """Initialize infinite dataset wrapper."""
        super().__init__()
        self.dataset = dataset
        
        # Set basic dataset attributes
        self.set_attrs(
            total_len=len(dataset),
            batch_size=1,
            shuffle=False,
            drop_last=False,
            num_workers=0,
            buffer_size=1
        )
        
        # Forward all attributes from the original dataset that we don't explicitly define
        self.__dict__.update({k: v for k, v in dataset.__dict__.items() if not hasattr(self, k)})
        
        # Set collate function directly 
        if hasattr(dataset, 'collate_fn'):
            self.collate_fn = dataset.collate_fn

    def __getitem__(self, index):
        """Get item at index from dataset."""
        return self.dataset[index % len(self.dataset)]

    # 新增：覆盖默认的collate_batch方法，使用自定义collate_fn
    def collate_batch(self, batch):
        return self.collate_fn(batch)


def collate_fn(batch):
    """Collates data samples into batches.
    
    Args:
        batch: List of samples from dataset
        
    Returns:
        Dictionary with batched data
    """
    n = len(batch)
    if n == 0:
        return None
    
    out = {}
    keys = batch[0].keys()
    
    for k in keys:
        values = [item[k] for item in batch]
        
        if k == "img":
            # Images should always be stacked
            out[k] = jt.stack(values, 0)
            
        elif k in ["masks", "keypoints", "bboxes", "cls", "segments"]:
            # These items may have different sizes per sample
            if isinstance(values[0], jt.Var):
                try:
                    # Try stacking if shapes match
                    out[k] = jt.stack(values, 0)
                except:
                    # Fall back to list if shapes don't match
                    out[k] = values
            else:
                out[k] = values
                
        elif k == "batch_idx":
            # Special handling for batch indices
            if isinstance(values[0], (list, tuple)):
                out[k] = []
                for i, v in enumerate(values):
                    out[k].extend([i] * len(v))
                out[k] = jt.array(out[k])
            else:
                out[k] = jt.array(values)
                
        elif isinstance(values[0], (int, float)):
            # Basic numeric types
            out[k] = jt.array(values)
            
        else:
            # Keep other types as lists
            out[k] = values
            
    return out

class InfiniteDataLoader:
    """
    DataLoader wrapper for infinite iteration.
    """
    def __init__(self, dataset, batch_size, shuffle=False, num_workers=0,
                 pin_memory=False, worker_init_fn=None, drop_last=False,
                 buffer_size=512, collate_fn=None):  # 默认 buffer_size 为 512
        """Initialize InfiniteDataLoader."""
        # Store original dataset
        self.original_dataset = dataset
        self.dataset = dataset  # 为了兼容性，保持dataset属性
        
        # Use dataset's collate_fn if available, otherwise use provided or default
        if collate_fn is None and hasattr(dataset, 'collate_fn'):
            collate_fn = dataset.collate_fn
        
        # Store configuration
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.worker_init_fn = worker_init_fn
        self.collate_fn = collate_fn
        self.shuffle = shuffle
        self.drop_last = drop_last
        
        # Calculate number of batches
        self.num_batches = len(dataset) // batch_size
        if not drop_last and len(dataset) % batch_size != 0:
            self.num_batches += 1

    def __len__(self):
        """Return length of dataset."""
        return self.num_batches

    def __iter__(self):
        """Return self as iterator."""
        # Create a new dataset for this iteration
        self.dataset = InfiniteDataset(self.original_dataset)
        
        # Set dataset attributes
        self.dataset.set_attrs(
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            drop_last=self.drop_last,
            num_workers=self.num_workers,
            buffer_size=512
        )
        
        if self.collate_fn:
            self.dataset.collate_fn = self.collate_fn
        
        # Create iterator
        self.iterator = self.dataset.__iter__()
        return self

    def __next__(self):
        """Get next batch."""
        try:
            batch = next(self.iterator)
        except StopIteration:
            raise StopIteration
        return batch

    def reset(self):
        """Reset iterator."""
        self.iterator = self.dataset.__iter__()


def seed_worker(worker_id):  # noqa
    """Set dataloader worker seed."""
    # Use Python's random and numpy instead of torch/jittor specific RNG
    seed = int(random.random() * 2**32)  # Generate random seed
    np.random.seed(seed + worker_id)
    random.seed(seed + worker_id)


def build_yolo_dataset(cfg, img_path, batch, data, mode="train", rect=False, stride=32, multi_modal=False):
    """Build YOLO Dataset."""
    return YOLODataset(
        img_path=img_path,
        imgsz=cfg.imgsz,
        batch_size=batch,
        augment=mode == "train",  # augmentation
        hyp=cfg,  # TODO: probably add a get_hyps_from_cfg function
        rect=cfg.rect or rect,  # rectangular batches
        cache=cfg.cache or None,
        single_cls=cfg.single_cls or False,
        stride=int(stride),
        pad=0.0 if mode == "train" else 0.5,
        prefix=colorstr(f"{mode}: "),
        task=cfg.task,
        classes=cfg.classes,
        data=data,
        fraction=cfg.fraction if mode == "train" else 1.0,
    )


def build_dataloader(dataset, batch, workers, shuffle=True, rank=-1, buffer_size=512):  # 修改默认 buffer_size 为 512
    """Return an InfiniteDataLoader for training or validation set."""
    batch = min(batch, len(dataset))
    workers = min(os.cpu_count() or 1, workers)  # Limit workers
    
    loader = InfiniteDataLoader(
        dataset=dataset,
        batch_size=batch,
        shuffle=shuffle and rank == -1,
        num_workers=workers,
        pin_memory=PIN_MEMORY,
        worker_init_fn=seed_worker,
        drop_last=False,
        buffer_size=buffer_size  # 使用更新后的 buffer_size
    )
    
    return loader


def check_source(source):
    """Check source type and return corresponding flag values."""
    webcam, screenshot, from_img, in_memory, tensor = False, False, False, False, False
    if isinstance(source, (str, int, Path)):  # int for local usb camera
        source = str(source)
        is_file = Path(source).suffix[1:] in (IMG_FORMATS | VID_FORMATS)
        is_url = source.lower().startswith(("https://", "http://", "rtsp://", "rtmp://", "tcp://"))
        webcam = source.isnumeric() or source.endswith(".streams") or (is_url and not is_file)
        screenshot = source.lower() == "screen"
        if is_url and is_file:
            source = check_file(source)  # download
    elif isinstance(source, LOADERS):
        in_memory = True
    elif isinstance(source, (list, tuple)):
        source = autocast_list(source)  # convert all list elements to PIL or np arrays
        from_img = True
    elif isinstance(source, (Image.Image, np.ndarray)):
        from_img = True
    elif isinstance(source, jt.Var):
        tensor = True
    else:
        raise TypeError("Unsupported image type. For supported types see https://docs.jittoryolo.com/modes/predict")

    return source, webcam, screenshot, from_img, in_memory, tensor


def load_inference_source(source=None, batch=1, vid_stride=1, buffer=False):
    """
    Loads an inference source for object detection and applies necessary transformations.

    Args:
        source (str, Path, Tensor, PIL.Image, np.ndarray): The input source for inference.
        batch (int, optional): Batch size for dataloaders. Default is 1.
        vid_stride (int, optional): The frame interval for video sources. Default is 1.
        buffer (bool, optional): Determined whether stream frames will be buffered. Default is False.

    Returns:
        dataset (Dataset): A dataset object for the specified input source.
    """
    source, stream, screenshot, from_img, in_memory, tensor = check_source(source)
    source_type = source.source_type if in_memory else SourceTypes(stream, screenshot, from_img, tensor)

    # Dataloader
    if tensor:
        dataset = LoadTensor(source)
    elif in_memory:
        dataset = source
    elif stream:
        dataset = LoadStreams(source, vid_stride=vid_stride, buffer=buffer)
    elif screenshot:
        dataset = LoadScreenshots(source)
    elif from_img:
        dataset = LoadPilAndNumpy(source)
    else:
        dataset = LoadImagesAndVideos(source, batch=batch, vid_stride=vid_stride)

    # Attach source types to the dataset
    setattr(dataset, "source_type", source_type)

    return dataset
