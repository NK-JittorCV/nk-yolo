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
    
    def __init__(
        self,
        dataset,
        batch_size=1,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        buffer_size=512,
        dataset_size=None,
    ):
        """Initialize infinite dataset wrapper.
        
        Args:
            dataset: Original dataset to wrap.
            batch_size (int): Batch size for data loading.
            shuffle (bool): Whether to shuffle data.
            drop_last (bool): Whether to drop last incomplete batch.
            num_workers (int): Number of worker threads.
            buffer_size (int): Buffer size for Jittor RingBuffer.
        """
        super().__init__()
        self.dataset = dataset
        
        # Keep a stable dataset length across iterator resets.
        # `len(dataset)` may change after Jittor iteration in MPI mode, while `dataset.ni`
        # stores the original sample count for YOLODataset.
        if dataset_size is not None:
            dataset_len = int(dataset_size)
        else:
            dataset_len = int(getattr(dataset, "ni", len(dataset)))
        
        # Set dataset attributes with correct values from the start
        self.set_attrs(
            total_len=dataset_len,  # Global length; Jittor will shard per rank in MPI mode
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=num_workers,
            buffer_size=buffer_size
        )
        
        # Forward all attributes from the original dataset that we don't explicitly define
        self.__dict__.update({k: v for k, v in dataset.__dict__.items() if k not in self.__dict__})
        
        # Set collate function directly
        self.collate_fn = dataset.collate_fn

    def __getitem__(self, index):
        """Get item at index, cycling through dataset if index exceeds length."""
        return self.dataset[index % len(self.dataset)]

    def close_mosaic(self, hyp):
        """Forward mosaic-closing to the wrapped dataset if supported."""
        result = self.dataset.close_mosaic(hyp)
        # Keep wrapper in sync with any updated transforms.
        self.transforms = self.dataset.transforms
        return result

    def collate_batch(self, batch):
        """Collate batch using custom collate function."""
        return self.collate_fn(batch)


def collate_fn(batch):
    """Collates data samples into batches.
    
    Args:
        batch (list): List of sample dictionaries from the dataset.
        
    Returns:
        dict: Dictionary with batched data, or None if batch is empty.
    """
    n = len(batch)
    if n == 0:
        return None
    
    out = {}
    keys = batch[0].keys()
    
    def _same_shape(vals):
        first = vals[0]
        if not isinstance(first, (np.ndarray, jt.Var)):
            return False
        shape = first.shape
        for v in vals:
            if not isinstance(v, (np.ndarray, jt.Var)):
                return False
            if v.shape != shape:
                return False
        return True

    for k in keys:
        values = [item[k] for item in batch]
        
        if k == "img":
            # Images should always be stacked (HWC or CHW numpy arrays)
            out[k] = np.stack(values, 0)
            
        elif k in ["masks", "keypoints", "bboxes", "cls", "segments"]:
            # These items may have different sizes per sample
            out[k] = np.stack(values, 0) if _same_shape(values) else values
                
        elif k == "batch_idx":
            # Special handling for batch indices
            if isinstance(values[0], (list, tuple, np.ndarray)):
                out[k] = []
                for i, v in enumerate(values):
                    out[k].extend([i] * len(v))
                out[k] = np.array(out[k], dtype=np.int32)
            else:
                out[k] = np.array(values, dtype=np.int32)
                
        elif isinstance(values[0], (int, float, np.integer, np.floating)):
            # Basic numeric types
            out[k] = np.array(values)
            
        else:
            # Keep other types as lists
            out[k] = values
            
    return out

class InfiniteDataLoader:
    """DataLoader wrapper for infinite iteration over a dataset."""
    
    def __init__(self, dataset, batch_size, shuffle=False, num_workers=0,
                 pin_memory=False, worker_init_fn=None, drop_last=False,
                 buffer_size=512, collate_fn=None, rank=-1):
        """Initialize InfiniteDataLoader.
        
        Args:
            dataset: The dataset to wrap.
            batch_size (int): Number of samples per batch.
            shuffle (bool): Whether to shuffle the dataset.
            num_workers (int): Number of worker threads.
            pin_memory (bool): Whether to pin memory.
            worker_init_fn (callable, optional): Function to initialize worker threads.
            drop_last (bool): Whether to drop last incomplete batch.
            buffer_size (int): Buffer size for Jittor RingBuffer.
            collate_fn (callable, optional): Function to collate batches.
        """
        from nkyolo.utils import RANK
        
        self.original_dataset = dataset
        self.dataset = dataset
        self.dataset_size = int(getattr(dataset, "ni", len(dataset)))
        
        # Use dataset's collate_fn if not explicitly provided
        if collate_fn is None:
            collate_fn = dataset.collate_fn
        
        # Store configuration
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.worker_init_fn = worker_init_fn
        self.collate_fn = collate_fn
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.rank = int(rank)
        # Store buffer_size for use in __iter__
        self.buffer_size = buffer_size
        # Track worker init context to avoid unnecessary worker respawn.
        self._worker_mpi_state = None
        self._worker_world_size = None
        
        # Calculate number of batches
        # In MPI, Jittor shards samples by rank at iteration time.
        # `len(dataset)` remains global, so adjust __len__ to local batches,
        # otherwise the trainer will over-iterate and repeatedly reset loader.
        dataset_size = self.dataset_size
        if self.rank >= 0 and jt.in_mpi and int(jt.world_size) > 1:
            world = int(jt.world_size)
            rank = int(jt.rank)
            local_size = dataset_size // world
            if rank < (dataset_size % world):
                local_size += 1
        else:
            local_size = dataset_size

        self.num_batches = local_size // batch_size
        if not drop_last and local_size % batch_size != 0:
            self.num_batches += 1
        self.num_batches = max(1, int(self.num_batches))
        
        # Debug logging for distributed training
        if RANK >= 0:
            from nkyolo.utils import LOGGER
            LOGGER.info(
                f"InfiniteDataLoader rank {RANK}: dataset_size={dataset_size}, local_size={local_size}, "
                f"batch_size={batch_size}, num_batches={self.num_batches}"
            )

    def __len__(self):
        """Return length of dataset."""
        return self.num_batches

    def __iter__(self):
        """Return self as iterator."""
        # Jittor RingBuffer size is in bytes. Use configured buffer_size or 512MB minimum.
        final_buffer_size = max(int(self.buffer_size), 512 * 1024 * 1024)
        cur_mpi_state = bool(jt.in_mpi)
        cur_world_size = int(jt.world_size) if cur_mpi_state else 1

        # Build wrapped dataset once and keep it stable across resets.
        if not isinstance(self.dataset, InfiniteDataset):
            self.dataset = InfiniteDataset(
                self.original_dataset,
                batch_size=self.batch_size,
                shuffle=self.shuffle,
                drop_last=self.drop_last,
                num_workers=self.num_workers,
                buffer_size=final_buffer_size,
                dataset_size=self.dataset_size,
            )
        elif (
            hasattr(self.dataset, "workers")
            and self._worker_mpi_state is not None
            and (
                self._worker_mpi_state != cur_mpi_state
                or self._worker_world_size != cur_world_size
            )
        ):
            # MPI context can change under jt.single_process_scope (used by rank0-only validation).
            # Recreate workers only when MPI mode/world size changes, otherwise keep workers alive.
            self.dataset.reset()
        # NOTE: do not call dataset.set_attrs() on every __iter__.
        # Jittor's Dataset.set_attrs() triggers reset(), which tears down/re-spawns workers.
        # Repeated re-spawn across epochs can leak worker processes and make later epochs slower.
        
        if self.collate_fn:
            self.dataset.collate_fn = self.collate_fn
        
        self.iterator = self.dataset.__iter__()
        if self.num_workers > 0:
            self._worker_mpi_state = cur_mpi_state
            self._worker_world_size = cur_world_size
        return self

    def __next__(self):
        """Get next batch."""
        batch = next(self.iterator)
        return batch

    def reset(self):
        """Reset iterator."""
        if isinstance(self.dataset, InfiniteDataset):
            self.iterator = self.dataset.__iter__()
        else:
            self.__iter__()


def seed_worker(worker_id):  # noqa
    """Set random seed for dataloader worker.
    
    Args:
        worker_id (int): The worker process/thread ID.
    """
    # Data loading should run on CPU to avoid CUDA initialization in worker processes.
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    jt.flags.use_cuda = 0
    # Use Python's random and numpy instead of Jittor-specific RNG
    seed = int(random.random() * 2**32)  # Generate random seed
    np.random.seed(seed + worker_id)
    random.seed(seed + worker_id)


def build_yolo_dataset(cfg, img_path, batch, data, mode="train", rect=False, stride=32, multi_modal=False):
    """Build YOLO Dataset.
    
    Args:
        cfg: Configuration object with dataset and augmentation parameters.
        img_path (str): Path to directory containing images.
        batch (int): Batch size for the dataset.
        data (dict): Data dictionary containing dataset configuration.
        mode (str): Dataset mode, either "train" or "val".
        rect (bool): Whether to use rectangular training.
        stride (int): Model stride for ensuring image dimensions are divisible.
        multi_modal (bool): Whether to use multi-modal data.
    
    Returns:
        YOLODataset: Configured YOLO dataset instance.
    """
    if isinstance(stride, jt.Var):
        stride = int(stride.max().item())
    elif isinstance(stride, (list, tuple, np.ndarray)):
        stride = int(np.max(stride))
    else:
        stride = int(stride)

    # Jittor MPI automatically shards datasets per rank. Avoid manual splitting in MPI mode.
    split_by_rank = (mode == "train") and (not jt.in_mpi)

    return YOLODataset(
        img_path=img_path,
        imgsz=cfg.imgsz,
        batch_size=batch,
        augment=mode == "train",  # augmentation
        hyp=cfg,  # TODO: probably add a get_hyps_from_cfg function
        rect=cfg.rect or rect,  # rectangular batches
        cache=cfg.cache or None,
        single_cls=cfg.single_cls or False,
        stride=stride,
        pad=0.0 if mode == "train" else 0.5,
        prefix=colorstr(f"{mode}: "),
        task=cfg.task,
        classes=cfg.classes,
        data=data,
        fraction=cfg.fraction if mode == "train" else 1.0,
        split_by_rank=split_by_rank,
    )


def build_dataloader(dataset, batch, workers, shuffle=True, rank=-1, buffer_size=None):
    """Return an InfiniteDataLoader for training or validation set.
    
    Args:
        dataset: The dataset to load data from.
        batch (int): Batch size for data loading.
        workers (int): Number of worker threads.
        shuffle (bool): Whether to shuffle dataset. Only applied if rank == -1.
        rank (int): Process rank for distributed training. -1 for single process.
        buffer_size (int, optional): Buffer size for Jittor RingBuffer. If None, uses large fixed value.
    
    Returns:
        InfiniteDataLoader: Configured data loader.
    """
    from nkyolo.utils import RANK, LOGGER
    
    batch = min(batch, len(dataset))
    workers = min(os.cpu_count() or 1, workers)
    effective_rank = RANK if RANK >= 0 else rank
    if effective_rank >= 0 and workers > 0:
        # In Jittor MPI, worker processes are forked from rank processes and can consume large host RAM.
        # Keep a conservative default cap to avoid host OOM-kill (signal 9) in multi-GPU runs.
        if jt.in_mpi and int(jt.world_size) > 1:
            max_workers = int(os.getenv("NKYOLO_DDP_MAX_WORKERS", "2") or 2)
            max_workers = max(0, max_workers)
            if workers > max_workers:
                LOGGER.warning(
                    f"WARNING ⚠️ DDP detected, capping dataloader workers from {workers} to {max_workers} "
                    "(set NKYOLO_DDP_MAX_WORKERS to override)."
                )
                workers = max_workers

        # Escape hatch: explicitly disable workers for debugging extreme instability.
        disable_ddp_workers = str(os.getenv("NKYOLO_DDP_WORKERS", "1")).lower() in {"0", "false", "no"}
        if disable_ddp_workers:
            LOGGER.warning("WARNING ⚠️ DDP detected, forcing dataloader workers=0 (NKYOLO_DDP_WORKERS=0).")
            workers = 0
    
    if buffer_size is None:
        # Jittor RingBuffer size is in bytes. Default to 512MB to avoid worker overflow.
        buffer_size = 512 * 1024 * 1024
    
    # Keep the natural tail batch. Forcing drop_last under MPI can produce unstable final steps.
    drop_last = False

    loader = InfiniteDataLoader(
        dataset=dataset,
        batch_size=batch,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=PIN_MEMORY,
        worker_init_fn=seed_worker,
        drop_last=drop_last,
        buffer_size=buffer_size,
        rank=rank,
    )
    
    return loader


def check_source(source):
    """Check source type and return corresponding flag values.
    
    Args:
        source: Input source (str, Path, int, LOADERS, list, PIL.Image, np.ndarray, or jt.Var).
    
    Returns:
        tuple: (source, webcam, screenshot, from_img, in_memory, tensor) flags.
    
    Raises:
        TypeError: If source type is not supported.
    """
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
    """Loads an inference source for object detection and applies necessary transformations.
    
    Args:
        source: Input source (str, Path, jt.Var, PIL.Image, np.ndarray, LOADERS, optional).
        batch (int): Batch size for dataloaders.
        vid_stride (int): Frame interval for video sources.
        buffer (bool): Whether stream frames will be buffered.
    
    Returns:
        Dataset: Dataset object for the specified input source.
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
