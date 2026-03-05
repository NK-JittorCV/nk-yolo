# NK-YOLO 🚀, AGPL-3.0 license
# Refer to https://github.com/ultralytics/ultralytics/blob/main/ultralytics/data/dataset.py

from pathlib import Path
from itertools import repeat
from multiprocessing.pool import ThreadPool
import time

import numpy as np
import jittor as jt

from nkyolo.utils import LOCAL_RANK, NUM_THREADS, TQDM, remove_colorstr
from nkyolo.utils.ops import resample_segments

from .augment import (
    Compose,
    Format,
    Instances,
    LetterBox,
    v8_transforms,
)
from .base import BaseDataset
from .utils import (
    HELP_URL,
    LOGGER,
    get_hash,
    img2label_paths,
    load_dataset_cache_file,
    save_dataset_cache_file,
    verify_image_label,
)

# NK-YOLO dataset *.jittor_cache version, >= 0.0.1 for YOLOv8
DATASET_CACHE_VERSION = "0.0.1"

class YOLODataset(BaseDataset):
    """
    Dataset class for loading object detection and/or segmentation labels in YOLO format.

    Args:
        data (dict, optional): A dataset YAML dictionary. Defaults to None.
        task (str): An explicit arg to point current task, Defaults to 'detect'.

    Returns:
        (torch.utils.data.Dataset): A PyTorch dataset object that can be used for training an object detection model.
    """

    def __init__(self, *args, data=None, task="detect", **kwargs):
        """Initializes the YOLODataset with optional configurations for segments and keypoints."""
        self.use_segments = task == "segment"
        self.use_keypoints = task == "pose"
        self.use_obb = task == "obb"
        self.data = data
        assert not (self.use_segments and self.use_keypoints), "Can not use both segments and keypoints."
        super().__init__(*args, **kwargs)

    def _show_cache_scan_logs(self):
        """Return True when cache scanning progress should be shown."""
        prefix = remove_colorstr(str(getattr(self, "prefix", ""))).strip().lower()
        return not prefix.startswith("val:")

    def cache_labels(self, path=Path("./labels.jittor_cache")):
        """
        Cache dataset labels, check images and read shapes.

        Args:
            path (Path): Path where to save the cache file. Default is Path('./labels.jittor_cache').

        Returns:
            (dict): labels.
        """
        x = {"labels": []}
        nm, nf, ne, nc, msgs = 0, 0, 0, 0, []  # number missing, found, empty, corrupt, messages
        desc = f"{self.prefix}Scanning {path.parent / path.stem}..."
        total = len(self.im_files)
        nkpt, ndim = self.data.get("kpt_shape", (0, 0))
        if self.use_keypoints and (nkpt <= 0 or ndim not in {2, 3}):
            raise ValueError(
                "'kpt_shape' in data.yaml missing or incorrect. Should be a list with [number of "
                "keypoints, number of dims (2 for x,y or 3 for x,y,visible)], i.e. 'kpt_shape: [17, 3]'"
            )
        show_scan = self._show_cache_scan_logs()
        with ThreadPool(NUM_THREADS) as pool:
            results = pool.imap(
                func=verify_image_label,
                iterable=zip(
                    self.im_files,
                    self.label_files,
                    repeat(self.prefix),
                    repeat(self.use_keypoints),
                    repeat(len(self.data["names"])),
                    repeat(nkpt),
                    repeat(ndim),
                ),
            )
            pbar = TQDM(results, desc=desc, total=total, disable=not show_scan)
            for im_file, lb, shape, segments, keypoint, nm_f, nf_f, ne_f, nc_f, msg in pbar:
                nm += nm_f
                nf += nf_f
                ne += ne_f
                nc += nc_f
                if im_file:
                    x["labels"].append(
                        {
                            "im_file": im_file,
                            "shape": shape,
                            "cls": lb[:, 0:1],  # n, 1
                            "bboxes": lb[:, 1:],  # n, 4
                            "segments": segments,
                            "keypoints": keypoint,
                            "normalized": True,
                            "bbox_format": "xywh",
                        }
                    )
                if msg:
                    msgs.append(msg)
                if show_scan:
                    pbar.desc = f"{desc} {nf} images, {nm + ne} backgrounds, {nc} corrupt"
            pbar.close()

        if msgs:
            LOGGER.info("\n".join(msgs))
        if nf == 0:
            LOGGER.warning(f"{self.prefix}WARNING ⚠️ No labels found in {path}. {HELP_URL}")
        x["hash"] = get_hash(self.label_files + self.im_files)
        x["results"] = nf, nm, ne, nc, len(self.im_files)
        x["msgs"] = msgs  # warnings
        save_dataset_cache_file(self.prefix, path, x, DATASET_CACHE_VERSION)
        return x

    def get_labels(self):
        """Returns dictionary of labels for YOLO training."""
        self.label_files = img2label_paths(self.im_files)
        # Use source-file cache when dataset is provided as a txt list (e.g. coco128_train.txt).
        # This avoids collisions with directory-based datasets sharing the same labels folder.
        label_dir_cache = Path(self.label_files[0]).parent.with_suffix(".jittor_cache")
        cache_path = label_dir_cache
        src = self.img_path[0] if isinstance(self.img_path, (list, tuple)) and len(self.img_path) == 1 else self.img_path
        if isinstance(src, (str, Path)):
            src_path = Path(src)
            if src_path.is_file():
                cache_path = src_path.with_suffix(".jittor_cache")
        cache = None

        def _cache_ok(c, expected_hash):
            """Return True when cache dict matches current dataset content."""
            if not isinstance(c, dict):
                return False
            if c.get("version") != DATASET_CACHE_VERSION:
                return False
            if c.get("hash") != expected_hash:
                return False
            if not isinstance(c.get("labels"), list):
                return False
            results = c.get("results")
            return isinstance(results, (tuple, list)) and len(results) == 5

        def _try_load_cache(path, log_warning=True):
            """Best-effort cache loader that falls back to label scan on read errors."""
            try:
                return load_dataset_cache_file(path)
            except Exception as e:
                if log_warning and LOCAL_RANK in {-1, 0}:
                    LOGGER.warning(
                        f"{self.prefix}Unable to load dataset cache {path}: {type(e).__name__}: {e}. "
                        "Falling back to label scan."
                    )
                return None

        current_hash = get_hash(self.label_files + self.im_files)
        cache_file_found = any(
            p.exists()
            for p in (
                cache_path,
                cache_path.with_suffix(".cache"),
                cache_path.with_suffix(".npy"),
                cache_path.with_suffix(".cache.npy"),
            )
        )
        exists = False
        if cache_file_found:
            cache = _try_load_cache(cache_path)
            exists = _cache_ok(cache, current_hash)
            if cache is not None and not exists and LOCAL_RANK in {-1, 0}:
                LOGGER.warning(f"{self.prefix}Dataset cache is stale or invalid, rebuilding: {cache_path}")

        use_mpi = bool(jt.in_mpi)
        mpi_world = int(jt.world_size) if use_mpi else 1
        mpi_rank = int(jt.rank) if use_mpi else -1

        if not exists:
            if use_mpi and mpi_world > 1:
                if mpi_rank == 0:
                    cache = self.cache_labels(cache_path)
                else:
                    max_wait = 300.0
                    start = time.time()
                    wait_reason = f"timed out after {max_wait:.0f}s"
                    while time.time() - start < max_wait:
                        if not cache_path.exists():
                            time.sleep(0.05)
                            continue
                        cache = _try_load_cache(cache_path, log_warning=False)
                        if cache is None:
                            wait_reason = "failed because cache file is unreadable"
                            break
                        latest_hash = get_hash(self.label_files + self.im_files)
                        if _cache_ok(cache, latest_hash):
                            break
                        time.sleep(0.05)
                    if cache is None or (not _cache_ok(cache, get_hash(self.label_files + self.im_files))):
                        LOGGER.warning(f"{self.prefix}Rank {mpi_rank}: Cache wait {wait_reason}, rebuilding locally.")
                        cache = self.cache_labels(cache_path)
            else:
                cache = self.cache_labels(cache_path)

        # Re-check with a refreshed hash because verify pass may repair image files and change hash.
        latest_hash = get_hash(self.label_files + self.im_files)
        if not _cache_ok(cache, latest_hash):
            # Retry once in case another rank replaced the cache file between checks.
            if cache_path.exists():
                refreshed_cache = _try_load_cache(cache_path, log_warning=False)
                if refreshed_cache is not None:
                    cache = refreshed_cache
            latest_hash = get_hash(self.label_files + self.im_files)

        if not _cache_ok(cache, latest_hash):
            LOGGER.warning(f"{self.prefix}Cache mismatch after rebuild, forcing local rebuild: {cache_path}")
            cache = self.cache_labels(cache_path)
            latest_hash = get_hash(self.label_files + self.im_files)
            if not _cache_ok(cache, latest_hash):
                raise RuntimeError(f"{self.prefix}Dataset cache check failed after rebuild: {cache_path}")

        # Display cache
        nf, nm, ne, nc, n = cache.pop("results")  # found, missing, empty, corrupt, total
        if exists and LOCAL_RANK in {-1, 0} and self._show_cache_scan_logs():
            d = f"Scanning {cache_path}... {nf} images, {nm + ne} backgrounds, {nc} corrupt"
            TQDM(None, desc=self.prefix + d, total=n, initial=n)  # display results
            if cache["msgs"]:
                LOGGER.info("\n".join(cache["msgs"]))  # display warnings

        # Read cache
        [cache.pop(k) for k in ("hash", "version", "msgs")]  # remove items
        labels = cache["labels"]
        if not labels:
            LOGGER.warning(f"WARNING ⚠️ No images found in {cache_path}, training may not work correctly. {HELP_URL}")
        self.im_files = [lb["im_file"] for lb in labels]  # update im_files

        # Check if the dataset is all boxes or all segments
        lengths = ((len(lb["cls"]), len(lb["bboxes"]), len(lb["segments"])) for lb in labels)
        len_cls, len_boxes, len_segments = (sum(x) for x in zip(*lengths))
        if len_segments and len_boxes != len_segments:
            LOGGER.warning(
                f"WARNING ⚠️ Box and segment counts should be equal, but got len(segments) = {len_segments}, "
                f"len(boxes) = {len_boxes}. To resolve this only boxes will be used and all segments will be removed. "
                "To avoid this please supply either a detect or segment dataset, not a detect-segment mixed dataset."
            )
            for lb in labels:
                lb["segments"] = []
        if len_cls == 0:
            LOGGER.warning(f"WARNING ⚠️ No labels found in {cache_path}, training may not work correctly. {HELP_URL}")
        return labels

    def build_transforms(self, hyp=None):
        """Builds and appends transforms to the list."""
        if self.augment:
            hyp.mosaic = hyp.mosaic if self.augment and not self.rect else 0.0
            hyp.mixup = hyp.mixup if self.augment and not self.rect else 0.0
            transforms = v8_transforms(self, self.imgsz, hyp)
        else:
            transforms = Compose([LetterBox(new_shape=(self.imgsz, self.imgsz), scaleup=False)])
        transforms.append(
            Format(
                bbox_format="xywh",
                normalize=True,
                return_mask=self.use_segments,
                return_keypoint=self.use_keypoints,
                return_obb=self.use_obb,
                batch_idx=True,
                mask_ratio=hyp.mask_ratio,
                mask_overlap=hyp.overlap_mask,
                bgr=hyp.bgr if self.augment else 0.0,  # only affect training.
            )
        )
        return transforms

    def close_mosaic(self, hyp):
        """Sets mosaic, copy_paste and mixup options to 0.0 and builds transformations."""
        hyp.mosaic = 0.0  # set mosaic ratio=0.0
        hyp.copy_paste = 0.0  # keep the same behavior as previous v8 close-mosaic
        hyp.mixup = 0.0  # keep the same behavior as previous v8 close-mosaic
        self.transforms = self.build_transforms(hyp)

    def update_labels_info(self, label):
        """
        Custom your label format here.

        Note:
            cls is not with bboxes now, classification and semantic segmentation need an independent cls label
            Can also support classification and semantic segmentation by adding or removing dict keys there.
        """
        bboxes = label.pop("bboxes")
        segments = label.pop("segments", [])
        keypoints = label.pop("keypoints", None)
        bbox_format = label.pop("bbox_format")
        normalized = label.pop("normalized")

        # NOTE: do NOT resample oriented boxes
        segment_resamples = 100 if self.use_obb else 1000
        if len(segments) > 0:
            # list[np.array(1000, 2)] * num_samples
            # (N, 1000, 2)
            segments = np.stack(resample_segments(segments, n=segment_resamples), axis=0)
        else:
            segments = np.zeros((0, segment_resamples, 2), dtype=np.float32)
        label["instances"] = Instances(bboxes, segments, keypoints, bbox_format=bbox_format, normalized=normalized)
        return label

    @staticmethod
    def collate_fn(batch):
        """Collates data samples into batches."""
        if not batch:
            return None
        
        new_batch = {}
        for k in batch[0].keys():
            values = [item[k] for item in batch]
            
            if k == "img":
                # Images can be stacked directly because they are already preprocessed to the same size.
                new_batch[k] = np.stack(values, 0)
            elif k == "cls":
                # Class indices have variable lengths, so concatenate into a 1D array.
                new_batch[k] = np.concatenate(values, 0) if len(values) else np.zeros((0, 1), dtype=np.int32)
            elif k == "bboxes":
                # Bounding boxes have variable lengths, so concatenate into a 1D array.
                new_batch[k] = np.concatenate(values, 0) if len(values) else np.zeros((0, 4), dtype=np.float32)
            elif k == "batch_idx":
                new_batch["batch_idx"] = [item["batch_idx"] for item in batch]
                for i in range(len(new_batch["batch_idx"])):
                    new_batch["batch_idx"][i] = new_batch["batch_idx"][i] + i  # add target image index for build_targets()
                new_batch["batch_idx"] = np.concatenate(new_batch["batch_idx"], 0).astype(np.int32)
            else:
                # Keep other fields as-is.
                new_batch[k] = values
        
        return new_batch
