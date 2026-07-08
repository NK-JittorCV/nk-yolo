# NK-YOLO 🚀 AGPL-3.0 License
# Refer to https://github.com/ultralytics/ultralytics/blob/main/ultralytics/engine/trainer.py
"""
Train a model on a dataset.

Usage:
    $ yolo mode=train model=yolov8n.pt data=coco8.yaml imgsz=640 epochs=100 batch=16
"""

import gc
import json
import math
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import warnings
import importlib.util
from copy import copy
from datetime import datetime
from pathlib import Path

import numpy as np
import jittor as jt
from jittor import nn, optim
from nkyolo.cfg import get_cfg, get_save_dir
from nkyolo.data.utils import check_cls_dataset, check_det_dataset
from nkyolo.nn.tasks import attempt_load_one_weight, attempt_load_weights
from nkyolo.utils import (
    DEFAULT_CFG,
    LOCAL_RANK,
    LOGGER,
    RANK,
    TQDM,
    __version__,
    callbacks,
    clean_url,
    colorstr,
    emojis,
    get_world_size,
    yaml_save,
)
from nkyolo.utils.autobatch import check_train_batch_size
from nkyolo.utils.checks import check_file, check_imgsz, check_model_file_from_stem, print_args
from nkyolo.utils.dist import (
    ddp_cleanup,
    generate_ddp_command,
    get_visible_devices,
    parse_device_list,
    resolve_device_id,
)
from nkyolo.utils.files import get_latest_run
from nkyolo.utils.jittor_utils import (
    EarlyStopping,
    LambdaLR,
    ModelEMA,
    autocast,
    convert_optimizer_state_dict_to_fp16,
    init_seeds,
    one_cycle,
    select_device,
    strip_optimizer,
    safe_deepcopy_jittor,
    state_dict_to_jittor,
)

_JT_DISTRIBUTED_AVAILABLE = importlib.util.find_spec("jittor.distributed") is not None

class BaseTrainer:
    """
    A base class for creating trainers.

    Attributes:
        args (SimpleNamespace): Configuration for the trainer.
        validator (BaseValidator): Validator instance.
        model (nn.Module): Model instance.
        callbacks (defaultdict): Dictionary of callbacks.
        save_dir (Path): Directory to save results.
        wdir (Path): Directory to save weights.
        last (Path): Path to the last checkpoint.
        best (Path): Path to the best checkpoint.
        save_period (int): Save checkpoint every x epochs (disabled if < 1).
        batch_size (int): Batch size for training.
        epochs (int): Number of epochs to train for.
        start_epoch (int): Starting epoch for training.
        device (jt.device): Device to use for training.
        amp (bool): Flag to enable AMP (Automatic Mixed Precision).
        scaler (amp.GradScaler): Gradient scaler for AMP.
        data (str): Path to data.
        trainset (jt.utils.data.Dataset): Training dataset.
        testset (jt.utils.data.Dataset): Testing dataset.
        ema (nn.Module): EMA (Exponential Moving Average) of the model.
        resume (bool): Resume training from a checkpoint.
        lf (nn.Module): Loss function.
        scheduler (LambdaLR): Learning rate scheduler.
        best_fitness (float): The best fitness value achieved.
        fitness (float): Current fitness value.
        loss (float): Current loss value.
        tloss (float): Total loss value.
        loss_names (list): List of loss names.
        csv (Path): Path to results CSV file.
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """
        Initialize the BaseTrainer class.

        Args:
            cfg (str, optional): Path to a configuration file.
            overrides (dict, optional): Configuration overrides.
            _callbacks (list, optional): List of callback functions.
        """
        # HUB session handling (placeholder for future implementation)
        self.hub_session = overrides.pop("session", None) if overrides else None
        
        self.args = get_cfg(cfg, overrides)
        self.check_resume(overrides)
        self.device = select_device(self.args.device, self.args.batch)
        self.model_base = None

        # Update device string for consistent logging
        if "cuda" in str(self.device):
            self.args.device = os.getenv("CUDA_VISIBLE_DEVICES", str(self.device))
        else:
            self.args.device = str(self.device)
            
        self.validator = None
        self.metrics = None
        self.plots = {}
        self.freeze_layer_names = []
        init_seeds(self.args.seed, deterministic=self.args.deterministic)

        # Dirs
        self.save_dir = get_save_dir(self.args)
        self.args.name = self.save_dir.name  # update name for loggers
        self.wdir = self.save_dir / "weights"  # weights dir
        if RANK in {-1, 0}:
            self.wdir.mkdir(parents=True, exist_ok=True)  # make dir
            self.args.save_dir = str(self.save_dir)
            yaml_save(self.save_dir / "args.yaml", vars(self.args))  # save run args
        self.last, self.best = self.wdir / "last.pkl", self.wdir / "best.pkl"  # checkpoint paths
        self.save_period = self.args.save_period

        self.batch_size = self.args.batch
        self.val_batch_size = None
        self.epochs = self.args.epochs
        self.start_epoch = 0
        self.train_loader = None
        self.test_loader = None
        if RANK == -1:
            print_args(vars(self.args))

        # Device
        if self.device in {"cpu", "mps"}:
            self.args.workers = 0  # faster CPU training as time dominated by inference, not dataloading

        # Model and Dataset
        self.model = check_model_file_from_stem(self.args.model)  # add suffix, i.e. yolov8n -> yolov8n.pt

        self.trainset, self.testset = self.get_dataset()
        self.ema = None
        self.ema_steps = 0
        
        # Optimization utils init
        self.lf = None
        self.scheduler = None

        # Epoch level metrics
        self.best_fitness = None
        self.fitness = None
        self.loss = None
        self.tloss = None
        self.loss_names = ["Loss"]
        self.csv = self.save_dir / "results.csv"
        self.plot_idx = [0, 1, 2]

        # HUB
        self.hub_session = None

        # Callbacks
        self.callbacks = _callbacks or callbacks.get_default_callbacks()
        if RANK in {-1, 0}:
            callbacks.add_integration_callbacks(self)

    def add_callback(self, event: str, callback):
        """Appends the given callback."""
        self.callbacks[event].append(callback)

    def set_callback(self, event: str, callback):
        """Overrides the existing callbacks with the given callback."""
        self.callbacks[event] = [callback]

    def run_callbacks(self, event: str):
        """Run all existing callbacks associated with a particular event."""
        for callback in self.callbacks.get(event, []):
            callback(self)

    def train(self):
        """Allow device='', device=None on Multi-GPU systems to default to device=0."""
        # Check if we're already in an MPI environment (to avoid recursive mpirun calls)
        is_mpi_env = bool(jt.in_mpi)
        
        device_list = parse_device_list(self.args.device)
        if device_list:
            world_size = len(device_list)
        elif self.args.device in {"cpu", "mps"}:  # i.e. device='cpu' or 'mps'
            world_size = 0
        elif jt.has_cuda:  # i.e. device=None or device='' or device=number
            world_size = 1  # default to device 0
        else:  # i.e. device=None or device=''
            world_size = 0

        # If we're in MPI environment, get actual world size from Jittor
        if is_mpi_env:
            world_size = int(jt.world_size)

        # If not running under MPI, auto-launch MPI for multi-GPU DDP.
        if world_size > 1 and not is_mpi_env:
            LOGGER.info("Multi-GPU requested without MPI. Launching mpirun for DDP...")
            cmd = None
            file = None
            try:
                cmd, file, env = generate_ddp_command(world_size, self)
                LOGGER.info(f"DDP command: {' '.join(cmd)}")
                result = subprocess.run(cmd, env=env)
            except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as exc:
                LOGGER.warning(
                    "WARNING ⚠️ Failed to launch MPI DDP, falling back to single GPU. "
                    f"Reason: {exc}"
                )
                world_size = 1
                if device_list:
                    # Force single-device selection to avoid multi-GPU setup.
                    self.args.device = str(device_list[0])
                    self.device = select_device(self.args.device, self.args.batch, verbose=False)
            else:
                if result.returncode != 0:
                    raise RuntimeError(f"DDP subprocess failed with return code {result.returncode}")
                return
            finally:
                if file and os.path.exists(file):
                    ddp_cleanup(self, file)

        # Train in current process (MPI ranks handled externally)
        self._do_train(world_size)

    @staticmethod
    def _is_oom_error(exc: BaseException) -> bool:
        """Return True when exception message indicates CUDA/Jittor OOM."""
        msg = str(exc).lower()
        oom_markers = (
            "out of memory",
            "cuda out of memory",
            "cudaerrormemoryallocation",
            "memory allocation",
            "cudnn_status_alloc_failed",
            "cuda malloc",
            "cuda malloc failed",
        )
        return any(x in msg for x in oom_markers)

    def _abort_all_on_oom(self, exc: BaseException):
        """Terminate current process immediately on OOM so MPI can tear down all ranks."""
        rank_msg = f"rank {RANK}" if RANK >= 0 else "single process"
        LOGGER.error(f"CUDA OOM detected on {rank_msg}. Terminating all training processes immediately.")
        LOGGER.error(str(exc))
        self._clear_memory()
        os.kill(os.getpid(), signal.SIGKILL)

    @staticmethod
    def _get_ddp_barrier_dir() -> Path:
        """Return a cross-rank shared barrier directory for current MPI job."""
        job_id = (
            os.getenv("NKYOLO_DDP_BARRIER_ID")
            or os.getenv("PMIX_NAMESPACE")
            or os.getenv("OMPI_MCA_orte_ess_jobid")
            or f"ppid{os.getppid()}"
        )
        return Path(tempfile.gettempdir()) / "nkyolo_ddp_barrier" / str(job_id)

    def _sync_stop_flag(self, epoch: int, stop: bool, max_wait: float = 600.0) -> bool:
        """Share rank0 stop decision with other ranks."""
        if RANK == -1 or get_world_size() <= 1:
            return bool(stop)

        if jt.in_mpi and hasattr(jt, "mpi"):
            flag = np.array([1 if (bool(stop) and RANK == 0) else 0], dtype=np.int32)
            jt.mpi.broadcast(flag, 0)
            return bool(int(flag[0]))

        stop_dir = self._get_ddp_barrier_dir() / "stop_flags"
        stop_dir.mkdir(parents=True, exist_ok=True)
        flag_file = stop_dir / f"epoch_{epoch}.txt"
        if RANK == 0:
            flag_file.write_text("1" if stop else "0")
            return bool(stop)
        start = time.time()
        while time.time() - start < max_wait:
            if flag_file.exists():
                try:
                    return flag_file.read_text().strip() == "1"
                except OSError:
                    return bool(stop)
            time.sleep(0.05)
        LOGGER.warning(f"Rank {RANK}: stop flag sync timed out at epoch {epoch}, fallback to local stop={stop}.")
        return bool(stop)

    def _epoch_barrier(self, epoch: int, phase: str, max_wait: float = 1200.0):
        """Barrier to keep all MPI ranks aligned around rank0-only work (val/save)."""
        if RANK == -1 or get_world_size() <= 1:
            return

        if jt.in_mpi and hasattr(jt, "mpi"):
            jt.sync_all(True)
            jt.mpi.mpi_barrier()
            return

        world = get_world_size()
        bdir = self._get_ddp_barrier_dir() / "epoch_barrier" / f"epoch_{epoch}_{phase}"
        bdir.mkdir(parents=True, exist_ok=True)
        (bdir / f"rank_{RANK}.ok").write_text("1")
        release_file = bdir / "release.ok"

        start = time.time()
        if RANK == 0:
            while time.time() - start < max_wait:
                if sum(1 for _ in bdir.glob("rank_*.ok")) >= world:
                    release_file.write_text("1")
                    return
                time.sleep(0.05)
            LOGGER.warning(f"Rank 0: epoch barrier '{phase}' timed out at epoch {epoch}.")
            release_file.write_text("1")
            return

        while time.time() - start < max_wait:
            if release_file.exists():
                return
            time.sleep(0.05)
        LOGGER.warning(f"Rank {RANK}: epoch barrier '{phase}' timed out at epoch {epoch}.")
        return

    def _sync_train_num_batches(self, epoch: int, local_nb: int, max_wait: float = 600.0) -> int:
        """Synchronize per-rank train step count and return a common step count."""
        local_nb = max(1, int(local_nb))
        if RANK == -1 or get_world_size() <= 1:
            return local_nb

        if jt.in_mpi and hasattr(jt, "mpi"):
            v = np.array([local_nb], dtype=np.int32)
            jt.mpi.broadcast(v, 0)
            return max(1, int(v[0]))

        world = get_world_size()
        sdir = self._get_ddp_barrier_dir() / "train_num_batches" / f"epoch_{epoch}"
        sdir.mkdir(parents=True, exist_ok=True)
        (sdir / f"rank_{RANK}.txt").write_text(str(local_nb))
        resolved_file = sdir / "resolved.txt"

        if RANK == 0:
            start = time.time()
            while time.time() - start < max_wait:
                rank_files = list(sdir.glob("rank_*.txt"))
                if len(rank_files) >= world:
                    values = []
                    for rf in rank_files:
                        try:
                            values.append(int(rf.read_text().strip()))
                        except (OSError, ValueError):
                            continue
                    synced = max(1, min(values)) if values else local_nb
                    resolved_file.write_text(str(synced))
                    return synced
                time.sleep(0.05)
            LOGGER.warning(
                f"Rank {RANK}: train step sync timed out at epoch {epoch}, fallback to local_nb={local_nb}."
            )
            return local_nb

        start = time.time()
        while time.time() - start < max_wait:
            if resolved_file.exists():
                try:
                    return max(1, int(resolved_file.read_text().strip()))
                except (OSError, ValueError):
                    return local_nb
            time.sleep(0.05)
        LOGGER.warning(f"Rank {RANK}: wait train step sync timed out at epoch {epoch}, fallback to {local_nb}.")
        return local_nb

    def _get_configured_val_batch(self, base_batch: int) -> int:
        """Resolve effective validation batch size from args with sane fallback."""
        requested = int(getattr(self.args, "val_batch", -1) or -1)
        if requested > 0:
            return max(1, requested)
        # Keep validation batch aligned with train batch by default.
        return max(1, base_batch)

    def _reset_val_loader(self, val_batch: int):
        """Rebuild validation dataloader with a new batch size on rank0/single process."""
        if RANK not in {-1, 0}:
            return
        val_batch = max(1, int(val_batch))
        self.val_batch_size = val_batch
        self.test_loader = self.get_dataloader(self.testset, batch_size=val_batch, rank=-1, mode="val")
        if self.validator is not None:
            self.validator.dataloader = self.test_loader
            self.validator.args.batch = val_batch

    def _setup_scheduler(self):
        """Initialize training learning rate scheduler."""
        if self.args.cos_lr:
            self.lf = one_cycle(1, self.args.lrf, self.epochs)  # cosine 1->hyp['lrf']
        else:
            self.lf = lambda x: max(1 - x / self.epochs, 0) * (1.0 - self.args.lrf) + self.args.lrf  # linear
        self.scheduler = LambdaLR(self.optimizer, lr_lambda=self.lf)

    def _setup_ddp(self, world_size):
        """Initializes and sets the distributed training parameters for Jittor."""
        # Jittor automatically handles distributed setup via MPI.
        # CUDA_VISIBLE_DEVICES should list all GPUs for the job; LOCAL_RANK selects the device internally.
        if RANK >= 0:
            device_value = getattr(self.args, "device", "")
            device_arg = str(device_value).lower().strip()
            visible_devices = get_visible_devices()
            device_list = visible_devices or parse_device_list(device_value)
            device_id = resolve_device_id(device_list, LOCAL_RANK, RANK)

            force_cpu = device_arg in {"cpu", "mps"} or os.environ.get("CUDA_VISIBLE_DEVICES") == "-1"
            if jt.has_cuda and not force_cpu:
                jt.flags.use_cuda = 1
                # Ensure CUDA_VISIBLE_DEVICES lists all GPUs if provided.
                if device_list and (not os.environ.get("CUDA_VISIBLE_DEVICES") or os.environ.get("CUDA_VISIBLE_DEVICES") == "-1"):
                    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(x) for x in device_list)

                # Device string for logging and explicit tensor moves.
                self.device = f"cuda:{device_id}"
            else:
                os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
                jt.flags.use_cuda = 0
                self.device = "cpu"
            
            actual_world_size = get_world_size(default=world_size)

            LOGGER.info(f'DDP info: RANK {RANK}, LOCAL_RANK {LOCAL_RANK}, WORLD_SIZE {actual_world_size}, DEVICE {self.device}')
            
            # Jittor uses MPI for distributed training, no need for explicit init_process_group
            # The distributed context is automatically set up by MPI
            # Each process will automatically get a different subset of data

    def _setup_train(self, world_size):
        """Builds dataloaders and optimizer on correct rank process."""
        self.run_callbacks("on_pretrain_routine_start")
        ckpt = self.setup_model()
        self.set_model_attributes()
        self.model_base = self.model

        if jt.has_cuda and hasattr(jt, "cudnn") and hasattr(jt.cudnn, "set_algorithm_cache_size"):
            # Larger cache avoids repeated cuDNN algorithm searches that can stall training.
            jt.cudnn.set_algorithm_cache_size(10000)
        
        # Setup AMP (Automatic Mixed Precision)
        self._setup_amp(world_size)
        
        # Setup DDP (Distributed Data Parallel) if needed
        if world_size > 1:
            self._setup_model_ddp(world_size)

        # NOTE: Skip mpi_param_broadcast to avoid MPI_Bcast size mismatch errors.
        # We rely on consistent initialization (same Jittor seed across ranks) and
        # identical checkpoint loading on each rank to keep parameters in sync.

        # Freeze layers after potential DDP wrapping so grad flags are applied on the active model.
        self._freeze_layers()
        
        # Check imgsz
        gs = max(int(np.max(self.model.stride.numpy())), 32)
        self.args.imgsz = check_imgsz(self.args.imgsz, stride=gs, floor=gs, max_dim=1)
        self.stride = gs
        
        # Batch size
        if self.batch_size < 1 and RANK == -1:
            self.args.batch = self.batch_size = self.auto_batch()
        
        # Dataloaders
        # Use per-rank micro-batch in DDP so global effective batch equals args.batch.
        if world_size > 1:
            if self.batch_size < world_size:
                raise ValueError(
                    f"Batch size {self.batch_size} must be >= world_size {world_size} for DDP training."
                )
            if self.batch_size % world_size != 0 and RANK in {-1, 0}:
                LOGGER.warning(
                    f"WARNING ⚠️ batch={self.batch_size} is not divisible by world_size={world_size}. "
                    f"Using per-rank batch={self.batch_size // world_size}, effective global batch="
                    f"{(self.batch_size // world_size) * world_size}."
                )
            batch_size = max(1, self.batch_size // world_size)
        else:
            batch_size = self.batch_size
        self.train_loader = self.get_dataloader(self.trainset, batch_size=batch_size, rank=LOCAL_RANK, mode="train")
        if getattr(self.args, "ema", True) and RANK in {-1, 0}:
            self.ema = ModelEMA(self.model_base or self.model)
        if RANK in {-1, 0}:
            val_batch = self._get_configured_val_batch(batch_size)
            self._reset_val_loader(val_batch)
            self.validator = self.get_validator()
            metric_keys = self.validator.metrics.keys + self.label_loss_items(prefix="val")
            self.metrics = dict(zip(metric_keys, [0] * len(metric_keys)))
            if self.args.plots:
                self.plot_training_labels()
        
        # Optimizer
        self.accumulate = max(round(self.args.nbs / self.batch_size), 1)
        weight_decay = self.args.weight_decay * self.batch_size * self.accumulate / self.args.nbs
        iterations = math.ceil(len(self.train_loader.dataset) / max(self.batch_size, self.args.nbs)) * self.epochs
        self.optimizer = self.build_optimizer(
            model=self.model,
            name=self.args.optimizer,
            lr=self.args.lr0,
            momentum=self.args.momentum,
            decay=weight_decay,
            iterations=iterations,
        )
        
        # Scheduler
        self._setup_scheduler()
        self.stopper, self.stop = EarlyStopping(patience=self.args.patience), False
        self.resume_training(ckpt)
        self.scheduler.last_epoch = self.start_epoch - 1
        self._ddp_touch_params = None
        self.run_callbacks("on_pretrain_routine_end")

    def _jit_warmup(self, all_ranks=False):
        """Warm up Jittor JIT before training to reduce compile stalls."""
        if not all_ranks and RANK not in {-1, 0}:
            return
        if not self.train_loader:
            return
        warmup_batches = int(getattr(self.args, "jit_warmup_batches", 1) or 1)
        warmup_batches = max(1, min(warmup_batches, len(self.train_loader)))
        if RANK in {-1, 0}:
            LOGGER.info(f"JIT warmup: compiling {warmup_batches} train batch(es) before epoch loop (forward-only).")

        self._model_train()
        world = max(1, get_world_size())
        loader_iter = iter(self.train_loader)
        for wi in range(warmup_batches):
            if RANK in {-1, 0}:
                LOGGER.info(f"JIT warmup batch {wi + 1}/{warmup_batches}...")
            try:
                batch = next(loader_iter)
            except StopIteration:
                if hasattr(self.train_loader, "reset"):
                    self.train_loader.reset()
                loader_iter = iter(self.train_loader)
                batch = next(loader_iter)

            batch = self.preprocess_batch(batch)
            with autocast(enabled=self.amp, device=self.device):
                _ = self.model(batch)

        if RANK != -1 and world > 1:
            jt.sync_all(True)
        # Reset loader so training starts from the beginning.
        if hasattr(self.train_loader, "reset"):
            self.train_loader.reset()
        self._clear_memory(0.5)
    
    def _freeze_layers(self):
        """Freeze specified layers based on args.freeze."""
        freeze_list = (
            self.args.freeze
            if isinstance(self.args.freeze, list)
            else range(self.args.freeze)
            if isinstance(self.args.freeze, int)
            else []
        )
        always_freeze_names = [".dfl"]
        freeze_layer_names = [f"model.{x}." for x in freeze_list] + always_freeze_names
        self.freeze_layer_names = freeze_layer_names
        
        for k, v in self.model.named_parameters():
            is_bn_running_stat = "running_mean" in k or "running_var" in k or "num_batches_tracked" in k
            freeze_param = is_bn_running_stat or "stride" in k or ".dfl" in k or any(x in k for x in freeze_layer_names)
            if any(x in k for x in freeze_layer_names):
                LOGGER.info(f"Freezing layer '{k}'")

            if isinstance(v, jt.Var):
                if freeze_param:
                    v.stop_grad()
                elif str(v.dtype).startswith("float"):
                    v.start_grad()
    
    def _setup_amp(self, world_size):
        """Setup Automatic Mixed Precision (AMP) for training."""
        requested_amp = bool(getattr(self.args, "amp", False))
        use_cuda = jt.has_cuda and "cuda" in str(self.device).lower()
        self.amp = bool(requested_amp and use_cuda)
        if requested_amp and not self.amp and RANK in {-1, 0}:
            LOGGER.warning("WARNING ⚠️ AMP requested but CUDA is unavailable, falling back to FP32.")
        if self.amp and RANK in {-1, 0}:
            amp_level = int(os.getenv("NKYOLO_AMP_LEVEL", "3") or 3)
            LOGGER.info(f"AMP enabled (Jittor auto_mixed_precision_level={amp_level}).")
            if amp_level <= 3:
                LOGGER.info("AMP level<=3 is stability mode in Jittor and may show little/no speedup.")
        self.scaler = None
    
    def _setup_model_ddp(self, world_size):
        """Setup Distributed Data Parallel (DDP) wrapper for model in multi-GPU training."""
        if world_size <= 1:
            return
        
        if _JT_DISTRIBUTED_AVAILABLE and RANK > -1:
            from jittor import distributed as jt_dist

            self.model = jt_dist.ParallelModel(self.model)
            LOGGER.info(f"Jittor ParallelModel initialized for rank {RANK}")
        elif RANK >= 0 and jt.has_cuda:
            device_id = LOCAL_RANK if LOCAL_RANK >= 0 else RANK
            LOGGER.info(f"Jittor distributed training enabled for rank {RANK} on device {device_id}")

    def _do_train(self, world_size=1):
        """Train completed, evaluate and plot if specified by arguments."""
        if world_size > 1:
            self._setup_ddp(world_size)
        self._setup_train(world_size)
        # Clean any stale barrier flags from previous runs.
        if RANK in {-1, 0}:
            barrier_dir = self._get_ddp_barrier_dir()
            if barrier_dir.exists():
                shutil.rmtree(barrier_dir, ignore_errors=True)

        # nb (number of batches per epoch) is derived from Jittor's MPI-aware dataset
        # and is only accurate after the dataloader iterator is created.
        nb = None
        nw = -1  # warmup iterations (set after nb is known)
        last_opt_step = -1
        self.epoch_time = None
        self.epoch_time_start = time.time()
        # Delay timing start until first batch finishes to exclude JIT compile time.
        self.train_time_start = None
        self.run_callbacks("on_train_start")
        LOGGER.info(
            f'Image sizes {self.args.imgsz} train, {self.args.imgsz} val\n'
            f'Using {self.train_loader.num_workers * (world_size or 1)} dataloader workers\n'
            f"Logging results to {colorstr('bold', self.save_dir)}\n"
            f'Starting training for ' + (f"{self.args.time} hours..." if self.args.time else f"{self.epochs} epochs...")
        )
        epoch = self.start_epoch
        ddp_warmup_done = False
        self.optimizer.zero_grad()  # zero any resumed gradients to ensure stability on train start
        while True:
            self.epoch = epoch
            self.run_callbacks("on_train_epoch_start")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # suppress 'Detected lr_scheduler.step() before optimizer.step()'
                self.scheduler.step()

            self._model_train()
            # Jittor handles distributed sampling automatically via MPI
            # Jittor dataloader does not require sampler epoch updates
            # Update dataloader attributes (optional)
            mosaic_switched = False
            if epoch == (self.epochs - self.args.close_mosaic):
                self._close_dataloader_mosaic()
                self.train_loader.reset()
                mosaic_switched = True

            # In MPI training, compile train-step graph on all ranks *after* any mosaic switch.
            # This avoids rank desync where one rank compiles while another rank is inside NCCL all-reduce.
            if jt.in_mpi and int(jt.world_size) > 1 and getattr(self.args, "jit_warmup", True):
                if (not ddp_warmup_done) or mosaic_switched:
                    if RANK in {-1, 0}:
                        LOGGER.info("JIT warmup on all ranks (train-step compile).")
                    self._jit_warmup(all_ranks=True)
                    ddp_warmup_done = True

            # Initialize iterator after any reset to keep it in sync.
            loader_iter = iter(self.train_loader)
            if nb is None:
                loader_nb = len(self.train_loader)
                local_nb = max(1, int(loader_nb))
                train_items = int(getattr(self.train_loader, "dataset_size", 0) or 0)
                if train_items <= 0:
                    ds_obj = getattr(self.train_loader, "original_dataset", None) or getattr(self.train_loader, "dataset", None)
                    train_items = int(getattr(ds_obj, "ni", len(ds_obj) if ds_obj is not None else 0) or 0)
                nb = self._sync_train_num_batches(epoch, local_nb) if RANK != -1 else local_nb
                if RANK >= 0:
                    per_rank_bs = int(getattr(self.train_loader, "batch_size", self.batch_size))
                    LOGGER.info(
                        f"Rank {RANK}: loader batches={loader_nb}, local batches={local_nb}, synced batches={nb} "
                        f"(dataset size: {train_items}, batch_size: {per_rank_bs}, global_batch: {per_rank_bs * max(world_size, 1)})"
                    )
                if self.args.close_mosaic:
                    base_idx = (self.epochs - self.args.close_mosaic) * nb
                    self.plot_idx.extend([base_idx, base_idx + 1, base_idx + 2])
            # Warmup iterations (compute once after nb is known)
            if nw < 0:
                nw = max(round(self.args.warmup_epochs * nb), 100) if self.args.warmup_epochs > 0 else -1
            reset_warned = False
            if RANK in {-1, 0}:
                LOGGER.info(self.progress_string())
                pbar = TQDM(range(nb), total=nb)
            else:
                pbar = range(nb)
            self.tloss = None
            for i in pbar:
                try:
                    batch = next(loader_iter)
                except StopIteration:
                    # Keep ranks aligned by cycling the loader if it ends early.
                    if not reset_warned and RANK in {-1, 0}:
                        LOGGER.warning(
                            "WARNING ⚠️ Dataloader exhausted early; resetting to keep DDP ranks aligned. "
                            "This can happen if per-rank dataset lengths differ."
                        )
                        reset_warned = True
                    self.train_loader.reset()
                    loader_iter = iter(self.train_loader)
                    batch = next(loader_iter)
                self.run_callbacks("on_train_batch_start")
                # Warmup
                ni = i + nb * epoch
                if ni <= nw:
                    xi = [0, nw]  # x interp
                    self.accumulate = max(1, int(np.interp(ni, xi, [1, self.args.nbs / self.batch_size]).round()))
                    for j, x in enumerate(self.optimizer.param_groups):
                        # Bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                        x["lr"] = np.interp(
                            ni, xi, [self.args.warmup_bias_lr if j == 0 else 0.0, x.get("initial_lr", self.args.lr0) * self.lf(epoch)]
                        )
                        if "momentum" in x:
                            x["momentum"] = np.interp(ni, xi, [self.args.warmup_momentum, self.args.momentum])

                batch = self.preprocess_batch(batch)
                with autocast(enabled=self.amp, device=self.device):
                    self.loss, self.loss_items = self.model(batch)
                touch = self._ddp_touch_loss_params()
                if touch is not None:
                    self.loss = self.loss + touch
                if RANK != -1:
                    self.loss *= world_size
                self.tloss = (
                    (self.tloss * i + self.loss_items) / (i + 1) if self.tloss is not None else self.loss_items
                )
                if self.train_time_start is None:
                    self.train_time_start = time.time()

                # Backward (use optimizer.backward(loss) instead of loss.backward())
                if self.scaler is not None:
                    self.scaler.backward(self.loss)
                else:
                    self.optimizer.backward(self.loss)

                # Optimize - https://pyjt.org/docs/master/notes/amp_examples.html
                if ni - last_opt_step >= self.accumulate:
                    self.optimizer_step()
                    last_opt_step = ni

                    # Timed stopping
                    if self.args.time:
                        self.stop = (time.time() - self.train_time_start) > (self.args.time * 3600)
                        if self.stop:  # training time exceeded
                            break

                # Log
                if RANK in {-1, 0}:
                    loss_length = self.tloss.shape[0] if len(self.tloss.shape) else 1
                    if isinstance(self.tloss, jt.Var):
                        loss_vals = self.tloss.reshape((-1,)).numpy().tolist()
                    elif isinstance(self.tloss, np.ndarray):
                        loss_vals = self.tloss.reshape(-1).tolist()
                    elif isinstance(self.tloss, (list, tuple)):
                        loss_vals = list(self.tloss)
                    else:
                        loss_vals = [float(self.tloss)]
                    loss_vals = [float(x) for x in loss_vals[:loss_length]]
                    memory_str = self._get_memory_str()  # Get memory string or empty
                    format_str = "%11s" * (1 + (1 if memory_str else 0)) + "%11.4g" * loss_length + "%11i%11i"
                    log_items = [f"{epoch + 1}/{self.epochs}"]
                    if memory_str:
                        log_items.append(memory_str)
                    instances = self._batch_instances(batch)
                    imgsz = int(batch["img"].shape[-1]) if hasattr(batch.get("img", None), "shape") else int(self.args.imgsz)
                    log_items.extend([*loss_vals, instances, imgsz])
                    pbar.set_description(format_str % tuple(log_items))
                    self.run_callbacks("on_batch_end")
                    if self.args.plots and ni in self.plot_idx:
                        self.plot_training_samples(batch, ni)

                self.run_callbacks("on_train_batch_end")

            self.lr = {f"lr/pg{ir}": x["lr"] for ir, x in enumerate(self.optimizer.param_groups)}  # for loggers
            self.run_callbacks("on_train_epoch_end")
            if RANK != -1:
                self._epoch_barrier(epoch, "train_done")
            final_epoch = epoch + 1 >= self.epochs
            if RANK in {-1, 0} and self.ema:
                self.ema.update_attr(
                    self.model,
                    include=["yaml", "nc", "args", "names", "stride", "class_weights"],
                )

            # Validation (all ranks enter validate() under MPI to keep single_process_scope sync aligned).
            do_val = bool(self.args.val or final_epoch)
            if do_val:
                self._clear_memory(threshold=0.5)  # prevent VRAM spike
                val_metrics, val_fitness = self.validate()
                if RANK in {-1, 0}:
                    self.metrics, self.fitness = val_metrics, val_fitness

            # Important: materialize loss scalars on all ranks before rank-divergent code.
            # In Jittor MPI, forcing this only on rank0 can deadlock at epoch end.
            train_loss_items = self.label_loss_items(self.tloss)

            if RANK in {-1, 0}:
                self.save_metrics(metrics={**train_loss_items, **self.metrics, **self.lr})
                self.stop |= self.stopper(epoch + 1, self.fitness) or final_epoch
                if self.args.time:
                    self.stop |= (time.time() - self.train_time_start) > (self.args.time * 3600)

                # Save model (rank0 only inside save_model).
                if self.args.save or final_epoch:
                    self.save_model()
                    self.run_callbacks("on_model_save")
            if RANK != -1:
                self._epoch_barrier(epoch, "epoch_done")

            # Scheduler
            t = time.time()
            self.epoch_time = t - self.epoch_time_start
            self.epoch_time_start = t
            if self.args.time:
                mean_epoch_time = (t - (self.train_time_start or t)) / (epoch - self.start_epoch + 1)
                self.epochs = self.args.epochs = math.ceil(self.args.time * 3600 / mean_epoch_time)
                self._setup_scheduler()
                self.scheduler.last_epoch = self.epoch  # do not move
                self.stop |= epoch >= self.epochs  # stop if exceeded epochs
            self.run_callbacks("on_fit_epoch_end")
            self._clear_memory(0.5)  # clear if memory utilization > 50%

            # Early Stopping
            if RANK != -1:
                self.stop = self._sync_stop_flag(epoch, self.stop)
            if self.stop:
                break  # must break all DDP ranks
            epoch += 1

        if RANK in {-1, 0}:
            seconds = time.time() - (self.train_time_start or time.time())
            LOGGER.info(f"\n{epoch - self.start_epoch + 1} epochs completed in {seconds / 3600:.3f} hours.")

        if getattr(self.args, "final_eval", True):
            if RANK != -1:
                self._epoch_barrier(epoch, "before_final_eval")
            try:
                self.final_eval()
            except Exception as e:
                # Surface the failure (exit code must not be 0 when best.pkl was
                # never validated) — but only after the barrier in `finally` has
                # released the other ranks from a timeout-less MPI wait.
                LOGGER.error(f"Final eval failed: {type(e).__name__}: {e}")
                raise
            finally:
                if RANK != -1:
                    self._epoch_barrier(epoch, "after_final_eval")

        if RANK in {-1, 0}:
            if self.args.plots:
                LOGGER.debug("Train end: plotting metrics...")
                self.plot_metrics()
            LOGGER.debug("Train end: running callbacks...")
            self.run_callbacks("on_train_end")
            LOGGER.debug("Train end: callbacks finished.")
        self._cleanup_loaders()
        LOGGER.debug("Train end: dataloaders closed.")
        self._clear_memory()
        LOGGER.debug("Train end: memory cleared.")
        self.run_callbacks("teardown")
        LOGGER.debug("Train end: teardown finished.")
        # NOTE: no os._exit() here — a hard exit inside train() silently kills
        # best-weights reload in Model.train() and any user code after it when
        # ranks execute the entry script. The generated DDP wrapper (see
        # nkyolo/utils/dist.py) hard-exits after trainer.train() for launches
        # that need it; loader teardown above handles lingering workers.

    def auto_batch(self, max_num_obj=0):
        """Calculate optimal batch size based on model and device memory constraints."""
        return check_train_batch_size(
            model=self.model,
            imgsz=self.args.imgsz,
            amp=self.amp,
            batch=self.batch_size,
            max_num_obj=max_num_obj,
        )  # returns batch size

    def _get_gpu_memory_stats(self):
        """Return GPU memory stats as (used_gb, total_gb, usage_fraction) or None when unavailable."""
        device_str = str(self.device).lower()
        if "cpu" in device_str or "mps" in device_str or not jt.has_cuda:
            return None
        mi = jt.get_mem_info()
        used_bytes = int(getattr(mi, "total_cuda_used", 0) or 0)
        total_bytes = int(getattr(mi, "total_cuda_ram", 0) or 0)
        if total_bytes <= 0:
            return None
        used_gb = used_bytes / (1024**3)
        total_gb = total_bytes / (1024**3)
        usage = used_bytes / total_bytes
        return used_gb, total_gb, usage

    def _get_memory(self, fraction=False):
        """Get accelerator memory utilization in GB or as a fraction of total memory."""
        device_str = str(self.device).lower()

        if "mps" in device_str:
            return __import__("psutil").virtual_memory().percent / 100 if fraction else 0.0

        if "cpu" in device_str:
            return 0.0

        stats = self._get_gpu_memory_stats()
        if stats is None:
            return None
        used_gb, _, usage = stats
        return usage if fraction else used_gb

    def _get_memory_str(self):
        """Get memory string for display, returns empty string if unavailable."""
        stats = self._get_gpu_memory_stats()
        if stats is None:
            return ""
        used_gb, _, _ = stats
        return f"{used_gb:.1f}G"

    @staticmethod
    def _batch_instances(batch):
        """Return number of GT instances in current batch."""
        cls = batch.get("cls", None) if isinstance(batch, dict) else None
        if cls is None:
            return 0
        shape = getattr(cls, "shape", None)
        if shape is not None and len(shape):
            return int(shape[0])
        if isinstance(cls, (list, tuple)):
            total = 0
            for x in cls:
                xs = getattr(x, "shape", None)
                if xs is not None and len(xs):
                    total += int(xs[0])
                elif hasattr(x, "__len__"):
                    total += len(x)
                else:
                    total += 1
            return int(total)
        return int(len(cls)) if hasattr(cls, "__len__") else 0

    def _model_train(self):
        """Set model in training mode."""
        self.model.train()
        # Freeze BN stat for frozen layers
        for n, m in self.model.named_modules():
            if any(filter(lambda f: f in n, self.freeze_layer_names)) and isinstance(m, nn.BatchNorm2d):
                m.eval()

    def _ddp_touch_loss_params(self):
        """Return a tiny term that references all trainable params to stabilize DDP all-reduce graphs."""
        if RANK == -1 or get_world_size() <= 1 or self.optimizer is None:
            return None
        touch_scale = float(getattr(self.args, "ddp_touch_scale", 1e-12) or 0.0)
        if touch_scale <= 0.0:
            return None
        if self._ddp_touch_params is None:
            params = []
            for pg in self.optimizer.param_groups:
                for p in pg.get("params", []):
                    if isinstance(p, jt.Var) and not p.is_stop_grad():
                        params.append(p)
            self._ddp_touch_params = params
        if not self._ddp_touch_params:
            return None
        # Build one small vector op instead of a long scalar add-chain to reduce JIT graph complexity.
        terms = [p.reshape((-1,))[:1] for p in self._ddp_touch_params]
        if not terms:
            return None
        return jt.concat(terms, dim=0).sum() * touch_scale

    def _clear_memory(self, threshold: float = None):
        """Clear accelerator memory by calling garbage collector (Jittor uses jt.gc() per official docs)."""
        if threshold:
            assert 0 <= threshold <= 1, "Threshold must be between 0 and 1."
            memory_frac = self._get_memory(fraction=True)
            if memory_frac is None or memory_frac <= threshold:
                return
        gc.collect()
        if jt.has_cuda and "cuda" in str(self.device).lower():
            jt.gc()  # Jittor's official way to release GPU memory

    def read_results_csv(self):
        """Read results.csv into a dict using pandas."""
        import pandas as pd  # scope for faster 'import nkyolo'

        return pd.read_csv(self.csv).to_dict(orient="list")

    def save_model(self):
        """Save model training checkpoints with additional metadata."""
        if RANK not in {-1, 0}:
            return

        def _half_state_dict(obj):
            """Create a portable FP16 state dict without mutating the model."""
            from nkyolo.utils.jittor_utils import state_dict_to_numpy

            if isinstance(obj, dict):
                return state_dict_to_numpy(obj, fp16=True)
            return state_dict_to_numpy(obj.state_dict(), fp16=True)
        
        # Prepare checkpoint data
        base_model = self.model_base or self.model
        ema_state = None
        if self.ema:
            if self.ema.ema is not None:
                ema_model = self.ema.ema
                ema_updates = self.ema.updates
            else:
                ema_state = self.ema.state_dict()
                ema_model = base_model
                ema_updates = self.ema.updates
        else:
            ema_model = base_model
            ema_updates = 0
        save_optimizer = bool(getattr(self.args, "save_optimizer", True))
        save_results = bool(getattr(self.args, "save_results", True))
        optimizer_state = None
        if save_optimizer:
            optimizer_state = convert_optimizer_state_dict_to_fp16(safe_deepcopy_jittor(self.optimizer.state_dict()))
        train_results = self.read_results_csv() if save_results else None
        checkpoint_data = {
            "epoch": self.epoch,
            "best_fitness": self.best_fitness,
            "model": None,  # resume and final checkpoints derive from EMA
            "ema": _half_state_dict(ema_state if ema_state is not None else ema_model),
            "updates": ema_updates,
            "optimizer": optimizer_state,
            "train_args": vars(self.args),  # save as dict
            "train_metrics": {**self.metrics, **{"fitness": self.fitness}},
            "train_results": train_results,
            "date": datetime.now().isoformat(),
            "version": __version__,
            "license": "AGPL-3.0 (https://ultralytics.com/license)",
            "docs": "https://docs.ultralytics.com",
            "model_yaml": ema_model.yaml,  # save model config for model reconstruction
            "names": base_model.names,
            "nc": base_model.nc,
        }

        # Save checkpoints directly to files
        jt.save(checkpoint_data, str(self.last))  # save last.pkl (portable Jittor checkpoint)
        if self.best_fitness == self.fitness:
            jt.save(checkpoint_data, str(self.best))  # save best.pkl
        if (self.save_period > 0) and (self.epoch % self.save_period == 0):
            jt.save(checkpoint_data, str(self.wdir / f"epoch{self.epoch}.pkl"))  # save epoch, i.e. 'epoch3.pkl'
        # if self.args.close_mosaic and self.epoch == (self.epochs - self.args.close_mosaic - 1):
        #    jt.save(checkpoint_data, str(self.wdir / "last_mosaic.pkl"))  # save mosaic checkpoint

    def get_dataset(self):
        """
        Get train, val path from data dict if it exists.

        Returns None if data format is not recognized.
        """
        if self.args.task == "classify":
            data = check_cls_dataset(self.args.data)
        elif self.args.data.split(".")[-1] in {"yaml", "yml"} or self.args.task in {
            "detect",
            "segment",
            "pose",
            "obb",
        }:
            data = check_det_dataset(self.args.data)
            if "yaml_file" in data:
                self.args.data = data["yaml_file"]  # for validating 'yolo train data=url.zip' usage
        else:
            raise ValueError(emojis(f"Dataset '{clean_url(self.args.data)}' error ❌ unsupported format"))
        self.data = data
        return data["train"], data.get("val") or data.get("test")

    def setup_model(self):
        """Load/create/download model for any task."""
        if isinstance(self.model, jt.nn.Module):  # if model is loaded beforehand. No setup needed
            return None  # Explicitly return None when model is already set

        cfg, weights = self.model, None
        ckpt = None
        model_path = str(self.model)
        if Path(model_path).suffix.lower() in {".pt", ".pth", ".pkl"}:
            weights, ckpt = attempt_load_one_weight(self.model)
            cfg = weights.yaml
        elif isinstance(self.args.pretrained, (str, Path)):
            weights, _ = attempt_load_one_weight(self.args.pretrained)
        self.model = self.get_model(cfg=cfg, weights=weights, verbose=bool(self.args.verbose))  # calls Model(cfg, weights)
        return ckpt

    def optimizer_step(self):
        """Perform a single step of the training optimizer with gradient clipping and EMA update."""
        if self.scaler is not None:
            self.scaler.unscale_(self.optimizer)
            self.optimizer.clip_grad_norm(10.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.clip_grad_norm(10.0)
            self.optimizer.step()
        self.optimizer.zero_grad()
        if self.ema:
            interval = int(getattr(self.args, "ema_interval", 1) or 1)
            interval = max(1, interval)
            self.ema_steps += 1
            if self.ema_steps % interval == 0:
                self.ema.update(self.model_base or self.model, updates=self.ema_steps)

    def preprocess_batch(self, batch):
        """Allows custom preprocessing model inputs and ground truths depending on task type."""
        return batch

    def _validate_inprocess_with_backoff(self):
        """Run in-process validation and auto-reduce val batch size if CUDA OOM occurs."""
        if self.validator is None:
            return {}

        cur_batch = int(getattr(self, "val_batch_size", 0) or 0)
        if cur_batch <= 0:
            cur_batch = int(getattr(self.test_loader, "batch_size", 1))
        cur_batch = max(1, cur_batch)

        while True:
            try:
                return self.validator(self)
            except (RuntimeError, MemoryError) as e:
                if (not self._is_oom_error(e)) or cur_batch <= 1:
                    raise
                next_batch = max(1, cur_batch // 2)
                LOGGER.warning(
                    f"WARNING ⚠️ Validation OOM at batch={cur_batch}, retrying with batch={next_batch}."
                )
                self._clear_memory()
                self._reset_val_loader(next_batch)
                cur_batch = next_batch

    def validate(self):
        """
        Runs validation on test set using self.validator.

        The returned dict is expected to contain "fitness" key.
        """
        def _run_inprocess():
            # Force EMA weights for validation when EMA is enabled.
            model_for_ema = self.model_base or self.model
            ema_applied = False
            if self.ema and getattr(self.ema, "ema", None) is None:
                ema_applied = self.ema.apply_to(model_for_ema)
            metrics_local = self._validate_inprocess_with_backoff()
            if ema_applied:
                self.ema.restore(model_for_ema)
            return metrics_local

        metrics = {}
        if jt.in_mpi and int(jt.world_size) > 1:
            @jt.single_process_scope(rank=0)
            def _rank0_validate():
                return _run_inprocess()

            res = _rank0_validate()
            if RANK == 0 and isinstance(res, dict):
                metrics = res
        else:
            metrics = _run_inprocess()

        if RANK not in {-1, 0}:
            return {}, -1.0

        fitness = metrics.pop("fitness", -self.loss.detach().cpu().numpy())  # use loss as fitness measure if not found
        if not self.best_fitness or self.best_fitness < fitness:
            self.best_fitness = fitness
        return metrics, fitness

    def get_model(self, cfg=None, weights=None, verbose=True):
        """Get model and raise NotImplementedError for loading cfg files."""
        raise NotImplementedError("This task trainer doesn't support loading cfg files")

    def get_validator(self):
        """Returns a NotImplementedError when the get_validator function is called."""
        raise NotImplementedError("get_validator function not implemented in trainer")

    def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode="train"):
        """Returns dataloader derived from jt.data.Dataloader."""
        raise NotImplementedError("get_dataloader function not implemented in trainer")

    def build_dataset(self, img_path, mode="train", batch=None):
        """Build dataset."""
        raise NotImplementedError("build_dataset function not implemented in trainer")

    def label_loss_items(self, loss_items=None, prefix="train"):
        """
        Returns a loss dict with labelled training loss items tensor.

        Note:
            This is not needed for classification but necessary for segmentation & detection
        """
        return {"loss": loss_items} if loss_items is not None else ["loss"]

    def set_model_attributes(self):
        """To set or update model parameters before training."""
        self.model.names = self.data["names"]

    def build_targets(self, preds, targets):
        """Builds target tensors for training YOLO model."""
        pass

    def progress_string(self):
        """Returns a string describing training progress."""
        return ""

    # TODO: may need to put these following functions into callback
    def plot_training_samples(self, batch, ni):
        """Plots training samples during YOLO training."""
        pass

    def plot_training_labels(self):
        """Plots training labels for YOLO model."""
        pass

    def save_metrics(self, metrics):
        """Saves training metrics to a CSV file."""
        if RANK not in {-1, 0}:
            return
        keys, vals = list(metrics.keys()), list(metrics.values())
        n = len(metrics) + 2  # number of cols
        s = "" if self.csv.exists() else (("%s," * n % tuple(["epoch", "time"] + keys)).rstrip(",") + "\n")  # header
        t = time.time() - self.train_time_start
        with open(self.csv, "a") as f:
            f.write(s + ("%.6g," * n % tuple([self.epoch + 1, t] + vals)).rstrip(",") + "\n")

    def plot_metrics(self):
        """Plot and display metrics visually."""
        pass

    def on_plot(self, name, data=None):
        """Registers plots (e.g. to be consumed in callbacks)."""
        path = Path(name)
        self.plots[path] = {"data": data, "timestamp": time.time()}

    def _close_loader(self, loader, name: str):
        """Best-effort close for a dataloader and its worker processes."""
        if loader is None:
            return
        try:
            loader.close()
        except Exception as e:
            LOGGER.warning(f"Loader cleanup warning for {name}: {type(e).__name__}: {e}")

    def _cleanup_loaders(self):
        """Release dataloader iterators/workers before process teardown (idempotent)."""
        validator_loader = self.validator.dataloader if self.validator is not None else None
        if self.train_loader is None and self.test_loader is None and validator_loader is None:
            return
        LOGGER.debug("Train end: closing dataloaders...")
        self._close_loader(self.train_loader, "train_loader")
        if self.test_loader is not validator_loader:
            self._close_loader(self.test_loader, "test_loader")
        self._close_loader(validator_loader, "validator.dataloader")
        self.train_loader = None
        self.test_loader = None
        if self.validator is not None:
            self.validator.dataloader = None

    def final_eval(self):
        """Performs final evaluation and validation for object detection YOLO model."""
        def _json_safe(v):
            if isinstance(v, Path):
                return str(v)
            if isinstance(v, np.generic):
                return v.item()
            if isinstance(v, (list, tuple)):
                return [_json_safe(x) for x in v]
            if isinstance(v, dict):
                return {str(k): _json_safe(x) for k, x in v.items()}
            return v

        def _standalone_val_kwargs():
            # Only reachable from _run_rank0_final_eval, which returns early
            # when self.validator is None.
            src_args = self.validator.args
            keys = (
                "imgsz",
                "split",
                "save_json",
                "conf",
                "iou",
                "max_det",
                "single_cls",
                "agnostic_nms",
                "classes",
                "rect",
                "dnn",
                "verbose",
                "save_txt",
                "save_conf",
                "plots",
                "half",
                "val_half",
            )
            kwargs = {}
            for k in keys:
                if hasattr(src_args, k):
                    kwargs[k] = _json_safe(getattr(src_args, k))
            kwargs["data"] = str(self.args.data)
            # Single source: _run_rank0_final_eval sets validator.args.batch
            # before any final-eval path runs.
            kwargs["batch"] = int(src_args.batch)
            # The eval subprocess runs without any MPI context (env is scrubbed
            # below), so normal dataloader workers are safe — keep the
            # configured values instead of serializing the whole val pass.
            kwargs["workers"] = int(getattr(src_args, "workers", 8))
            kwargs["val_workers"] = int(getattr(src_args, "val_workers", -1))
            kwargs["device"] = "" if "cuda" in str(self.device).lower() else str(self.device)
            kwargs["project"] = str(self.save_dir.parent)
            kwargs["name"] = self.save_dir.name
            kwargs["exist_ok"] = True
            return kwargs

        def _run_final_eval_subprocess(weight_path: Path):
            metrics_path = self.save_dir / "final_eval_metrics.json"
            if metrics_path.exists():
                metrics_path.unlink()

            # Snapshot kwargs BEFORE cleanup: _standalone_val_kwargs reads
            # test_loader.batch_size, which _cleanup_loaders() nulls.
            kwargs_json = json.dumps(_standalone_val_kwargs(), ensure_ascii=True)

            # The eval process shares this rank's visible GPUs — release the
            # training loaders and cached GPU memory before it allocates.
            self._cleanup_loaders()
            self._clear_memory()

            env = os.environ.copy()
            for k in (
                "OMPI_COMM_WORLD_RANK",
                "OMPI_COMM_WORLD_LOCAL_RANK",
                "OMPI_COMM_WORLD_SIZE",
                "PMI_RANK",
                "PMI_LOCAL_RANK",
                "PMI_SIZE",
                "RANK",
                "LOCAL_RANK",
                "WORLD_SIZE",
                "MASTER_ADDR",
                "MASTER_PORT",
            ):
                env.pop(k, None)

            script = """
import json
import sys

from nkyolo import YOLO

weight_path = sys.argv[1]
metrics_path = sys.argv[2]
kwargs = json.loads(sys.argv[3])

model = YOLO(weight_path)
metrics = model.val(**kwargs)
payload = getattr(metrics, "results_dict", None)
if payload is None:
    payload = metrics if isinstance(metrics, dict) else {}
with open(metrics_path, "w", encoding="utf-8") as f:
    json.dump(payload, f)
"""
            LOGGER.info(f"Final eval: launching isolated single-process validation for {weight_path}...")
            proc = subprocess.run(
                [sys.executable, "-c", script, str(weight_path), str(metrics_path), kwargs_json],
                cwd=str(Path(__file__).resolve().parents[2]),
                env=env,
                check=False,
            )
            if proc.returncode != 0:
                raise RuntimeError(f"Final eval subprocess failed with exit code {proc.returncode}.")
            if not metrics_path.exists():
                raise FileNotFoundError(f"Final eval metrics file was not created: {metrics_path}")
            with open(metrics_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, dict) else {}

        def _run_rank0_final_eval():
            if self.validator is None:
                return None
            ckpt = {}
            metrics = None
            for f in self.last, self.best:
                if not f.exists():
                    continue
                if f is self.last:
                    LOGGER.info(f"\nFinal eval: stripping optimizer from {f}...")
                    ckpt = strip_optimizer(f)
                    continue

                k = "train_results"  # update best.pkl train_metrics from last.pkl
                LOGGER.info(f"\nFinal eval: preparing {f}...")
                strip_optimizer(f, updates={k: ckpt[k]} if k in ckpt else None)
                # Both eval paths read these through validator.args
                # (_standalone_val_kwargs copies batch/plots into the subprocess).
                self.validator.args.batch = int(self.val_batch_size or getattr(self.test_loader, "batch_size", 1))
                self.validator.args.plots = self.args.plots
                LOGGER.info(f"\nFinal eval: validating {f} on {self.device}...")
                # get_world_size() > 1 implies RANK == 0 here: this function only
                # runs for RANK in {-1, 0}, and world size is 1 when RANK == -1.
                if get_world_size() > 1:
                    metrics = _run_final_eval_subprocess(f)
                else:
                    # Final standalone validation should run on the training
                    # device, not on the raw persisted device string.
                    self.validator.args.device = self.device
                    self.validator.device = self.device
                    metrics = self.validator(model=f)
                if isinstance(metrics, dict):
                    metrics.pop("fitness", None)
            return metrics

        if RANK in {-1, 0}:
            res = _run_rank0_final_eval()
            if isinstance(res, dict):
                # Only overwrite last-epoch metrics when final eval produced
                # results (best.pkl may not exist, e.g. save=False + early stop).
                self.metrics = res
                self.run_callbacks("on_fit_epoch_end")

    def check_resume(self, overrides):
        """Check if resume checkpoint exists and update arguments accordingly."""
        resume = self.args.resume
        if resume:
            exists = isinstance(resume, (str, Path)) and Path(resume).exists()
            last = Path(check_file(resume) if exists else get_latest_run())
            if not last.exists():
                raise FileNotFoundError(
                    "Resume checkpoint not found. Please pass a valid checkpoint to resume from, "
                    "i.e. 'yolo train resume model=path/to/last.pkl'"
                )

            ckpt_args = attempt_load_weights(last).args
            if not Path(ckpt_args["data"]).exists():
                ckpt_args["data"] = self.args.data

            resume = True
            self.args = get_cfg(ckpt_args)
            self.args.model = self.args.resume = str(last)  # reinstate model
            for k in (
                "imgsz",
                "batch",
                "device",
                "close_mosaic",
            ):
                if k in overrides:
                    setattr(self.args, k, overrides[k])
        self.resume = resume

    def resume_training(self, ckpt):
        """Resume YOLO training from given epoch and best fitness."""
        if ckpt is None or not self.resume:
            return
        best_fitness = 0.0
        start_epoch = ckpt.get("epoch", -1) + 1
        if ckpt.get("optimizer", None) is not None:
            self.optimizer.load_state_dict(ckpt["optimizer"])  # optimizer
            best_fitness = ckpt["best_fitness"]
        if self.ema and ckpt.get("ema"):
            self.ema.load_state_dict(state_dict_to_jittor(ckpt["ema"]))  # EMA
            self.ema.updates = ckpt["updates"]
        assert start_epoch > 0, (
            f"{self.args.model} training to {self.epochs} epochs is finished, nothing to resume.\n"
            f"Start a new training without resuming, i.e. 'yolo train model={self.args.model}'"
        )
        LOGGER.info(f"Resuming training {self.args.model} from epoch {start_epoch + 1} to {self.epochs} total epochs")
        if self.epochs < start_epoch:
            LOGGER.info(
                f"{self.model} has been trained for {ckpt['epoch']} epochs. Fine-tuning for {self.epochs} more epochs."
            )
            self.epochs += ckpt["epoch"]  # finetune additional epochs
        self.best_fitness = best_fitness
        self.start_epoch = start_epoch
        if start_epoch > (self.epochs - self.args.close_mosaic):
            self._close_dataloader_mosaic()

    def _close_dataloader_mosaic(self):
        """Update dataloaders to stop using mosaic augmentation."""
        LOGGER.info("Closing dataloader mosaic")
        self.train_loader.dataset.close_mosaic(hyp=copy(self.args))

    def build_optimizer(self, model, name="auto", lr=0.001, momentum=0.9, decay=1e-5, iterations=1e5):
        """
        Constructs an optimizer for the given model, based on the specified optimizer name, learning rate, momentum,
        weight decay, and number of iterations.

        Args:
            model (jt.nn.Module): The model for which to build an optimizer.
            name (str, optional): The name of the optimizer to use. If 'auto', the optimizer is selected
                based on the number of iterations. Default: 'auto'.
            lr (float, optional): The learning rate for the optimizer. Default: 0.001.
            momentum (float, optional): The momentum factor for the optimizer. Default: 0.9.
            decay (float, optional): The weight decay for the optimizer. Default: 1e-5.
            iterations (float, optional): The number of iterations, which determines the optimizer if
                name is 'auto'. Default: 1e5.

        Returns:
            (jt.optim.Optimizer): The constructed optimizer.
        """
        bn = tuple(v for k, v in nn.__dict__.items() if "Norm" in k)  # normalization layers, i.e. BatchNorm2d()

        def _group_params():
            groups = [], [], []
            for module_name, module in model.named_modules():
                for param_name, param in module.named_parameters(recurse=False):
                    fullname = f"{module_name}.{param_name}" if module_name else param_name
                    if "running_mean" in fullname or "running_var" in fullname or "num_batches_tracked" in fullname:
                        continue
                    if isinstance(param, jt.Var) and param.is_stop_grad():
                        continue
                    if "bias" in fullname:  # bias (no decay)
                        groups[2].append(param)
                    elif isinstance(module, bn):  # weight (no decay)
                        groups[1].append(param)
                    else:  # weight (with decay)
                        groups[0].append(param)
            return groups

        g = _group_params()
        if name == "auto":
            LOGGER.info(
                f"{colorstr('optimizer:')} 'optimizer=auto' found, "
                f"ignoring 'lr0={self.args.lr0}' and 'momentum={self.args.momentum}' and "
                f"determining best 'optimizer', 'lr0' and 'momentum' automatically... "
            )
            nc = getattr(model, "nc", 10)  # number of classes
            lr_fit = round(0.002 * 5 / (4 + nc), 6)  # lr0 fit equation to 6 decimal places
            name, lr, momentum = ("SGD", 0.01, 0.9) if iterations > 10000 else ("AdamW", lr_fit, 0.9)
            self.args.warmup_bias_lr = 0.0  # no higher than 0.01 for Adam

        if not (g[0] or g[1] or g[2]):
            for k, v in model.named_parameters():
                is_bn_running_stat = "running_mean" in k or "running_var" in k or "num_batches_tracked" in k
                freeze_param = is_bn_running_stat or "stride" in k or ".dfl" in k or any(
                    x in k for x in self.freeze_layer_names
                )
                if isinstance(v, jt.Var) and not freeze_param and str(v.dtype).startswith("float"):
                    v.start_grad()
            g = _group_params()

        base_group = None
        base_weight_decay = 0.0
        if g[2]:
            base_group = g[2]
            base_weight_decay = 0.0
        elif g[0]:
            base_group = g[0]
            base_weight_decay = decay
        elif g[1]:
            base_group = g[1]
            base_weight_decay = 0.0
        else:
            raise ValueError("No trainable parameters found for optimizer construction.")

        if name in {"Adam", "Adamax", "AdamW", "NAdam", "RAdam"}:
            optimizer = getattr(optim, name, optim.Adam)(base_group, lr=lr, betas=(momentum, 0.999), weight_decay=0.0)
        elif name == "RMSProp":
            optimizer = optim.RMSprop(base_group, lr=lr, momentum=momentum)
        elif name == "SGD":
            optimizer = optim.SGD(base_group, lr=lr, momentum=momentum, nesterov=True)
        else:
            raise NotImplementedError(
                f"Optimizer '{name}' not found in list of available optimizers "
                f"[Adam, AdamW, NAdam, RAdam, RMSProp, SGD, auto]."
            )

        optimizer.param_groups[0]["weight_decay"] = base_weight_decay
        if base_group is not g[0] and g[0]:
            optimizer.add_param_group({"params": g[0], "weight_decay": decay})  # add g0 with weight_decay
        if base_group is not g[1] and g[1]:
            optimizer.add_param_group({"params": g[1], "weight_decay": 0.0})  # add g1 (BatchNorm2d weights)
        if base_group is not g[2] and g[2]:
            optimizer.add_param_group({"params": g[2], "weight_decay": 0.0})  # add g2 (biases)
        LOGGER.info(
            f"{colorstr('optimizer:')} {type(optimizer).__name__}(lr={lr}, momentum={momentum}) with parameter groups "
            f'{len(g[1])} weight(decay=0.0), {len(g[0])} weight(decay={decay}), {len(g[2])} bias(decay=0.0)'
        )
        return optimizer
