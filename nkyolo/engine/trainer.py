# NK-YOLO 🚀 AGPL-3.0 License
# Refer to https://github.com/ultralytics/ultralytics/blob/main/ultralytics/engine/trainer.py
"""
Train a model on a dataset.

Usage:
    $ yolo mode=train model=yolov8n.pt data=coco8.yaml imgsz=640 epochs=100 batch=16
"""

import gc
import math
import os
import pickle
import shutil
import signal
import subprocess
import tempfile
import time
import warnings
import importlib.util
from copy import copy
from datetime import datetime, timedelta
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
        self.epochs = self.args.epochs
        self.start_epoch = 0
        if RANK == -1:
            print_args(vars(self.args))

        # Device
        if self.device in {"cpu", "mps"}:
            self.args.workers = 0  # faster CPU training as time dominated by inference, not dataloading

        # Model and Dataset
        self.model = check_model_file_from_stem(self.args.model)  # add suffix, i.e. yolov8n -> yolov8n.pt

        self.trainset, self.testset = self.get_dataset()
        self.ema = None
        
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
        is_mpi_env = bool(jt.mpi)
        
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

        # If dataset is tiny, DDP often deadlocks; fall back to single-process training.
        if world_size > 1 and not is_mpi_env:
            total_samples = len(self.trainset) if self.trainset is not None else 0
            per_rank = math.ceil(total_samples / world_size) if total_samples else 0
            per_rank_batch = 0
            if per_rank:
                per_rank_batch = max(1, min(self.args.batch // world_size, per_rank))
            num_batches = math.ceil(per_rank / per_rank_batch) if per_rank_batch else 0
            if per_rank < 2 or num_batches < 2:
                LOGGER.warning(
                    f"WARNING ⚠️ DDP disabled due to small dataset "
                    f"(total={total_samples}, per_rank={per_rank}, batches/rank={num_batches})."
                )
                world_size = 1
                if device_list:
                    # Force single-device selection to avoid multi-GPU setup.
                    self.args.device = str(device_list[0])
                    self.device = select_device(self.args.device, self.args.batch, verbose=False)

        # Run subprocess if DDP training and NOT already in MPI environment, else train normally
        if world_size > 1 and not is_mpi_env:
            # Argument checks
            if self.args.rect:
                LOGGER.warning("WARNING ⚠️ 'rect=True' is incompatible with Multi-GPU training, setting 'rect=False'")
                self.args.rect = False
            if self.args.batch < 1.0:
                LOGGER.warning(
                    "WARNING ⚠️ 'batch<1' for AutoBatch is incompatible with Multi-GPU training, setting "
                    "default 'batch=16'"
                )
                self.args.batch = 16

            # Command
            cmd, file, env = generate_ddp_command(world_size, self)
            LOGGER.info(f'{colorstr("DDP:")} debug command {" ".join(cmd)}')
            subprocess.run(cmd, check=True, env=env)
            ddp_cleanup(self, str(file))

        else:
            self._do_train(world_size)

    def _get_world_size(self, default=1):
        """Return world size from Jittor MPI when available, or fallback to env/default."""
        if jt.mpi:
            return int(jt.world_size)
        if "OMPI_COMM_WORLD_SIZE" in os.environ:
            return int(os.environ["OMPI_COMM_WORLD_SIZE"])
        if "PMI_SIZE" in os.environ:
            return int(os.environ["PMI_SIZE"])
        if "WORLD_SIZE" in os.environ:
            return int(os.environ["WORLD_SIZE"])
        return default

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

    def _ddp_barrier(self, tag: str, max_wait: float = 300.0):
        """Filesystem-based barrier for MPI DDP to keep ranks in sync."""
        if RANK == -1:
            return
        world_size = self._get_world_size(default=1)
        if world_size <= 1:
            return
        barrier_dir = self.save_dir / "_ddp_barrier"
        barrier_dir.mkdir(parents=True, exist_ok=True)
        flag = barrier_dir / f"{tag}_rank{RANK}"
        flag.write_text("1")
        start = time.time()
        while time.time() - start < max_wait:
            if all((barrier_dir / f"{tag}_rank{r}").exists() for r in range(world_size)):
                break
            time.sleep(0.05)
        else:
            LOGGER.warning(f"Rank {RANK}: Barrier '{tag}' timed out after {max_wait}s")
            return

        if RANK == 0:
            for r in range(world_size):
                (barrier_dir / f"{tag}_rank{r}").unlink()
    def _broadcast_object(self, obj, src=0):
        """Broadcast an object from source rank to all ranks in MPI environment.
        
        Args:
            obj: Object to broadcast (only used on src rank)
            src: Source rank to broadcast from (default: 0)
            
        Returns:
            The broadcasted object (same value on all ranks)
        """
        if RANK == -1:
            # Not in distributed training, just return the object
            return obj
        
        # Use temporary file for broadcasting in MPI environment
        # This is a simple implementation using file system
        broadcast_file = self.save_dir / f"_broadcast_{src}.pkl"
        
        if RANK == src:
            # Source rank: write object to file
            with open(broadcast_file, 'wb') as f:
                pickle.dump(obj, f)
            # Small delay to ensure file is written
            time.sleep(0.01)
        
        # All ranks: wait for file and read it
        max_wait = 5.0  # Maximum wait time in seconds
        wait_time = 0.0
        while not broadcast_file.exists() and wait_time < max_wait:
            time.sleep(0.01)
            wait_time += 0.01
        
        if not broadcast_file.exists():
            LOGGER.warning(f"Rank {RANK}: Broadcast file not found after {max_wait}s, using default value")
            return obj if RANK == src else None
        
        # Read the broadcasted object
        with open(broadcast_file, 'rb') as f:
            result = pickle.load(f)
        
        # Clean up: only rank 0 removes the file
        if RANK == 0:
            broadcast_file.unlink()
        
        return result

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
            # CRITICAL: Disable Jittor's parallel compilation in MPI environments to prevent segfaults
            # Multiple MPI processes compiling operators simultaneously causes memory corruption
            # This must be set BEFORE any Jittor operations that trigger compilation
            # Use environment variables to control Jittor compilation behavior
            os.environ["JIT_PARALLEL"] = "0"  # Disable Jittor's parallel JIT compilation
            os.environ["JITTOR_COMPILE_THREADS"] = "1"  # Use single-threaded compilation
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
            
            # Get actual world size from environment if available
            actual_world_size = world_size
            if jt.mpi:
                actual_world_size = int(jt.world_size)
            elif "OMPI_COMM_WORLD_SIZE" in os.environ:
                actual_world_size = int(os.environ["OMPI_COMM_WORLD_SIZE"])
            elif "PMI_SIZE" in os.environ:
                actual_world_size = int(os.environ["PMI_SIZE"])
            elif "WORLD_SIZE" in os.environ:
                actual_world_size = int(os.environ["WORLD_SIZE"])
            
            # Always print DDP info for all ranks - use print to bypass logging level restriction
            import sys
            print(f'[Rank {RANK}] DDP info: RANK {RANK}, LOCAL_RANK {LOCAL_RANK}, WORLD_SIZE {actual_world_size}, '
                  f'DEVICE {self.device}, CUDA_VISIBLE_DEVICES={os.environ.get("CUDA_VISIBLE_DEVICES", "not set")}', 
                  file=sys.stderr, flush=True)
            LOGGER.info(f'DDP info: RANK {RANK}, LOCAL_RANK {LOCAL_RANK}, WORLD_SIZE {actual_world_size}, DEVICE {self.device}')
            
            # Jittor uses MPI for distributed training, no need for explicit init_process_group
            # The distributed context is automatically set up by MPI
            # Each process will automatically get a different subset of data

    def _setup_train(self, world_size):
        """Builds dataloaders and optimizer on correct rank process."""
        self.run_callbacks("on_pretrain_routine_start")
        ckpt = self.setup_model()
        self.set_model_attributes()
        
        # Setup AMP (Automatic Mixed Precision)
        self._setup_amp(world_size)
        
        # Setup DDP (Distributed Data Parallel) if needed
        if world_size > 1:
            self._setup_model_ddp(world_size)

        # Ensure model parameters are identical across MPI ranks after initialization/loading.
        if jt.mpi and int(jt.world_size) > 1:
            self.model.mpi_param_broadcast(root=0)

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
        # Jittor MPI expects the Dataset batch size to be the global batch size
        # (sum across all ranks), not per-rank batch size.
        if world_size > 1:
            if self.batch_size < world_size:
                raise ValueError(
                    f"Batch size {self.batch_size} must be >= world_size {world_size} for Jittor MPI training."
                )
            if self.batch_size % world_size != 0 and RANK in {-1, 0}:
                per_rank = math.ceil(self.batch_size / world_size)
                LOGGER.warning(
                    f"WARNING ⚠️ Jittor MPI uses global batch size; batch={self.batch_size} is not divisible by "
                    f"world_size={world_size}. Per-rank batch will be {per_rank}. "
                    "For exact parity with single-GPU, use a batch divisible by world_size."
                )
            batch_size = self.batch_size
        else:
            batch_size = self.batch_size // max(world_size, 1)
        self.train_loader = self.get_dataloader(self.trainset, batch_size=batch_size, rank=LOCAL_RANK, mode="train")
        if RANK in {-1, 0}:
            self.test_loader = self.get_dataloader(
                self.testset, batch_size=batch_size if self.args.task == "obb" else batch_size * 2, rank=-1, mode="val"
            )
            self.validator = self.get_validator()
            metric_keys = self.validator.metrics.keys + self.label_loss_items(prefix="val")
            self.metrics = dict(zip(metric_keys, [0] * len(metric_keys)))
            self.ema = ModelEMA(self.model)
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
        self.run_callbacks("on_pretrain_routine_end")
    
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
        # AMP is temporarily disabled.
        # self.amp = bool(getattr(self.args, "amp", False))
        requested_amp = bool(getattr(self.args, "amp", False))
        if requested_amp:
            raise NotImplementedError("AMP is temporarily disabled.")
        self.amp = False
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
            barrier_dir = self.save_dir / "_ddp_barrier"
            if barrier_dir.exists():
                shutil.rmtree(barrier_dir, ignore_errors=True)

        nb = len(self.train_loader)  # number of batches per process
        if RANK >= 0:
            LOGGER.info(f"Rank {RANK}: Number of batches per epoch = {nb} "
                       f"(dataset size: {len(self.trainset)}, batch_size: {self.batch_size // max(world_size, 1)})")
        nw = max(round(self.args.warmup_epochs * nb), 100) if self.args.warmup_epochs > 0 else -1  # warmup iterations
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
        if self.args.close_mosaic:
            base_idx = (self.epochs - self.args.close_mosaic) * nb
            self.plot_idx.extend([base_idx, base_idx + 1, base_idx + 2])
        epoch = self.start_epoch
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
            pbar = enumerate(self.train_loader)
            # Update dataloader attributes (optional)
            if epoch == (self.epochs - self.args.close_mosaic):
                self._close_dataloader_mosaic()
                self.train_loader.reset()

            if RANK in {-1, 0}:
                LOGGER.info(self.progress_string())
                pbar = TQDM(enumerate(self.train_loader), total=nb)
            self.tloss = None
            for i, batch in pbar:
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

                # Forward (AMP disabled)
                # with autocast(self.amp):
                batch = self.preprocess_batch(batch)
                self.loss, self.loss_items = self.model(batch)
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
                        if RANK != -1:  # if DDP training
                            self.stop = self._broadcast_object(self.stop, src=0)
                        if self.stop:  # training time exceeded
                            break

                # Log
                if RANK in {-1, 0}:
                    loss_length = self.tloss.shape[0] if len(self.tloss.shape) else 1
                    memory_str = self._get_memory_str()  # Get memory string or empty
                    format_str = "%11s" * (1 + (1 if memory_str else 0)) + "%11.4g" * (2 + loss_length)
                    log_items = [f"{epoch + 1}/{self.epochs}"]
                    if memory_str:
                        log_items.append(memory_str)
                    log_items.extend([
                        *(self.tloss if loss_length > 1 else jt.unsqueeze(self.tloss, 0)),  # losses
                        batch["cls"][0].shape[0],  # batch size, i.e. 8
                        batch["img"][0].shape[-1],  # imgsz, i.e 640
                    ])
                    pbar.set_description(format_str % tuple(log_items))
                    self.run_callbacks("on_batch_end")
                    if self.args.plots and ni in self.plot_idx:
                        self.plot_training_samples(batch, ni)

                self.run_callbacks("on_train_batch_end")

            self.lr = {f"lr/pg{ir}": x["lr"] for ir, x in enumerate(self.optimizer.param_groups)}  # for loggers
            self.run_callbacks("on_train_epoch_end")
            if RANK != -1:
                # Keep all ranks in sync before rank0-only work.
                self._ddp_barrier(f"epoch_{epoch}_pre")
            if RANK in {-1, 0}:
                final_epoch = epoch + 1 >= self.epochs
                self.ema.update_attr(self.model, include=["yaml", "nc", "args", "names", "stride", "class_weights"])

                # Validation
                if self.args.val or final_epoch or self.stopper.possible_stop or self.stop:
                    self._clear_memory(threshold=0.5)  # prevent VRAM spike
                    self.metrics, self.fitness = self.validate()
                self.save_metrics(metrics={**self.label_loss_items(self.tloss), **self.metrics, **self.lr})
                self.stop |= self.stopper(epoch + 1, self.fitness) or final_epoch
                if self.args.time:
                    self.stop |= (time.time() - self.train_time_start) > (self.args.time * 3600)

                # Save model
                if self.args.save or final_epoch:
                    self.save_model()
                    self.run_callbacks("on_model_save")
            if RANK != -1:
                # Wait for rank0 to finish validation/saving before next epoch.
                self._ddp_barrier(f"epoch_{epoch}_post")

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
            if RANK != -1:  # if DDP training
                self.stop = self._broadcast_object(self.stop, src=0)
            if self.stop:
                break  # must break all DDP ranks
            epoch += 1

        if RANK in {-1, 0}:
            # Do final val with best.pkl
            seconds = time.time() - (self.train_time_start or time.time())
            LOGGER.info(f"\n{epoch - self.start_epoch + 1} epochs completed in {seconds / 3600:.3f} hours.")
            if getattr(self.args, "final_eval", True):
                self.final_eval()
            if self.args.plots:
                self.plot_metrics()
            self.run_callbacks("on_train_end")
        self._clear_memory()
        self.run_callbacks("teardown")

    def auto_batch(self, max_num_obj=0):
        """Calculate optimal batch size based on model and device memory constraints."""
        return check_train_batch_size(
            model=self.model,
            imgsz=self.args.imgsz,
            amp=self.amp,
            batch=self.batch_size,
            max_num_obj=max_num_obj,
        )  # returns batch size

    def _get_memory(self, fraction=False):
        """Get accelerator memory utilization in GB or as a fraction of total memory."""
        device_str = str(self.device).lower()
        
        # Handle non-CUDA devices
        if "mps" in device_str:
            return __import__("psutil").virtual_memory().percent / 100 if fraction else 0.0
        
        if "cpu" in device_str:
            return 0.0
        
        # Handle CUDA devices - Jittor has no direct memory API, return None to indicate unavailable
        return None

    def _get_memory_str(self):
        """Get memory string for display, returns empty string if unavailable."""
        memory = self._get_memory()
        return f"{memory:.3g}G" if memory is not None and memory > 0 else ""

    def _model_train(self):
        """Set model in training mode."""
        self.model.train()
        # Freeze BN stat for frozen layers
        for n, m in self.model.named_modules():
            if any(filter(lambda f: f in n, self.freeze_layer_names)) and isinstance(m, nn.BatchNorm2d):
                m.eval()

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

        def _half_state_dict(model):
            """Create a portable FP16 state dict without mutating the model."""
            from nkyolo.utils.jittor_utils import state_dict_to_numpy

            return state_dict_to_numpy(model.state_dict(), fp16=True)
        
        # Prepare checkpoint data
        ema_model = self.ema.ema if self.ema and self.ema.ema is not None else self.model
        ema_updates = self.ema.updates if self.ema and self.ema.ema is not None else 0
        checkpoint_data = {
            "epoch": self.epoch,
            "best_fitness": self.best_fitness,
            "model": None,  # resume and final checkpoints derive from EMA
            "ema": _half_state_dict(ema_model),
            "updates": ema_updates,
            "optimizer": convert_optimizer_state_dict_to_fp16(safe_deepcopy_jittor(self.optimizer.state_dict())),
            "train_args": vars(self.args),  # save as dict
            "train_metrics": {**self.metrics, **{"fitness": self.fitness}},
            "train_results": self.read_results_csv(),
            "date": datetime.now().isoformat(),
            "version": __version__,
            "license": "AGPL-3.0 (https://ultralytics.com/license)",
            "docs": "https://docs.ultralytics.com",
            "model_yaml": ema_model.yaml,  # save model config for model reconstruction
            "names": self.model.names,
            "nc": self.model.nc,
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
        self.model = self.get_model(cfg=cfg, weights=weights, verbose=RANK == -1)  # calls Model(cfg, weights)
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
            self.ema.update(self.model)

    def preprocess_batch(self, batch):
        """Allows custom preprocessing model inputs and ground truths depending on task type."""
        return batch

    def validate(self):
        """
        Runs validation on test set using self.validator.

        The returned dict is expected to contain "fitness" key.
        """
        metrics = self.validator(self)
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

    def final_eval(self):
        """Performs final evaluation and validation for object detection YOLO model."""
        ckpt = {}
        for f in self.last, self.best:
            if f.exists():
                if f is self.last:
                    ckpt = strip_optimizer(f)
                elif f is self.best:
                    k = "train_results"  # update best.pkl train_metrics from last.pkl
                    strip_optimizer(f, updates={k: ckpt[k]} if k in ckpt else None)
                    LOGGER.info(f"\nValidating {f}...")
                    self.validator.args.plots = self.args.plots
                    self.metrics = self.validator(model=f)
                    self.metrics.pop("fitness", None)
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
        if self.ema and self.ema.ema is not None and ckpt.get("ema"):
            self.ema.ema.load_state_dict(state_dict_to_jittor(ckpt["ema"]))  # EMA
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
        g = [], [], []  # optimizer parameter groups
        bn = tuple(v for k, v in nn.__dict__.items() if "Norm" in k)  # normalization layers, i.e. BatchNorm2d()
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

        for module_name, module in model.named_modules():
            for param_name, param in module.named_parameters(recurse=False):
                fullname = f"{module_name}.{param_name}" if module_name else param_name
                if "running_mean" in fullname or "running_var" in fullname or "num_batches_tracked" in fullname:
                    continue
                if isinstance(param, jt.Var) and param.is_stop_grad():
                    continue
                if "bias" in fullname:  # bias (no decay)
                    g[2].append(param)
                elif isinstance(module, bn):  # weight (no decay)
                    g[1].append(param)
                else:  # weight (with decay)
                    g[0].append(param)

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
