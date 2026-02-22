# NK-YOLO 🚀 AGPL-3.0 License
# Refer to https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/dist.py

import os
import re
import shutil
import socket
import sys
import tempfile

from . import USER_CONFIG_DIR


def find_free_network_port() -> int:
    """
    Finds a free port on localhost.

    It is useful in single-node training when we don't want to connect to a real main node but have to set the
    `MASTER_PORT` environment variable.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]  # port


def parse_device_list(device) -> list:
    """Parse device input into a list of integer GPU ids."""
    if device is None:
        return []
    if isinstance(device, int):
        return [device]
    if isinstance(device, (list, tuple)):
        parsed = []
        for item in device:
            if isinstance(item, int):
                parsed.append(item)
            elif isinstance(item, str) and item.strip().isdigit():
                parsed.append(int(item.strip()))
        return parsed
    if not isinstance(device, str):
        return []

    device_str = device.strip().lower()
    if not device_str or device_str in {"cpu", "mps", "none", "null"}:
        return []
    device_str = device_str.replace("cuda:", "")
    for ch in "[]()'\"":
        device_str = device_str.replace(ch, "")
    parts = [p for p in re.split(r"[,\s]+", device_str) if p]
    return [int(p) for p in parts if p.isdigit()]


def get_visible_devices() -> list:
    """Parse CUDA_VISIBLE_DEVICES into a list of GPU ids."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not visible or visible == "-1":
        return []
    return parse_device_list(visible)


def resolve_device_id(device_list, local_rank, rank) -> int:
    """Resolve the GPU id for the current rank from a device list."""
    if device_list:
        index = local_rank if local_rank >= 0 else rank if rank >= 0 else 0
        if index < len(device_list):
            return device_list[index]
        return device_list[-1]
    if local_rank >= 0:
        return local_rank
    if rank >= 0:
        return rank
    return 0


def generate_ddp_file(trainer):
    """Generates a DDP file and returns its file name."""
    module, name = f"{trainer.__class__.__module__}.{trainer.__class__.__name__}".rsplit(".", 1)

    # Extract device list from trainer args for proper GPU assignment
    overrides_dict = vars(trainer.args).copy()
    
    # Parse device list from original device parameter
    # device can be "0,1", [0,1], or "cuda:0"
    original_device = overrides_dict.get('device', '')
    device_list = parse_device_list(original_device)
    
    # Store device list for use in generated script
    device_list_str = str(device_list)

    content = f"""
# NK-YOLO Multi-GPU training temp file (should be automatically deleted after use)
# This file is executed by mpirun, which sets MPI environment variables
# Each MPI process will use its own device based on LOCAL_RANK
import os

# Resolve rank/local-rank before importing Jittor to ensure CUDA_VISIBLE_DEVICES is set early.
_rank_env = os.environ.get("OMPI_COMM_WORLD_RANK") or os.environ.get("PMI_RANK") or os.environ.get("RANK") or "0"
_local_rank_env = (
    os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK")
    or os.environ.get("PMI_LOCAL_RANK")
    or os.environ.get("LOCAL_RANK")
    or "0"
)
_rank_id = int(_rank_env) if (_rank_env.isdigit() or (_rank_env.startswith("-") and _rank_env[1:].isdigit())) else 0
_local_rank_id = int(_local_rank_env) if (_local_rank_env.isdigit() or (_local_rank_env.startswith("-") and _local_rank_env[1:].isdigit())) else 0

# Set Jittor compile environment early to avoid multi-process cache races/segfaults
os.environ.setdefault("JIT_PARALLEL", "0")
os.environ.setdefault("JITTOR_COMPILE_THREADS", "1")
_cache_root = os.environ.get("JITTOR_CACHE_PATH") or os.path.expanduser("~/.cache/jittor")
_rank_cache = f"{{_cache_root}}_rank{{_rank_id}}"
os.environ["JITTOR_CACHE_PATH"] = _rank_cache
os.makedirs(_rank_cache, exist_ok=True)
os.makedirs(os.path.join(_rank_cache, "jit"), exist_ok=True)

overrides = {overrides_dict}
device_list = {device_list_str}

# In MPI, CUDA_VISIBLE_DEVICES should list all GPUs used by the job.
# LOCAL_RANK selects the device internally; do NOT narrow per rank.
device_value = overrides.get("device", "")
device_arg = str(device_value).lower().strip()
force_cpu = device_arg in {{"cpu", "mps"}} or os.environ.get("CUDA_VISIBLE_DEVICES") == "-1"
if force_cpu:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
elif device_list:
    existing = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not existing or existing == "-1":
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(x) for x in device_list)

if __name__ == "__main__":
    from {module} import {name}
    from nkyolo.utils import DEFAULT_CFG_DICT
    import jittor as jt
    
    # Disable Jittor's internal parallel compiler
    if _rank_id >= 0:
        jt.flags.use_parallel_op_compiler = 0
        if os.environ.get("JITTOR_CACHE_PATH"):
            jt.flags.cache_path = os.environ["JITTOR_CACHE_PATH"]

    cfg = DEFAULT_CFG_DICT.copy()
    cfg.update(save_dir='')   # handle the extra key 'save_dir'
    trainer = {name}(cfg=cfg, overrides=overrides)
    trainer.args.model = "{getattr(trainer.hub_session, 'model_url', trainer.args.model)}"
    results = trainer.train()
    # Ensure MPI ranks terminate cleanly after training/validation
    import sys as _sys, os as _os
    _sys.stdout.flush()
    _sys.stderr.flush()
    _os._exit(0)
"""
    (USER_CONFIG_DIR / "DDP").mkdir(exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix="_temp_",
        suffix=f"{id(trainer)}.py",
        mode="w+",
        encoding="utf-8",
        dir=USER_CONFIG_DIR / "DDP",
        delete=False,
    ) as file:
        file.write(content)
    return file.name


def generate_ddp_command(world_size, trainer):
    """Generates and returns command for Jittor distributed training using MPI."""
    import __main__  # noqa local import to avoid https://github.com/Lightning-AI/lightning/issues/15218
    import subprocess

    # Check if mpirun is available
    subprocess.run(["mpirun", "--version"], capture_output=True, check=True, timeout=5)
    if not trainer.resume:
        shutil.rmtree(trainer.save_dir)  # remove the save_dir
    file = generate_ddp_file(trainer)
    
    # Jittor uses MPI for distributed training
    # Set environment variables for Jittor distributed training
    port = find_free_network_port()
    
    # Use mpirun for Jittor distributed training
    # Format: mpirun -np <world_size> python <script>
    # MPI will automatically set RANK, LOCAL_RANK, and WORLD_SIZE environment variables
    # Use --allow-run-as-root if needed, and --bind-to none to avoid CPU binding issues
    cmd = ["mpirun"]
    if os.name != "nt" and os.geteuid() == 0:
        cmd.append("--allow-run-as-root")
    cmd += [
        "-np", str(world_size),
        "--bind-to", "none",  # Don't bind processes to specific CPUs
        sys.executable,
        file,
    ]
    
    # Set environment variables for Jittor distributed training
    env = os.environ.copy()
    env["MASTER_ADDR"] = "127.0.0.1"
    env["MASTER_PORT"] = str(port)
    # Mark that we're launching MPI to avoid recursive calls
    env["NKYOLO_MPI_LAUNCHED"] = "1"
    # Disable Jittor's parallel compilation to prevent segfaults in MPI environments
    # Multiple MPI processes compiling operators simultaneously causes memory corruption
    env["JIT_PARALLEL"] = "0"  # Disable Jittor's parallel JIT compilation
    env["JITTOR_COMPILE_THREADS"] = "1"  # Use single-threaded compilation
    # MPI will set these automatically:
    # - OMPI_COMM_WORLD_SIZE (OpenMPI) or PMI_SIZE (other MPI implementations)
    # - OMPI_COMM_WORLD_RANK (OpenMPI) or PMI_RANK (other MPI implementations)
    # - RANK and LOCAL_RANK will be set by our code based on MPI environment
    
    return cmd, file, env


def ddp_cleanup(trainer, file):
    """Delete temp file if created."""
    if f"{id(trainer)}.py" in file:  # if temp_file suffix in file
        os.remove(file)
