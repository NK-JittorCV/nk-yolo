#!/usr/bin/env python3
import argparse
import contextlib
import gc
import importlib.util
import json
import math
import os
import statistics
import time
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Sequence, Tuple


_JT_DISTRIBUTED_AVAILABLE = importlib.util.find_spec("jittor.distributed") is not None

@dataclass
class OpSpec:
    name: str
    make_torch: Callable[[], Any]
    make_jittor: Callable[[], Any]
    make_inputs: Callable[[str, Any, Any], List[Any]]


@dataclass
class BenchResult:
    framework: str
    op: str
    mode: str
    dtype: str
    rank: int
    world_size: int
    fwd_ms_avg: float
    fwd_ms_p50: float
    fwd_ms_p95: float
    bwd_ms_avg: Optional[float]
    bwd_ms_p50: Optional[float]
    bwd_ms_p95: Optional[float]
    peak_mem_mb: Optional[float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare Jittor and PyTorch ops on GPU (latency & memory)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--gpu", type=str, default="0", help="GPU index for CUDA_VISIBLE_DEVICES (ignored in --ddp)")
    parser.add_argument("--dtype", type=str, default="fp32", choices=["fp32", "fp16"], help="dtype for inputs and modules")
    parser.add_argument("--batch", type=int, default=1, help="batch size")
    parser.add_argument("--imgsz", type=int, default=224, help="image size H=W for conv/pool/bn/relu")
    parser.add_argument("--warmup", type=int, default=10, help="warmup iterations")
    parser.add_argument("--iters", type=int, default=50, help="benchmark iterations")
    parser.add_argument("--ops", type=str, default="all", help="comma-separated ops to run or 'all'")
    parser.add_argument("--list-ops", action="store_true", help="list available ops and exit")
    parser.add_argument("--output", type=str, default="", help="optional output path for json")
    parser.add_argument("--framework", type=str, default="both", choices=["torch", "jittor", "both"], help="which framework(s) to run")
    parser.add_argument("--ddp", action="store_true", help="enable DDP (torchrun/mpirun required)")
    parser.add_argument("--ddp-backend", type=str, default="nccl", help="torch DDP backend")
    return parser.parse_args()


def mean_p50_p95(values: Sequence[float]) -> Tuple[float, float, float]:
    if not values:
        return 0.0, 0.0, 0.0
    values_sorted = sorted(values)
    mean = float(statistics.mean(values_sorted))
    p50 = float(statistics.median(values_sorted))
    p95_index = max(0, int(math.ceil(0.95 * len(values_sorted))) - 1)
    p95 = float(values_sorted[p95_index])
    return mean, p50, p95


def build_ops(args, torch_nn, jt_nn, torch, jt) -> List[OpSpec]:
    batch = args.batch
    imgsz = args.imgsz

    conv_in = 3
    conv_out = 16
    linear_in = 1024
    linear_out = 1024
    matmul_m = 1024
    matmul_k = 1024
    matmul_n = 1024

    def make_tensor(framework: str, shape: Tuple[int, ...], torch_dtype, jt_dtype):
        if framework == "torch":
            return torch.randn(*shape, device="cuda", dtype=torch_dtype)
        x = jt.randn(shape, dtype=jt_dtype)
        return x.cuda()

    def conv_inputs(framework: str, torch_dtype, jt_dtype) -> List[Any]:
        shape = (batch, conv_in, imgsz, imgsz)
        return [make_tensor(framework, shape, torch_dtype, jt_dtype)]

    def bn_inputs(framework: str, torch_dtype, jt_dtype) -> List[Any]:
        shape = (batch, conv_out, imgsz, imgsz)
        return [make_tensor(framework, shape, torch_dtype, jt_dtype)]

    def relu_inputs(framework: str, torch_dtype, jt_dtype) -> List[Any]:
        shape = (batch, conv_out, imgsz, imgsz)
        return [make_tensor(framework, shape, torch_dtype, jt_dtype)]

    def pool_inputs(framework: str, torch_dtype, jt_dtype) -> List[Any]:
        shape = (batch, conv_out, imgsz, imgsz)
        return [make_tensor(framework, shape, torch_dtype, jt_dtype)]

    def linear_inputs(framework: str, torch_dtype, jt_dtype) -> List[Any]:
        shape = (batch, linear_in)
        return [make_tensor(framework, shape, torch_dtype, jt_dtype)]

    def layernorm_inputs(framework: str, torch_dtype, jt_dtype) -> List[Any]:
        shape = (batch, linear_in)
        return [make_tensor(framework, shape, torch_dtype, jt_dtype)]

    def softmax_inputs(framework: str, torch_dtype, jt_dtype) -> List[Any]:
        shape = (batch, linear_in)
        return [make_tensor(framework, shape, torch_dtype, jt_dtype)]

    def matmul_inputs(framework: str, torch_dtype, jt_dtype) -> List[Any]:
        a = make_tensor(framework, (batch, matmul_m, matmul_k), torch_dtype, jt_dtype)
        b = make_tensor(framework, (batch, matmul_k, matmul_n), torch_dtype, jt_dtype)
        return [a, b]

    return [
        OpSpec(
            name="conv2d",
            make_torch=lambda: torch_nn.Conv2d(conv_in, conv_out, 3, 1, 1),
            make_jittor=lambda: jt_nn.Conv2d(conv_in, conv_out, 3, 1, 1),
            make_inputs=conv_inputs,
        ),
        OpSpec(
            name="batchnorm2d",
            make_torch=lambda: torch_nn.BatchNorm2d(conv_out),
            make_jittor=lambda: jt_nn.BatchNorm(conv_out),
            make_inputs=bn_inputs,
        ),
        OpSpec(
            name="relu",
            make_torch=lambda: torch_nn.ReLU(),
            make_jittor=lambda: jt_nn.ReLU(),
            make_inputs=relu_inputs,
        ),
        OpSpec(
            name="maxpool2d",
            make_torch=lambda: torch_nn.MaxPool2d(kernel_size=2, stride=2),
            make_jittor=lambda: jt_nn.MaxPool2d(kernel_size=2, stride=2),
            make_inputs=pool_inputs,
        ),
        OpSpec(
            name="avgpool2d",
            make_torch=lambda: torch_nn.AvgPool2d(kernel_size=2, stride=2),
            make_jittor=lambda: jt_nn.AvgPool2d(kernel_size=2, stride=2),
            make_inputs=pool_inputs,
        ),
        OpSpec(
            name="linear",
            make_torch=lambda: torch_nn.Linear(linear_in, linear_out),
            make_jittor=lambda: jt_nn.Linear(linear_in, linear_out),
            make_inputs=linear_inputs,
        ),
        OpSpec(
            name="layernorm",
            make_torch=lambda: torch_nn.LayerNorm(linear_in),
            make_jittor=lambda: jt_nn.LayerNorm(linear_in),
            make_inputs=layernorm_inputs,
        ),
        OpSpec(
            name="softmax",
            make_torch=lambda: torch_nn.Softmax(dim=-1),
            make_jittor=lambda: jt_nn.Softmax(dim=-1),
            make_inputs=softmax_inputs,
        ),
        OpSpec(
            name="matmul",
            make_torch=lambda: torch.matmul,
            make_jittor=lambda: lambda a, b: a @ b,
            make_inputs=matmul_inputs,
        ),
    ]


def prepare_module(op, is_fp16: bool, framework: str, torch, jt):
    if framework == "torch":
        if isinstance(op, torch.nn.Module):
            op = op.to("cuda")
            if is_fp16:
                op = op.half()
        return op
    if isinstance(op, jt.nn.Module):
        op = op.cuda()
        if is_fp16:
            op = op.half()
    return op


def torch_sync():
    import torch
    torch.cuda.synchronize()


def jt_sync(jt):
    jt.sync_all(True)


def enable_grad(framework: str, inputs: List[Any]):
    if framework == "torch":
        for x in inputs:
            x.requires_grad_(True)
    else:
        for x in inputs:
            x.start_grad()


def zero_grads(framework: str, op, inputs: List[Any], torch=None, optimizer=None):
    if framework == "torch":
        if isinstance(op, torch.nn.Module):
            op.zero_grad(set_to_none=True)
        for x in inputs:
            x.grad = None
    else:
        if optimizer is not None:
            optimizer.zero_grad()


def build_jittor_optimizer(op_inst, jt):
    if not isinstance(op_inst, jt.nn.Module):
        return None
    params = list(op_inst.parameters())
    if not params:
        return None
    return jt.optim.SGD(params, lr=0.0)


def has_trainable_torch_params(op_inst, torch) -> bool:
    if not isinstance(op_inst, torch.nn.Module):
        return False
    for p in op_inst.parameters():
        if p.requires_grad:
            return True
    return False


def has_trainable_jittor_params(op_inst, jt) -> bool:
    if not isinstance(op_inst, jt.nn.Module):
        return False
    for p in op_inst.parameters():
        if isinstance(p, jt.Var) and not p.is_stop_grad():
            return True
    return False


def compute_loss(output) -> Any:
    if isinstance(output, (list, tuple)):
        return sum(y.sum() for y in output)
    return output.sum()


def parse_env_int(name: str) -> Optional[int]:
    value = os.environ.get(name)
    if value is None or not value.isdigit():
        return None
    return int(value)


def setup_torch_ddp(torch, backend: str) -> Optional[Tuple[int, int, int]]:
    if not torch.distributed.is_available():
        return None
    rank = parse_env_int("RANK")
    world_size = parse_env_int("WORLD_SIZE")
    local_rank = parse_env_int("LOCAL_RANK")
    if rank is None or world_size is None or local_rank is None:
        return None
    torch.cuda.set_device(local_rank)
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend=backend, init_method="env://")
    return rank, world_size, local_rank


def setup_jittor_ddp(jt) -> Optional[Tuple[int, int]]:
    if not jt.mpi:
        return None
    return int(jt.rank), int(jt.world_size)


def jittor_peak_memory_mb(jt, run_fn: Callable[[], None]) -> Optional[float]:
    with jt.flag_scope(trace_py_var=3, profile_memory_enable=1):
        run_fn()
        jt_sync(jt)
        info = jt.get_max_memory_info()

    div1 = "[!@#div1!@#]"
    parts = info.split(div1)
    if not parts:
        return None
    max_str = parts[0]
    if not max_str.isdigit():
        return None
    max_bytes = int(max_str)
    if max_bytes <= 0:
        return 0.0
    return max_bytes / (1024 * 1024)


def torch_peak_memory_mb(run_fn: Callable[[], None]) -> float:
    import torch
    torch.cuda.reset_peak_memory_stats()
    run_fn()
    torch_sync()
    return torch.cuda.max_memory_allocated() / (1024 * 1024)


def bench_op(
    framework: str,
    op: OpSpec,
    args,
    torch,
    jt,
    torch_dtype,
    jt_dtype,
    dtype_label: str,
    is_fp16: bool,
    ddp_enabled: bool,
    ddp_rank: int,
    ddp_world_size: int,
    ddp_local_rank: int,
) -> List[BenchResult]:
    results: List[BenchResult] = []

    op_inst = op.make_torch() if framework == "torch" else op.make_jittor()
    op_inst = prepare_module(op_inst, is_fp16, framework, torch, jt)
    if ddp_enabled:
        if framework == "torch":
            if has_trainable_torch_params(op_inst, torch):
                op_inst = torch.nn.parallel.DistributedDataParallel(
                    op_inst, device_ids=[ddp_local_rank], output_device=ddp_local_rank, broadcast_buffers=False
                )
        else:
            if has_trainable_jittor_params(op_inst, jt) and _JT_DISTRIBUTED_AVAILABLE:
                from jittor import distributed as jt_dist
                op_inst = jt_dist.ParallelModel(op_inst)

    inputs = op.make_inputs(framework, torch_dtype, jt_dtype)
    jt_optimizer = build_jittor_optimizer(op_inst, jt) if framework == "jittor" else None

    for mode in ["inference", "train"]:
        if framework == "torch":
            if isinstance(op_inst, torch.nn.Module):
                op_inst.train(mode == "train")
        else:
            if isinstance(op_inst, jt.nn.Module):
                if mode == "train":
                    op_inst.train()
                else:
                    op_inst.eval()

        def run_once():
            if mode == "train":
                enable_grad(framework, inputs)
            out = op_inst(*inputs)
            if mode == "train":
                loss = compute_loss(out)
                if framework == "torch":
                    loss.backward()
                else:
                    if jt_optimizer is not None:
                        jt_optimizer.backward(loss)
                    else:
                        jt.grad(loss, inputs)
            return out

        # Warmup
        for _ in range(args.warmup):
            if framework == "torch":
                ctx = torch.enable_grad() if mode == "train" else torch.no_grad()
            else:
                ctx = contextlib.nullcontext()
            with ctx:
                run_once()
            if framework == "torch":
                torch_sync()
            else:
                jt_sync(jt)
            zero_grads(framework, op_inst, inputs, torch=torch, optimizer=jt_optimizer)

        # Timing
        fwd_times: List[float] = []
        bwd_times: List[float] = []

        for _ in range(args.iters):
            if framework == "torch":
                ctx = torch.enable_grad() if mode == "train" else torch.no_grad()
            else:
                ctx = contextlib.nullcontext()

            with ctx:
                if mode == "train":
                    enable_grad(framework, inputs)
                if framework == "torch":
                    torch_sync()
                else:
                    jt_sync(jt)
                t0 = time.perf_counter()
                out = op_inst(*inputs)
                if framework == "torch":
                    torch_sync()
                else:
                    jt_sync(jt)
                t1 = time.perf_counter()
                fwd_times.append((t1 - t0) * 1000)

                if mode == "train":
                    loss = compute_loss(out)
                    if framework == "torch":
                        loss.backward()
                        torch_sync()
                    else:
                        if jt_optimizer is not None:
                            jt_optimizer.backward(loss)
                        else:
                            jt.grad(loss, inputs)
                        jt_sync(jt)
                    t2 = time.perf_counter()
                    bwd_times.append((t2 - t1) * 1000)

                zero_grads(framework, op_inst, inputs, torch=torch, optimizer=jt_optimizer)

        fwd_avg, fwd_p50, fwd_p95 = mean_p50_p95(fwd_times)
        if bwd_times:
            bwd_avg, bwd_p50, bwd_p95 = mean_p50_p95(bwd_times)
        else:
            bwd_avg = bwd_p50 = bwd_p95 = None

        # Memory (use official Jittor memory profiler)
        if framework == "torch":
            peak_mem_mb = torch_peak_memory_mb(run_once)
        else:
            peak_mem_mb = jittor_peak_memory_mb(jt, run_once)

        results.append(
            BenchResult(
                framework=framework,
                op=op.name,
                mode=mode,
                dtype=dtype_label,
                rank=ddp_rank,
                world_size=ddp_world_size,
                fwd_ms_avg=fwd_avg,
                fwd_ms_p50=fwd_p50,
                fwd_ms_p95=fwd_p95,
                bwd_ms_avg=bwd_avg,
                bwd_ms_p50=bwd_p50,
                bwd_ms_p95=bwd_p95,
                peak_mem_mb=peak_mem_mb,
            )
        )

        gc.collect()
        if framework == "torch":
            torch.cuda.empty_cache()
        else:
            jt.gc()

    return results


def format_table(rows: List[BenchResult]) -> str:
    headers = [
        "framework",
        "op",
        "mode",
        "dtype",
        "rank/world",
        "fwd_ms(avg/p50/p95)",
        "bwd_ms(avg/p50/p95)",
        "peak_mem_mb",
    ]

    table_rows = []
    for r in rows:
        fwd = f"{r.fwd_ms_avg:.3f}/{r.fwd_ms_p50:.3f}/{r.fwd_ms_p95:.3f}"
        if r.bwd_ms_avg is None:
            bwd = "-"
        else:
            bwd = f"{r.bwd_ms_avg:.3f}/{r.bwd_ms_p50:.3f}/{r.bwd_ms_p95:.3f}"
        peak = "-" if r.peak_mem_mb is None else f"{r.peak_mem_mb:.2f}"
        table_rows.append([r.framework, r.op, r.mode, r.dtype, f"{r.rank}/{r.world_size}", fwd, bwd, peak])

    cols = list(zip(headers, *table_rows)) if table_rows else [headers]
    widths = [max(len(str(x)) for x in col) for col in cols]

    def fmt_row(row):
        return " | ".join(str(v).ljust(widths[i]) for i, v in enumerate(row))

    lines = [fmt_row(headers), "-+-".join("-" * w for w in widths)]
    for row in table_rows:
        lines.append(fmt_row(row))
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    if not args.ddp:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    import torch
    import jittor as jt
    from torch import nn as torch_nn
    from jittor import nn as jt_nn

    if args.ddp and args.framework == "both":
        print("DDP mode requires running a single framework. Use --framework torch or --framework jittor.")
        return 1

    if not torch.cuda.is_available():
        print("Torch CUDA unavailable. This script only runs on GPU.")
        return 1
    if not jt.has_cuda:
        print("Jittor CUDA unavailable. This script only runs on GPU.")
        return 1

    jt.flags.use_cuda = 1

    _torch_ddp = False
    _jittor_ddp = False
    torch_rank = 0
    torch_world = 1
    torch_local_rank = 0
    jittor_rank = 0
    jittor_world = 1

    if args.ddp and args.framework in {"torch"}:
        ddp_info = setup_torch_ddp(torch, args.ddp_backend)
        if ddp_info is None:
            print("Torch DDP requires torchrun with RANK/WORLD_SIZE/LOCAL_RANK envs.")
            return 1
        torch_rank, torch_world, torch_local_rank = ddp_info
        _torch_ddp = torch_world > 1

    if args.ddp and args.framework in {"jittor"}:
        ddp_info = setup_jittor_ddp(jt)
        if ddp_info is None:
            print("Jittor DDP requires mpirun (MPI environment).")
            return 1
        jittor_rank, jittor_world = ddp_info
        _jittor_ddp = jittor_world > 1

    ops = build_ops(args, torch_nn, jt_nn, torch, jt)
    if args.list_ops:
        print("Available ops:")
        for op in ops:
            print(f"- {op.name}")
        return 0

    if args.ops != "all":
        requested = {o.strip() for o in args.ops.split(",") if o.strip()}
        ops = [o for o in ops if o.name in requested]
        if not ops:
            print("No matching ops found. Use --list-ops to see available ops.")
            return 1

    is_fp16 = args.dtype == "fp16"
    dtype_label = "fp16" if is_fp16 else "fp32"
    torch_dtype = torch.float16 if is_fp16 else torch.float32
    jt_dtype = jt.float16 if is_fp16 else jt.float32

    results: List[BenchResult] = []
    if args.framework in {"torch", "both"}:
        for op in ops:
            results.extend(
                bench_op(
                    "torch",
                    op,
                    args,
                    torch,
                    jt,
                    torch_dtype,
                    jt_dtype,
                    dtype_label,
                    is_fp16,
                    _torch_ddp,
                    torch_rank,
                    torch_world,
                    torch_local_rank,
                )
            )
    if args.framework in {"jittor", "both"}:
        for op in ops:
            results.extend(
                bench_op(
                    "jittor",
                    op,
                    args,
                    torch,
                    jt,
                    torch_dtype,
                    jt_dtype,
                    dtype_label,
                    is_fp16,
                    _jittor_ddp,
                    jittor_rank,
                    jittor_world,
                    0,
                )
            )

    print(format_table(results))

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump([r.__dict__ for r in results], f, indent=2)
        print(f"Results saved to {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
