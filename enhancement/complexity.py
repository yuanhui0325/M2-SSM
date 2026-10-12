import os
import sys
import time
import argparse
import torch
import torch.nn.functional as F

# 允许在 enhancement/ 目录直接运行
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

try:
    from sci_enhancer import SCIEnhancementNet
except ImportError:
    from enhancement.sci_enhancer import SCIEnhancementNet


def count_params_m(model):
    return sum(p.numel() for p in model.parameters()) / 1e6


@torch.no_grad()
def benchmark(model, H, W, device="cuda", iters=100, warmup=20, fp16=False):
    model.eval().to(device)

    x = torch.randn(1, 3, H, W, device=device)
    # 如果你推理时需要 pad 到偶数（stride=2），这里也保持一致
    pad_h = H % 2
    pad_w = W % 2
    if pad_h or pad_w:
        x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")

    # warmup
    for _ in range(warmup):
        if fp16:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                _ = model(x, temperature=0.01)[0]
        else:
            _ = model(x, temperature=0.01)[0]
    torch.cuda.synchronize()

    torch.cuda.reset_peak_memory_stats()

    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)
    times = []

    for _ in range(iters):
        starter.record()
        if fp16:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                _ = model(x, temperature=0.01)[0]
        else:
            _ = model(x, temperature=0.01)[0]
        ender.record()
        torch.cuda.synchronize()
        times.append(starter.elapsed_time(ender))  # ms

    avg_ms = sum(times) / len(times)
    peak_mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)

    return avg_ms, peak_mem_mb


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default="", help="checkpoint path (optional)")
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--fp16", action="store_true", help="use AMP fp16")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise RuntimeError("This benchmark is intended to run on CUDA GPU.")

    print("GPU:", torch.cuda.get_device_name(0))
    torch.backends.cudnn.benchmark = True

    model = SCIEnhancementNet()

    if args.ckpt and os.path.exists(args.ckpt):
        sd = torch.load(args.ckpt, map_location="cpu")
        # 兼容你训练保存的 dict 或直接 state_dict
        if isinstance(sd, dict) and "model" in sd:
            sd = sd["model"]
        # 去 module.
        new_sd = {k.replace("module.", ""): v for k, v in sd.items()}
        model.load_state_dict(new_sd, strict=True)
        print("Loaded ckpt:", args.ckpt)
    else:
        print("No checkpoint loaded (random weights).")

    params_m = count_params_m(model)
    print(f"Params: {params_m:.3f} M")

    # 你论文里建议至少测这几个分辨率
    sizes = [
        (512, 512),
        (1024, 1024),
        (1080, 1920),
    ]

    print("-" * 70)
    print(f"{'Resolution':>12} | {'Time (ms)':>10} | {'PeakMem (MB)':>12} | {'FP16':>5}")
    print("-" * 70)
    for H, W in sizes:
        t_ms, mem_mb = benchmark(
            model, H, W,
            device=device,
            iters=args.iters,
            warmup=args.warmup,
            fp16=args.fp16
        )
        print(f"{H}x{W: <6} | {t_ms:10.2f} | {mem_mb:12.1f} | {str(args.fp16):>5}")
    print("-" * 70)


if __name__ == "__main__":
    main()