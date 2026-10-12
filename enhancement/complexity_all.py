
import torch
import torch.nn.functional as F
import time

# ===== 改这里：导入你的模型 =====
from sci_enhancer import SCIEnhancementNet


# ---------------- Params ----------------
def count_params(model):
    return sum(p.numel() for p in model.parameters()) / 1e6


# ---------------- GFLOPs (thop) ----------------
def compute_gflops(model, H=512, W=512):
    try:
        from thop import profile
    except:
        print("Please install thop: pip install thop")
        return "-"

    model = model.cuda().eval()
    x = torch.randn(1, 3, H, W).cuda()

    macs, _ = profile(model, inputs=(x,), verbose=False)
    gflops = (2 * macs) / 1e9
    return gflops


# ---------------- Latency + Memory ----------------
@torch.no_grad()
def benchmark(model, H=512, W=512, iters=100, warmup=20):
    model = model.cuda().eval()
    x = torch.randn(1, 3, H, W).cuda()

    # warmup
    for _ in range(warmup):
        _ = model(x)

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)

    times = []

    for _ in range(iters):
        starter.record()
        _ = model(x)
        ender.record()
        torch.cuda.synchronize()
        times.append(starter.elapsed_time(ender))

    avg_time = sum(times) / len(times)
    mem = torch.cuda.max_memory_allocated() / 1024**2

    return avg_time, mem


# ---------------- main ----------------
def main():
    print("GPU:", torch.cuda.get_device_name(0))

    # ===== 初始化模型 =====
    model = SCIEnhancementNet()

    # ===== 如果有权重，加载 =====
    ckpt = "checkpoints/sci_best.pth"
    try:
        sd = torch.load(ckpt, map_location="cpu")
        if "model" in sd:
            sd = sd["model"]
        sd = {k.replace("module.", ""): v for k, v in sd.items()}
        model.load_state_dict(sd, strict=False)
        print("Checkpoint loaded.")
    except:
        print("No checkpoint loaded.")

    # ===== Params =====
    params = count_params(model)

    # ===== GFLOPs =====
    gflops = compute_gflops(model)

    # ===== Latency & Memory =====
    time_ms, mem_mb = benchmark(model)

    print("\n========== RESULT ==========")
    print(f"Params:   {params:.2f} M")
    print(f"GFLOPs:   {gflops}")
    print(f"Time:     {time_ms:.2f} ms")
    print(f"Memory:   {mem_mb:.2f} MB")
    print("============================")


if __name__ == "__main__":
    main()

