#import os
#import sys
#
## ==========================================
## 1. 核心修复：解决路径问题
## ==========================================
## 获取当前文件 (train.py) 的目录: .../fast-jpeg-main/enhancement
#current_dir = os.path.dirname(os.path.abspath(__file__))
## 获取上一级目录 (项目根目录): .../fast-jpeg-main
#parent_dir = os.path.dirname(current_dir)
## 将根目录加入 Python 搜索路径，这样才能识别 enhancement 包
#sys.path.append(parent_dir)
#
## ==========================================
## 2. 补全缺失的导入 (修复 NameError)
## ==========================================
#import torch
#import torch.optim as optim
#from torch.utils.data import DataLoader  # 修复 name 'DataLoader' is not defined
#from tqdm import tqdm  # 进度条库
#
## 现在可以正常导入同级模块了
#try:
#    # 这里的导入路径依然使用包名 enhancement.xxx
#    from enhancement.sci_enhancer import SCIEnhancementNet, SCILoss
#    from enhancement.dataset import SCIDataset
#except ImportError:
#    # 如果上面的失败，尝试直接导入 (备用方案)
#    from sci_enhancer import SCIEnhancementNet, SCILoss
#    from dataset import SCIDataset
#
## ==========================================
## 3. 辅助函数
## ==========================================
#def calculate_psnr(img1, img2):
#    """计算 PSNR"""
#    mse = torch.mean((img1 - img2) ** 2)
#    if mse == 0:
#        return 100
#    return 20 * torch.log10(1.0 / torch.sqrt(mse))
#
#def validate(model, val_loader, device):
#    """验证集评估"""
#    model.eval()
#    total_psnr = 0.0
#    count = 0
#    with torch.no_grad():
#        for lq, hq, _ in val_loader:
#            lq = lq.to(device)
#            hq = hq.to(device)
#            
#            # 验证时温度极低(0.01)，强制二值化
#            pred, _ = model(lq, temperature=0.01)
#            pred = torch.clamp(pred, 0, 1)
#            
#            total_psnr += calculate_psnr(pred, hq).item()
#            count += 1
#    return total_psnr / count if count > 0 else 0
#
## ==========================================
## 4. 训练主循环
## ==========================================
#def train():
#    # --- 参数配置 ---
#    BATCH_SIZE = 16       # 显存够大可以用 16 或 32，不够改回 4
#    NUM_EPOCHS = 100
#    LEARNING_RATE = 2e-4
#    
#    # 路径配置
#    TRAIN_DIR = "/data/lichaofei/data/SCICD-PC/train"
#    VAL_DIR = "/data/lichaofei/data/SCICD-PC/val"
#    
#    # 模型保存路径 (保存在当前 enhancement/checkpoints 下)
#    SAVE_DIR = os.path.join(current_dir, "checkpoints")
#    os.makedirs(SAVE_DIR, exist_ok=True)
#    
#    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#    print(f"🔧 使用设备: {device}")
#
#    # --- 数据加载 ---
#    if not os.path.exists(TRAIN_DIR):
#        print(f"❌ 错误: 找不到训练集路径 {TRAIN_DIR}")
#        return
#
#    print("正在加载数据 (使用 OpenCV 加速)...")
#    # num_workers=8 配合 OpenCV dataset，速度会非常快
#    train_dataset = SCIDataset(hq_dir=TRAIN_DIR, patch_size=128, jpeg_quality_range=(10, 40), is_train=True)
#    
#    train_loader = DataLoader(
#        train_dataset, 
#        batch_size=BATCH_SIZE, 
#        shuffle=True, 
#        num_workers=8,           # 多进程加载
#        pin_memory=True,         # 显存加速
#        persistent_workers=True  # 避免每个epoch重新创建进程
#    )
#
#    val_dataset = SCIDataset(hq_dir=VAL_DIR, patch_size=128, jpeg_quality_range=(20, 20), is_train=False)
#    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4)
#
#    # --- 模型初始化 ---
#    model = SCIEnhancementNet().to(device)
#    criterion = SCILoss().to(device)
#    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
#    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)
#    scaler = torch.cuda.amp.GradScaler() # 混合精度
#
#    best_psnr = 0.0
#    print("🚀 开始训练...")
#
#    # --- 循环 ---
#    for epoch in range(NUM_EPOCHS):
#        model.train()
#        
#        # 温度退火: 1.0 -> 0.01
#        if epoch < 10:
#            current_temp = 1.0
#        else:
#            current_temp = max(0.01, 1.0 * (0.9 ** (epoch - 10)))
#
#        epoch_loss = 0
#        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
#        
#        for lq, hq, gt_edge in loop:
#            lq = lq.to(device, non_blocking=True)
#            hq = hq.to(device, non_blocking=True)
#            gt_edge = gt_edge.to(device, non_blocking=True)
#
#            optimizer.zero_grad()
#
#            with torch.cuda.amp.autocast():
#                # 前向传播
#                pred_img, mask_pred = model(lq, temperature=current_temp)
#                # 计算损失
#                loss_dict = criterion(pred_img, hq, mask_pred, gt_edge)
#                total_loss = loss_dict[0] # 这里假设 dataset 返回的是 tuple (total, mse, polar, trans)
#
#            scaler.scale(total_loss).backward()
#            scaler.unscale_(optimizer)
#            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
#            scaler.step(optimizer)
#            scaler.update()
#
#            epoch_loss += total_loss.item()
#            loop.set_postfix(loss=total_loss.item(), temp=current_temp)
#
#        scheduler.step()
#        
#        # --- 验证 ---
#        avg_psnr = validate(model, val_loader, device)
#        print(f"🎉 Epoch {epoch+1} 完成 | Val PSNR: {avg_psnr:.2f} dB")
#
#        # 保存
#        torch.save(model.state_dict(), os.path.join(SAVE_DIR, "sci_latest.pth"))
#        if avg_psnr > best_psnr:
#            best_psnr = avg_psnr
#            torch.save(model.state_dict(), os.path.join(SAVE_DIR, "sci_best.pth"))
#            print(f"🌟 新纪录！最佳模型已保存 ({best_psnr:.2f} dB)")
#
#if __name__ == "__main__":
#    train()


#import os
#import sys
#import time
#
## 允许在 enhancement/ 目录直接 python train.py 运行
#current_dir = os.path.dirname(os.path.abspath(__file__))
#parent_dir = os.path.dirname(current_dir)
#sys.path.insert(0, parent_dir)
#
#import torch
#import torch.optim as optim
#from torch.utils.data import DataLoader
#from tqdm import tqdm
#
## 优先同目录导入（最稳）
#try:
#    from sci_enhancer import SCIEnhancementNet, SCILoss
#    from dataset import SCIDataset
#except ImportError:
#    from enhancement.sci_enhancer import SCIEnhancementNet, SCILoss
#    from enhancement.dataset import SCIDataset
#
#
#def calculate_psnr_batch(img1, img2, eps=1e-10):
#    mse = torch.mean((img1 - img2) ** 2, dim=(1, 2, 3))
#    psnr = 10.0 * torch.log10(1.0 / (mse + eps))
#    return psnr.mean()
#
#
#@torch.no_grad()
#def validate(model, val_loader, device, max_batches=200):
#    model.eval()
#    total = 0.0
#    n = 0
#    with torch.inference_mode():
#        for i, (lq, hq, _) in enumerate(val_loader):
#            if max_batches is not None and i >= max_batches:
#                break
#            lq = lq.to(device, non_blocking=True)
#            hq = hq.to(device, non_blocking=True)
##            pred, _ = model(lq, temperature=0.01)
#            pred, _, _ = model(lq, temperature=0.01)
#            pred = pred.clamp(0, 1)
#            total += calculate_psnr_batch(pred, hq).item()
#            n += 1
#    return total / max(n, 1)
#
#
#def train():
#    BATCH_SIZE = 8           # 先用 8，稳定后再尝试 16
#    NUM_EPOCHS = 100
#    LEARNING_RATE = 2e-4
#
#    TRAIN_DIR = "/data/lichaofei/data/SCICD-PC/train"
#    VAL_DIR = "/data/lichaofei/data/SCICD-PC/val"
#
#    SAVE_DIR = os.path.join(current_dir, "checkpoints")
#    os.makedirs(SAVE_DIR, exist_ok=True)
#
#    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#    use_amp = (device.type == "cuda")
#    print(f"device={device}, amp={use_amp}")
#    if use_amp:
#        print("gpu:", torch.cuda.get_device_name(0))
#        torch.backends.cudnn.benchmark = True
#        torch.backends.cuda.matmul.allow_tf32 = True
#        torch.backends.cudnn.allow_tf32 = True
#
#    if not os.path.exists(TRAIN_DIR):
#        raise FileNotFoundError(TRAIN_DIR)
#    if not os.path.exists(VAL_DIR):
#        raise FileNotFoundError(VAL_DIR)
#
#    train_dataset = SCIDataset(TRAIN_DIR, patch_size=128, jpeg_quality_range=(10, 40), is_train=True)
#    val_dataset = SCIDataset(VAL_DIR, patch_size=128, jpeg_quality_range=(20, 20), is_train=False)
#
#    num_workers = 8
#    train_loader = DataLoader(
#        train_dataset,
#        batch_size=BATCH_SIZE,
#        shuffle=True,
#        num_workers=num_workers,
#        pin_memory=use_amp,
#        persistent_workers=(num_workers > 0),
#        prefetch_factor=4 if num_workers > 0 else None,
#        drop_last=True,
#    )
#    val_loader = DataLoader(
#        val_dataset,
#        batch_size=1,
#        shuffle=False,
#        num_workers=2,
#        pin_memory=use_amp,
#    )
#
#    model = SCIEnhancementNet().to(device)
#    criterion = SCILoss().to(device)
#
#    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
#    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)
#    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
#
#    best_psnr = 0.0
#
#    for epoch in range(NUM_EPOCHS):
#        model.train()
#
#        if epoch < 10:
#            temp = 1.0
#        else:
#            temp = max(0.01, 1.0 * (0.9 ** (epoch - 10)))
#
#        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
#        epoch_loss = 0.0
#        t0 = time.time()
#
#        for lq, hq, gt_edge in pbar:
#            lq = lq.to(device, non_blocking=True)
#            hq = hq.to(device, non_blocking=True)
#            gt_edge = gt_edge.to(device, non_blocking=True)
#
#            optimizer.zero_grad(set_to_none=True)
#
#            with torch.cuda.amp.autocast(enabled=use_amp):
#                pred, mask_pred = model(lq, temperature=temp)
#                total_loss, loss_mse, loss_polar, loss_trans = criterion(pred, hq, mask_pred, gt_edge)
#                pred, mask_pred, mask_logits = model(lq, temperature=temp)
#
#                # transition warm-up：前10轮不加，后面逐渐加到 0.1
#                w_trans = 0.0 if epoch < 10 else min(0.1, (epoch - 10) / 20 * 0.1)
#
#                total_loss, loss_mse, loss_polar, loss_trans, loss_edge = criterion(
#                    pred, hq, mask_pred, mask_logits, gt_edge,
#                    w_polar=0.01, w_trans=w_trans, w_edge=0.1
#                )
#
#            if use_amp:
#                scaler.scale(total_loss).backward()
#                scaler.unscale_(optimizer)
#                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
#                scaler.step(optimizer)
#                scaler.update()
#            else:
#                total_loss.backward()
#                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
#                optimizer.step()
#
#            epoch_loss += total_loss.item()
#            pbar.set_postfix(loss=f"{total_loss.item():.4f}", temp=f"{temp:.3f}", lr=f"{optimizer.param_groups[0]['lr']:.2e}")
#
#        scheduler.step()
#        print(f"epoch_time_min={(time.time()-t0)/60:.2f}, train_loss={epoch_loss/max(len(train_loader),1):.4f}")
#
#        # 验证不用每个epoch全量（先快）
#        if (epoch + 1) % 5 == 0:
#            psnr = validate(model, val_loader, device, max_batches=200)
#            print(f"Val PSNR={psnr:.2f} dB")
#
#            if psnr > best_psnr:
#                best_psnr = psnr
#                torch.save(model.state_dict(), os.path.join(SAVE_DIR, "sci_best.pth"))
#                print(f"Saved best: {best_psnr:.2f} dB")
#
#        # 可续训 checkpoint
#        torch.save({
#            "epoch": epoch,
#            "model": model.state_dict(),
#            "optimizer": optimizer.state_dict(),
#            "scheduler": scheduler.state_dict(),
#            "scaler": scaler.state_dict(),
#            "best_psnr": best_psnr,
#        }, os.path.join(SAVE_DIR, "sci_latest.pth"))
#
#
#if __name__ == "__main__":
#    train()

import os
import sys
import time

# 允许在 enhancement/ 目录直接 python train.py 运行
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

# 优先同目录导入（最稳）
try:
    from sci_enhancer import SCIEnhancementNet, SCILoss
    from dataset import SCIDataset
except ImportError:
    from enhancement.sci_enhancer import SCIEnhancementNet, SCILoss
    from enhancement.dataset import SCIDataset


def calculate_psnr_batch(img1, img2, eps=1e-10):
    mse = torch.mean((img1 - img2) ** 2, dim=(1, 2, 3))
    psnr = 10.0 * torch.log10(1.0 / (mse + eps))
    return psnr.mean()


@torch.no_grad()
def validate(model, val_loader, device, max_batches=200):
    model.eval()
    total = 0.0
    n = 0
    with torch.inference_mode():
        for i, (lq, hq, _) in enumerate(val_loader):
            if max_batches is not None and i >= max_batches:
                break
            lq = lq.to(device, non_blocking=True)
            hq = hq.to(device, non_blocking=True)

            pred, _, _ = model(lq, temperature=0.01)
            pred = pred.clamp(0, 1)
            total += calculate_psnr_batch(pred, hq).item()
            n += 1
    return total / max(n, 1)


def train():
    BATCH_SIZE = 8
    NUM_EPOCHS = 100
    LEARNING_RATE = 2e-4

    TRAIN_DIR = "/data/lichaofei/data/SCICD-PC/train"
    VAL_DIR = "/data/lichaofei/data/SCICD-PC/val"

    SAVE_DIR = os.path.join(current_dir, "checkpoints")
    os.makedirs(SAVE_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = (device.type == "cuda")
    print(f"device={device}, amp={use_amp}")
    if use_amp:
        print("gpu:", torch.cuda.get_device_name(0))
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    if not os.path.exists(TRAIN_DIR):
        raise FileNotFoundError(TRAIN_DIR)
    if not os.path.exists(VAL_DIR):
        raise FileNotFoundError(VAL_DIR)

    train_dataset = SCIDataset(TRAIN_DIR, patch_size=128, jpeg_quality_range=(10, 40), is_train=True)
    val_dataset = SCIDataset(VAL_DIR, patch_size=128, jpeg_quality_range=(20, 20), is_train=False)

    num_workers = 8
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=use_amp,
        persistent_workers=(num_workers > 0),
        prefetch_factor=4 if num_workers > 0 else None,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        pin_memory=use_amp,
    )

    model = SCIEnhancementNet().to(device)
    criterion = SCILoss().to(device)

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)

    # AMP（兼容 torch.amp / torch.cuda.amp）
    try:
        from torch import amp
        scaler = amp.GradScaler("cuda", enabled=use_amp)
        autocast_ctx = lambda: amp.autocast("cuda", enabled=use_amp)
    except Exception:
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
        autocast_ctx = lambda: torch.cuda.amp.autocast(enabled=use_amp)

    best_psnr = 0.0

    for epoch in range(NUM_EPOCHS):
        model.train()

        # mask 温度退火
        if epoch < 10:
            temp = 1.0
        else:
            temp = max(0.01, 1.0 * (0.9 ** (epoch - 10)))

        # transition warm-up：前10轮不加，后面逐渐加到 0.1
        w_trans = 0.0 if epoch < 10 else min(0.1, (epoch - 10) / 20 * 0.1)

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
        epoch_loss = 0.0
        t0 = time.time()

        for lq, hq, gt_edge in pbar:
            lq = lq.to(device, non_blocking=True)
            hq = hq.to(device, non_blocking=True)
            gt_edge = gt_edge.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with autocast_ctx():
                pred, mask_pred, mask_logits = model(lq, temperature=temp)
                total_loss, loss_mse, loss_polar, loss_trans, loss_edge = criterion(
                    pred, hq,
                    mask_pred, mask_logits,
                    gt_edge,
                    w_polar=0.01,
                    w_trans=w_trans,
                    w_edge=0.1
                )

            if use_amp:
                scaler.scale(total_loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            epoch_loss += total_loss.item()
            pbar.set_postfix(
                loss=f"{total_loss.item():.4f}",
                mse=f"{loss_mse.item():.4f}",
                edge=f"{loss_edge.item():.4f}",
                trans=f"{loss_trans.item():.4f}",
                w_trans=f"{w_trans:.3f}",
                temp=f"{temp:.3f}",
                lr=f"{optimizer.param_groups[0]['lr']:.2e}",
            )

        scheduler.step()
        print(f"epoch_time_min={(time.time()-t0)/60:.2f}, train_loss={epoch_loss/max(len(train_loader),1):.4f}")

        # 验证
        if (epoch + 1) % 5 == 0:
            psnr = validate(model, val_loader, device, max_batches=200)
            print(f"Val PSNR={psnr:.2f} dB")

            if psnr > best_psnr:
                best_psnr = psnr
                torch.save(model.state_dict(), os.path.join(SAVE_DIR, "sci_best.pth"))
                print(f"Saved best: {best_psnr:.2f} dB")

        # 可续训 checkpoint
        torch.save({
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict() if use_amp else None,
            "best_psnr": best_psnr,
        }, os.path.join(SAVE_DIR, "sci_latest.pth"))


if __name__ == "__main__":
    train()