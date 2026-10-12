## test_enhancement.py 示例
#import torch
#from torchvision import transforms
#from PIL import Image
#from enhancement.sci_enhancer import SCIEnhancementNet
#
##def enhance_jpeg_output(jpeg_decoded_img_path):
##    # 1. 读取由 fast-jpeg-main 解码出来的图像
##    img = Image.open(jpeg_decoded_img_path).convert('RGB')
##    transform = transforms.ToTensor()
##    img_tensor = transform(img).unsqueeze(0) # (1, 3, H, W)
##    
##    # 2. 初始化网络 (推理时假设模型已经训练好并加载了权重)
##    model = SCIEnhancementNet()
##    model.eval()
##    
##    # 3. 前向传播 (推理时温度设置为极小值，确保掩码严格二值化)
##    with torch.no_grad():
##        enhanced_tensor, edge_mask = model(img_tensor, temperature=0.01)
##        
##    # 4. 保存增强后的图像
##    enhanced_img = transforms.ToPILImage()(enhanced_tensor.squeeze(0).clamp(0, 1))
##    enhanced_img.save("enhanced_output.png")
##    print("图像增强完成，长距离结构已修复，边缘已锐化！")
##
##if __name__ == "__main__":
##    # 假设 out_decoded.jpg 是由你的 decoder/decode.py 生成的
##    enhance_jpeg_output("out_decoded.jpg")
#
#def enhance_jpeg_output(jpeg_decoded_img_path, checkpoint_path):
#    # ... 读取图片部分 ...
#    
#    # 2. 初始化网络
#    model = SCIEnhancementNet()
#    
#    # 【关键修复】加载训练好的权重！
#    print(f"正在加载模型权重: {checkpoint_path}")
#    checkpoint = torch.load(checkpoint_path, map_location='cpu')
#    model.load_state_dict(checkpoint)
#    
#    model.eval()
#    # ... 后续代码 ...
#
#if __name__ == "__main__":
#    # 记得改成你实际训练出来的某个 epoch 的模型路径
#    ckpt_path = "checkpoints/sci_model_epoch_50.pth" 
#    enhance_jpeg_output("out_decoded.jpg", ckpt_path)



      
#
#import os
#import sys
#import cv2
#import numpy as np
#import torch
#from skimage.metrics import structural_similarity as ssim_func
#from tqdm import tqdm
#
## ==========================================
## 1. 路径修复 (确保能导入 enhancement)
## ==========================================
#current_dir = os.path.dirname(os.path.abspath(__file__))
## 如果脚本在 enhancement 文件夹内，父目录是根目录
## 如果脚本在 根目录，current_dir 就是根目录
#if current_dir.endswith('enhancement'):
#    sys.path.append(os.path.dirname(current_dir))
#else:
#    sys.path.append(current_dir)
#
#try:
#    from enhancement.sci_enhancer import SCIEnhancementNet
#except ImportError:
#    # 备用导入
#    from sci_enhancer import SCIEnhancementNet
#
## ==========================================
## 2. 指标计算工具
## ==========================================
#def calc_psnr(img1, img2):
#    """img1, img2: [0, 255] numpy arrays"""
#    img1 = img1.astype(np.float64)
#    img2 = img2.astype(np.float64)
#    mse = np.mean((img1 - img2) ** 2)
#    if mse == 0:
#        return 100
#    return 20 * np.log10(255.0 / np.sqrt(mse))
#
#def calc_ssim(img1, img2):
#    """img1, img2: [0, 255] numpy arrays, HWC"""
#    # win_size 必须小于图像最小边，设为 7 或 3 比较安全，multichannel=True处理彩色
#    return ssim_func(img1, img2, channel_axis=2, data_range=255)
#
## ==========================================
## 3. 核心测试逻辑
## ==========================================
#def run_test():
#    # ---------------- 配置区域 ----------------
#    # 测试集路径 (改成你的真实路径)
#    TEST_DIR = "/data/lichaofei/data/SCICD-PC/test"  
#    
#    # 模型权重路径 (改成你训练出来的 best.pth)
#    CHECKPOINT_PATH = "/home/chaofeili/fast-jpeg-main/enhancement/checkpoints/sci_best.pth" 
#    
#    # 要测试的 JPEG 质量列表
#    QUALITY_LIST = [10, 20, 30, 40]
#    
#    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#    # ----------------------------------------
#
#    # 1. 加载模型
#    print(f"Loading model from {CHECKPOINT_PATH} ...")
#    model = SCIEnhancementNet().to(DEVICE)
#    
#    # 加载权重 (处理可能的 DataParallel 这里的 module. 前缀)
#    state_dict = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
#    new_state_dict = {}
#    for k, v in state_dict.items():
#        name = k.replace('module.', '') # remove 'module.'
#        new_state_dict[name] = v
#    model.load_state_dict(new_state_dict)
#    
#    model.eval()
#
#    # 2. 获取图片列表
#    img_files = sorted([
#        os.path.join(TEST_DIR, f) for f in os.listdir(TEST_DIR) 
#        if f.lower().endswith(('.png', '.jpg', '.bmp', '.tif'))
#    ])
#    
#    if len(img_files) == 0:
#        print(f"❌ Error: No images found in {TEST_DIR}")
#        return
#
#    print(f"Found {len(img_files)} test images.")
#    print("="*90)
#    print(f"| Quality | Metric | {'JPEG (Baseline)':^15} | {'Ours (Mamba)':^15} | {'Boost':^10} |")
#    print("-" * 90)
#
#    # 3. 循环测试不同的 Q
#    for q in QUALITY_LIST:
#        
#        # 累加器
#        jpeg_psnr_sum, jpeg_ssim_sum = 0.0, 0.0
#        ours_psnr_sum, ours_ssim_sum = 0.0, 0.0
#        
#        for img_path in img_files: # tqdm(img_files, desc=f"Testing Q={q}", leave=False):
#            # --- A. 读取与预处理 ---
#            img_hq = cv2.imread(img_path) # BGR
#            if img_hq is None: continue
#            
#            # 确保尺寸是 8 的倍数 (Mamba/Transformer 有时需要)
#            h, w, c = img_hq.shape
#            pad_h = (8 - h % 8) % 8
#            pad_w = (8 - w % 8) % 8
#            if pad_h != 0 or pad_w != 0:
#                img_hq = np.pad(img_hq, ((0, pad_h), (0, pad_w), (0, 0)), mode='edge')
#
#            # --- B. 模拟 JPEG 压缩 (Baseline) ---
#            _, enc = cv2.imencode('.jpg', img_hq, [int(cv2.IMWRITE_JPEG_QUALITY), q])
#            img_lq = cv2.imdecode(enc, 1) # BGR
#
#            # --- C. 计算 Baseline 指标 ---
#            # 这里的指标计算全部基于 numpy uint8 [0, 255]
#            jpeg_psnr_sum += calc_psnr(img_hq, img_lq)
#            jpeg_ssim_sum += calc_ssim(img_hq, img_lq)
#
#            # --- D. 模型推理 (Ours) ---
#            # BGR -> RGB -> Normalize -> Tensor
#            img_lq_tensor = img_lq.astype(np.float32) / 255.0
#            img_lq_tensor = torch.from_numpy(np.ascontiguousarray(img_lq_tensor[:, :, ::-1].transpose(2, 0, 1)))
#            img_lq_tensor = img_lq_tensor.unsqueeze(0).to(DEVICE) # (1, C, H, W)
#
#            with torch.no_grad():
#                # 推理时温度设低，二值化边缘
#                enhanced_tensor, _ = model(img_lq_tensor, temperature=0.01)
#            
#            # Tensor -> Numpy -> BGR -> [0, 255]
#            enhanced_img = enhanced_tensor.squeeze(0).cpu().numpy().transpose(1, 2, 0)
#            enhanced_img = enhanced_img[:, :, ::-1] # RGB -> BGR
#            enhanced_img = np.clip(enhanced_img * 255.0, 0, 255).astype(np.uint8)
#
#            # --- E. 计算 Ours 指标 ---
#            ours_psnr_sum += calc_psnr(img_hq, enhanced_img)
#            ours_ssim_sum += calc_ssim(img_hq, enhanced_img)
#
#        # 4. 计算当前 Q 的平均分
#        count = len(img_files)
#        avg_jpeg_psnr = jpeg_psnr_sum / count
#        avg_jpeg_ssim = jpeg_ssim_sum / count
#        
#        avg_ours_psnr = ours_psnr_sum / count
#        avg_ours_ssim = ours_ssim_sum / count
#        
#        boost_psnr = avg_ours_psnr - avg_jpeg_psnr
#        boost_ssim = avg_ours_ssim - avg_jpeg_ssim
#
#        # 5. 打印表格行 (模仿截图格式)
#        print(f"|  Q={q:<3}  |  PSNR  |  {avg_jpeg_psnr:.2f} dB      |  {avg_ours_psnr:.2f} dB      | +{boost_psnr:.2f} dB |")
#        print(f"|         |  SSIM  |  {avg_jpeg_ssim:.4f}        |  {avg_ours_ssim:.4f}        | +{boost_ssim:.4f}   |")
#        print("-" * 90)
#
#if __name__ == "__main__":
#    run_test()



import os
import sys
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from skimage.metrics import structural_similarity as ssim_func
from tqdm import tqdm

# ==========================================
# 1) 路径修复（支持在 enhancement/ 内直接运行）
# ==========================================
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

try:
    from sci_enhancer import SCIEnhancementNet
except ImportError:
    from enhancement.sci_enhancer import SCIEnhancementNet


# ==========================================
# 2) 指标
# ==========================================
def calc_psnr(img1, img2):
    """img1, img2: uint8 [0,255], HWC"""
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100.0
    return 20 * np.log10(255.0 / np.sqrt(mse))

def calc_ssim(img1, img2):
    """img1, img2: uint8 [0,255], HWC"""
    # skimage SSIM：channel_axis=2 表示彩色图
    # 注意：win_size 必须小于最小边，默认会自动选；若你有小图可自行设置 win_size
    return ssim_func(img1, img2, channel_axis=2, data_range=255)


# ==========================================
# 3) checkpoint 加载（兼容 best / latest / DataParallel）
# ==========================================
def load_checkpoint_state_dict(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    if isinstance(ckpt, dict) and "model" in ckpt:
        ckpt = ckpt["model"]

    # 去掉可能的 module. 前缀
    new_sd = {}
    for k, v in ckpt.items():
        new_sd[k.replace("module.", "")] = v
    return new_sd


# ==========================================
# 4) 推理 pad/crop：只 pad 输入到偶数（stride=2），输出 crop 回原尺寸
# ==========================================
def pad_to_even_tensor(x):
    # x: [1,3,H,W]
    _, _, h, w = x.shape
    pad_h = h % 2
    pad_w = w % 2
    if pad_h == 0 and pad_w == 0:
        return x, (0, 0)
    x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
    return x, (pad_h, pad_w)

def crop_back(x, orig_h, orig_w):
    return x[:, :, :orig_h, :orig_w]


# ==========================================
# 5) 核心测试
# ==========================================
def run_test():
    TEST_DIR = "/data/lichaofei/data/SCICD-PC/test"
    CHECKPOINT_PATH = "/home/chaofeili/fast-jpeg-main/enhancement/checkpoints/sci_best.pth"
    QUALITY_LIST = [10, 20, 30, 40,50,60,70,80,90,100]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = (device.type == "cuda")
    print("device:", device)
    if use_amp:
        print("gpu:", torch.cuda.get_device_name(0))
        torch.backends.cudnn.benchmark = True

    # 1) load model
    print(f"Loading model from {CHECKPOINT_PATH} ...")
    model = SCIEnhancementNet().to(device).eval()
    sd = load_checkpoint_state_dict(CHECKPOINT_PATH, device)
    model.load_state_dict(sd, strict=True)

    # AMP context（可选提速）
    try:
        from torch import amp
        autocast_ctx = lambda: amp.autocast("cuda", enabled=use_amp)
    except Exception:
        autocast_ctx = lambda: torch.cuda.amp.autocast(enabled=use_amp)

    # 2) images
    img_files = sorted([
        os.path.join(TEST_DIR, f) for f in os.listdir(TEST_DIR)
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"))
    ])
    if len(img_files) == 0:
        print(f"❌ Error: No images found in {TEST_DIR}")
        return

    print(f"Found {len(img_files)} test images.")
    print("=" * 90)
    print(f"| Quality | Metric | {'JPEG (Baseline)':^15} | {'Ours (Mamba)':^15} | {'Boost':^10} |")
    print("-" * 90)

    # 3) loop qualities
    for q in QUALITY_LIST:
        jpeg_psnr_sum, jpeg_ssim_sum = 0.0, 0.0
        ours_psnr_sum, ours_ssim_sum = 0.0, 0.0
        n_used = 0

        for img_path in tqdm(img_files, desc=f"Q={q}", leave=False):
            img_hq = cv2.imread(img_path, cv2.IMREAD_COLOR)  # BGR uint8
            if img_hq is None:
                continue

            # A) baseline: 对原图做 JPEG（不要 pad HQ）
            ok, enc = cv2.imencode(".jpg", img_hq, [int(cv2.IMWRITE_JPEG_QUALITY), int(q)])
            if not ok:
                continue
            img_lq = cv2.imdecode(enc, cv2.IMREAD_COLOR)  # BGR uint8

            # baseline metrics (HQ vs LQ)
            jpeg_psnr_sum += calc_psnr(img_hq, img_lq)
            jpeg_ssim_sum += calc_ssim(img_hq, img_lq)

            # B) ours: LQ -> tensor -> pad to even -> model -> crop back -> uint8
            lq_rgb01 = (img_lq.astype(np.float32) / 255.0)[:, :, ::-1]  # RGB float [0,1]
            lq_t = torch.from_numpy(np.ascontiguousarray(lq_rgb01.transpose(2, 0, 1))).unsqueeze(0).to(device)

            orig_h, orig_w = img_lq.shape[:2]
            lq_pad, _ = pad_to_even_tensor(lq_t)

            with torch.no_grad():
                with autocast_ctx():
                    # 注意：新模型 forward 返回 3 个
                    out_pad, _, _ = model(lq_pad, temperature=0.01)
                out_pad = out_pad.float().clamp(0, 1)

            out = crop_back(out_pad, orig_h, orig_w)

            out_rgb = out.squeeze(0).cpu().numpy().transpose(1, 2, 0)  # RGB [0,1]
            out_bgr = (out_rgb[:, :, ::-1] * 255.0).round().clip(0, 255).astype(np.uint8)

            # ours metrics (HQ vs Ours) —— 同样在原图尺寸上算
            ours_psnr_sum += calc_psnr(img_hq, out_bgr)
            ours_ssim_sum += calc_ssim(img_hq, out_bgr)

            n_used += 1

        if n_used == 0:
            print(f"|  Q={q:<3}  |  PSNR  |  N/A            |  N/A            |  N/A      |")
            print("-" * 90)
            continue

        avg_jpeg_psnr = jpeg_psnr_sum / n_used
        avg_jpeg_ssim = jpeg_ssim_sum / n_used
        avg_ours_psnr = ours_psnr_sum / n_used
        avg_ours_ssim = ours_ssim_sum / n_used

        boost_psnr = avg_ours_psnr - avg_jpeg_psnr
        boost_ssim = avg_ours_ssim - avg_jpeg_ssim

        print(f"|  Q={q:<3}  |  PSNR  |  {avg_jpeg_psnr:.2f} dB      |  {avg_ours_psnr:.2f} dB      | {boost_psnr:+.2f} dB |")
        print(f"|         |  SSIM  |  {avg_jpeg_ssim:.4f}        |  {avg_ours_ssim:.4f}        | {boost_ssim:+.4f}   |")
        print("-" * 90)


if __name__ == "__main__":
    run_test()