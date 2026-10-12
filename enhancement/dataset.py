import os
import cv2
import torch
import numpy as np
import random
from torch.utils.data import Dataset

cv2.setNumThreads(0)

class SCIDataset(Dataset):
def init(self, hq_dir, patch_size=128, jpeg_quality_range=(10, 40), is_train=True):
super().init()
self.hq_dir = hq_dir
self.patch_size = patch_size
self.jpeg_quality_range = jpeg_quality_range
self.is_train = is_train

text

    self.hq_files = sorted([
        os.path.join(hq_dir, f) for f in os.listdir(hq_dir)
        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif'))
    ])

def __len__(self):
    return len(self.hq_files)

def _pad_to_min_size(self, img_bgr, min_h, min_w):
    h, w = img_bgr.shape[:2]
    pad_h = max(0, min_h - h)
    pad_w = max(0, min_w - w)
    if pad_h > 0 or pad_w > 0:
        img_bgr = cv2.copyMakeBorder(
            img_bgr, 0, pad_h, 0, pad_w,
            borderType=cv2.BORDER_REFLECT_101
        )
    return img_bgr

def __getitem__(self, idx):
    # 读取失败就换一张（用 while 防止递归过深）
    while True:
        img_path = self.hq_files[idx]
        img_hq = cv2.imread(img_path, cv2.IMREAD_COLOR)  # BGR
        if img_hq is not None:
            break
        idx = random.randint(0, len(self.hq_files) - 1)

    ps = self.patch_size
    img_hq = self._pad_to_min_size(img_hq, ps, ps)
    h, w = img_hq.shape[:2]

    if self.is_train:
        top = random.randint(0, h - ps)
        left = random.randint(0, w - ps)
        patch_hq = img_hq[top:top + ps, left:left + ps].copy()

        # 增强
        if random.random() < 0.5:
            patch_hq = cv2.flip(patch_hq, 1)  # hflip
        if random.random() < 0.5:
            patch_hq = cv2.flip(patch_hq, 0)  # vflip
        if random.random() < 0.5:
            patch_hq = np.ascontiguousarray(np.rot90(patch_hq, k=1))  # 90deg
    else:
        top = (h - ps) // 2
        left = (w - ps) // 2
        patch_hq = img_hq[top:top + ps, left:left + ps].copy()

    # JPEG 模拟（在 patch 上做）
    if isinstance(self.jpeg_quality_range, (tuple, list)):
        quality = random.randint(self.jpeg_quality_range[0], self.jpeg_quality_range[1])
    else:
        quality = int(self.jpeg_quality_range)

    ok, enc = cv2.imencode(".jpg", patch_hq, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        # 极少失败，fallback：直接用 HQ 当 LQ
        patch_lq = patch_hq.copy()
    else:
        patch_lq = cv2.imdecode(enc, cv2.IMREAD_COLOR)

    # Edge GT（Canny）
    gray = cv2.cvtColor(patch_hq, cv2.COLOR_BGR2GRAY)
edge = cv2.Canny(gray, 100, 200) # [0,255]
edge = (edge[:, :, None].astype(np.float32) / 255.0) # (H,W,1) in {0,1}
text

    edge = cv2.Canny(gray, 100, 200)  # [0,255]
    kernel = np.ones((3, 3), np.uint8)
    edge = cv2.dilate(edge, kernel, iterations=1)  # edge band
    edge = (edge[:, :, None].astype(np.float32) / 255.0)  # (H,W,1) in {0,1}

    # BGR->RGB, [0,1]
    hq = patch_hq.astype(np.float32) / 255.0
    lq = patch_lq.astype(np.float32) / 255.0
    hq = hq[:, :, ::-1]  # RGB
    lq = lq[:, :, ::-1]  # RGB

    # to tensor (C,H,W)
    hq_t = torch.from_numpy(np.ascontiguousarray(hq.transpose(2, 0, 1)))
    lq_t = torch.from_numpy(np.ascontiguousarray(lq.transpose(2, 0, 1)))
    edge_t = torch.from_numpy(np.ascontiguousarray(edge.transpose(2, 0, 1)))

    return lq_t, hq_t, edge_t