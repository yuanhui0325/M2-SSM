#import torch
#import torch.nn as nn
#import torch.nn.functional as F
#from mamba_simple import Mamba 
#
#
## ==========================================
## 1. 可微二值化边缘掩码分支 (带温度退火)
## ==========================================
#class BinarizedEdgeBranch(nn.Module):
#    def __init__(self, in_channels=3, hidden_channels=32):
#        super().__init__()
#        self.conv1 = nn.Conv2d(in_channels, hidden_channels, 3, padding=1)
#        self.conv2 = nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1)
#        self.conv_out = nn.Conv2d(hidden_channels, 1, 3, padding=1)
#        
#    def forward(self, x, temperature=1.0):
#        feat = F.relu(self.conv1(x))
#        feat = F.relu(self.conv2(feat))
#        logits = self.conv_out(feat)
#        # 使用温度系数的 Sigmoid。训练后期 temperature 趋近于 0，输出趋于二值化 (0或1)
#        mask_pred = torch.sigmoid(logits / temperature)
#        return mask_pred, feat
#
## ==========================================
## 2. 空间特征调制模块 (SFT - 注入边缘先验)
## ==========================================
#class SpatialFeatureModulation(nn.Module):
#    def __init__(self, mask_channels=1, feat_channels=64):
#        super().__init__()
#        self.conv_gamma = nn.Conv2d(mask_channels, feat_channels, 3, padding=1)
#        self.conv_beta = nn.Conv2d(mask_channels, feat_channels, 3, padding=1)
#        
#    def forward(self, x, mask):
#        # 根据边缘掩码生成缩放因子和偏移量
#        gamma = self.conv_gamma(mask)
#        beta = self.conv_beta(mask)
#        # 对主干网络的特征进行调制
#        return x * (1 + gamma) + beta
#
## ==========================================
## 3. 2D Vision Mamba 模块 (交叉扫描机制)
## ==========================================
##class Simple2DMambaBlock(nn.Module):
##    """
##    这里提供一个 Mamba 交叉扫描的结构框架。
##    实际应用中，内部的 1D 序列建模需要依赖 `mamba_ssm` 库的 Mamba 模块。
##    """
##    def __init__(self, channels):
##        super().__init__()
##        self.norm = nn.LayerNorm(channels)
##        # 假设这里有一个标准的 1D Mamba 层
##        # self.mamba = Mamba(d_model=channels, d_state=16, d_conv=4, expand=2)
##        self.linear_fuse = nn.Linear(channels * 4, channels)
##        
##    def forward(self, x):
##        B, C, H, W = x.shape
##        x_flat = x.flatten(2).transpose(1, 2) # (B, L, C)
##        x_norm = self.norm(x_flat)
##        
##        # 1. 交叉扫描 (Cross-Scan): 构造四个方向的序列
##        scan_tl_br = x_norm                       # 左上 -> 右下
##        scan_br_tl = torch.flip(x_norm, dims=[1]) # 右下 -> 左上
##        
##        x_norm_t = x_norm.view(B, H, W, C).transpose(1, 2).reshape(B, H*W, C)
##        scan_tr_bl = x_norm_t                     # 右上 -> 左下
##        scan_bl_tr = torch.flip(x_norm_t, dims=[1]) # 左下 -> 右上
##        
##        # 2. Mamba 序列建模 (这里用 Identity 代替真实的 Mamba 层进行示意)
##        # out_tl_br = self.mamba(scan_tl_br)
##        out_tl_br = scan_tl_br 
##        out_br_tl = torch.flip(scan_br_tl, dims=[1]) # 算完后翻转回原位置
##        out_tr_bl = scan_tr_bl.view(B, W, H, C).transpose(1, 2).reshape(B, H*W, C)
##        out_bl_tr = torch.flip(scan_bl_tr, dims=[1]).view(B, W, H, C).transpose(1, 2).reshape(B, H*W, C)
##        
##        # 3. 特征拼接与融合
##        fused = torch.cat([out_tl_br, out_br_tl, out_tr_bl, out_bl_tr], dim=-1)
##        fused = self.linear_fuse(fused) # (B, L, C)
##        
##        out = fused.transpose(1, 2).view(B, C, H, W)
##        return x + out # 残差连接
#class Simple2DMambaBlock(nn.Module):
#    """
#    集成了纯 PyTorch 版 Mamba 的 2D 交叉扫描模块。
#    无需安装 mamba_ssm，直接运行。
#    """
#    def __init__(self, channels):
#        super().__init__()
#        self.norm = nn.LayerNorm(channels)
#        
#        # 【修改点1】初始化 Mamba 模块
#        # d_state=16 是控制记忆容量的关键参数
#        self.mamba = Mamba(d_model=channels, d_state=16, d_conv=4, expand=2)
#        
#        # 融合层：因为有4个方向，拼接后通道数翻4倍，需要映射回去
#        self.linear_fuse = nn.Linear(channels * 4, channels)
#        
#    def forward(self, x):
#        B, C, H, W = x.shape
#        # (B, C, H, W) -> (B, H*W, C)
#        x_flat = x.flatten(2).transpose(1, 2) 
#        x_norm = self.norm(x_flat)
#        
#        # ==========================================
#        # 1. 准备四个扫描方向 (Cross-Scan Strategy)
#        # ==========================================
#        
#        # 方向1: 左上 -> 右下 (标准顺序)
#        scan_tl_br = x_norm
#        
#        # 方向2: 右下 -> 左上 (翻转序列)
#        scan_br_tl = torch.flip(x_norm, dims=[1])
#        
#        # 为了做垂直扫描，先转置 H 和 W
#        x_norm_t = x_norm.view(B, H, W, C).transpose(1, 2).reshape(B, H*W, C)
#        
#        # 方向3: 右上 -> 左下 (转置后的标准顺序)
#        scan_tr_bl = x_norm_t
#        
#        # 方向4: 左下 -> 右上 (转置后的翻转顺序)
#        scan_bl_tr = torch.flip(x_norm_t, dims=[1])
#        
#        # ==========================================
#        # 2. Mamba 序列建模 (核心步骤)
#        # ==========================================
#        
#        # 【修改点2】让 Mamba 真正处理这4个序列
#        # Mamba 能够记住长距离依赖，修复断裂的线条
#        feat_tl_br = self.mamba(scan_tl_br)
#        feat_br_tl = self.mamba(scan_br_tl)
#        feat_tr_bl = self.mamba(scan_tr_bl)
#        feat_bl_tr = self.mamba(scan_bl_tr)
#        
#        # ==========================================
#        # 3. 还原序列位置 (Un-scan)
#        # ==========================================
#        
#        # 还原方向1
#        out_tl_br = feat_tl_br
#        
#        # 还原方向2: 再次翻转回来
#        out_br_tl = torch.flip(feat_br_tl, dims=[1])
#        
#        # 还原方向3: 变形 -> 转置回 (H, W) -> 展平
#        out_tr_bl = feat_tr_bl.view(B, W, H, C).transpose(1, 2).reshape(B, H*W, C)
#        
#        # 还原方向4: 翻转 -> 变形 -> 转置回 -> 展平
#        out_bl_tr = torch.flip(feat_bl_tr, dims=[1]).view(B, W, H, C).transpose(1, 2).reshape(B, H*W, C)
#        
#        # ==========================================
#        # 4. 特征融合
#        # ==========================================
#        
#        # 将四个方向的特征拼接：(B, L, 4*C)
#        fused = torch.cat([out_tl_br, out_br_tl, out_tr_bl, out_bl_tr], dim=-1)
#        
#        # 降维融合：(B, L, 4*C) -> (B, L, C)
#        fused = self.linear_fuse(fused)
#        
#        # 变回图像形状：(B, L, C) -> (B, C, H, W)
#        out = fused.transpose(1, 2).view(B, C, H, W)
#        
#        return x + out # 残差连接
## ==========================================
## 4. 完整的屏幕内容增强网络
## ==========================================
#class SCIEnhancementNet(nn.Module):
#    def __init__(self, in_channels=3, feat_channels=64):
#        super().__init__()
#        self.edge_branch = BinarizedEdgeBranch(in_channels=in_channels)
#        
#        self.conv_in = nn.Conv2d(in_channels, feat_channels, 3, padding=1)
#        
#        # 使用 Mamba 修复长距离断裂
#        self.mamba_block1 = Simple2DMambaBlock(feat_channels)
#        self.sft1 = SpatialFeatureModulation(mask_channels=1, feat_channels=feat_channels)
#        
#        self.mamba_block2 = Simple2DMambaBlock(feat_channels)
#        self.sft2 = SpatialFeatureModulation(mask_channels=1, feat_channels=feat_channels)
#        
#        self.conv_out = nn.Conv2d(feat_channels, in_channels, 3, padding=1)
#
#    def forward(self, x, temperature=1.0):
#        # 1. 提取边缘掩码
#        mask_pred, _ = self.edge_branch(x, temperature)
#        
#        # 2. 主干网络特征提取
#        feat = F.relu(self.conv_in(x))
#        
#        # 3. Mamba 长距离修复 + SFT 边缘注意力注入
#        feat = self.mamba_block1(feat)
#        feat = self.sft1(feat, mask_pred)
#        
#        feat = self.mamba_block2(feat)
#        feat = self.sft2(feat, mask_pred)
#        
#        # 4. 图像重建 (结合全局残差)
#        out = self.conv_out(feat) + x
#        return out, mask_pred
#
## ==========================================
## 5. 自定义损失函数 (针对振铃和过渡伪影)
## ==========================================
#class SCILoss(nn.Module):
#    def __init__(self):
#        super().__init__()
#        self.mse = nn.MSELoss()
#
#    def polarization_loss(self, mask_pred):
#        # 强制掩码向 0 或 1 极化，在 0.5 时惩罚最大
#        return torch.mean(mask_pred * (1.0 - mask_pred))
#
#    def transition_penalty_loss(self, pred_img, gt_img, gt_mask):
#        # 在真实边缘区域，惩罚出现灰度过渡的像素
#        # 简化实现：使用形态学膨胀/腐蚀寻找局部极值，这里用局部最大池化近似
#        local_max = F.max_pool2d(gt_img, kernel_size=3, stride=1, padding=1)
#        local_min = -F.max_pool2d(-gt_img, kernel_size=3, stride=1, padding=1)
#        
#        # 计算像素到最大值和最小值的距离
#        dist_to_max = torch.abs(pred_img - local_max)
#        dist_to_min = torch.abs(pred_img - local_min)
#        
#        # 强制像素贴近最大值或最小值之一 (去除中间灰度)
#        min_dist = torch.min(dist_to_max, dist_to_min)
#        
#        # 只在边缘区域施加此惩罚
#        transition_loss = torch.sum(min_dist * gt_mask) / (torch.sum(gt_mask) + 1e-6)
#        return transition_loss
#
#    def forward(self, pred_img, gt_img, mask_pred, gt_mask):
#        loss_mse = self.mse(pred_img, gt_img)
#        loss_polar = self.polarization_loss(mask_pred)
#        loss_trans = self.transition_penalty_loss(pred_img, gt_img, gt_mask)
#        
#        # 权重可以根据实验调整
#        total_loss = loss_mse + 0.1 * loss_polar + 0.5 * loss_trans
#        return total_loss, loss_mse, loss_polar, loss_trans


       
       