import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .mamba_simple import Mamba
except Exception:
    from mamba_simple import Mamba


class BinarizedEdgeBranch(nn.Module):
    def __init__(self, in_channels=3, hidden_channels=32):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1)
        self.conv_out = nn.Conv2d(hidden_channels, 1, 3, padding=1)

    def forward(self, x, temperature=1.0):
        temperature = max(float(temperature), 1e-3)
        feat = F.relu(self.conv1(x))
        feat = F.relu(self.conv2(feat))
        logits = self.conv_out(feat)
        mask_pred = torch.sigmoid(logits / temperature)
        return mask_pred, logits


#class SpatialFeatureModulation(nn.Module):
#    def __init__(self, mask_channels=1, feat_channels=64):
#        super().__init__()
#        self.conv_gamma = nn.Conv2d(mask_channels, feat_channels, 3, padding=1)
#        self.conv_beta = nn.Conv2d(mask_channels, feat_channels, 3, padding=1)
#
#    def forward(self, x, mask):
#        gamma = self.conv_gamma(mask)
#        beta = self.conv_beta(mask)
#        return x * (1 + gamma) + beta

class SpatialFeatureModulation(nn.Module):
    def __init__(self, mask_channels=1, feat_channels=64, scale=0.1):
        super().__init__()
        self.scale = scale
        self.conv_gamma = nn.Conv2d(mask_channels, feat_channels, 3, padding=1)
        self.conv_beta = nn.Conv2d(mask_channels, feat_channels, 3, padding=1)

        # 关键：零初始化，保证初始 gamma=beta=0（SFT 初始为 identity）
        nn.init.zeros_(self.conv_gamma.weight)
        nn.init.zeros_(self.conv_gamma.bias)
        nn.init.zeros_(self.conv_beta.weight)
        nn.init.zeros_(self.conv_beta.bias)

    def forward(self, x, mask):
        gamma = self.scale * torch.tanh(self.conv_gamma(mask))
        beta = self.scale * torch.tanh(self.conv_beta(mask))
        return x * (1 + gamma) + beta


class AxialMambaBlock(nn.Module):
    """
    行序列 + 列序列（Axial）Mamba
    序列长度从 H*W(16384) 降为 H 或 W(128) => 速度大幅提升
    """
    def __init__(self, channels, d_state=16):
        super().__init__()
        self.norm = nn.LayerNorm(channels)
        self.mamba_row = Mamba(d_model=channels, d_state=d_state, d_conv=4, expand=2, use_checkpoint=False)
        self.mamba_col = Mamba(d_model=channels, d_state=d_state, d_conv=4, expand=2, use_checkpoint=False)
        self.fuse = nn.Conv2d(channels * 2, channels, kernel_size=1)

    def forward(self, x):
        B, C, H, W = x.shape

        # rows: (B*H, W, C)
        xr = x.permute(0, 2, 3, 1).reshape(B * H, W, C)
        xr = self.norm(xr)
        yr = self.mamba_row(xr).reshape(B, H, W, C).permute(0, 3, 1, 2)

        # cols: (B*W, H, C)
        xc = x.permute(0, 3, 2, 1).reshape(B * W, H, C)
        xc = self.norm(xc)
        yc = self.mamba_col(xc).reshape(B, W, H, C).permute(0, 3, 2, 1)

        out = self.fuse(torch.cat([yr, yc], dim=1))
        return x + out


class SCIEnhancementNet(nn.Module):
    def __init__(self, in_channels=3, feat_channels=64):
        super().__init__()
        self.edge_branch = BinarizedEdgeBranch(in_channels=in_channels)

        self.conv_in = nn.Conv2d(in_channels, feat_channels, 3, padding=1)

        # 下采样后做 Axial Mamba（更快）
        self.down = nn.Conv2d(feat_channels, feat_channels, 3, stride=2, padding=1)

        self.mamba1 = AxialMambaBlock(feat_channels, d_state=16)
        self.sft1 = SpatialFeatureModulation(mask_channels=1, feat_channels=feat_channels)

        self.mamba2 = AxialMambaBlock(feat_channels, d_state=16)
        self.sft2 = SpatialFeatureModulation(mask_channels=1, feat_channels=feat_channels)

        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv_out = nn.Conv2d(feat_channels, in_channels, 3, padding=1)
        nn.init.zeros_(self.conv_out.weight)
        if self.conv_out.bias is not None:
            nn.init.zeros_(self.conv_out.bias)

    def forward(self, x, temperature=1.0):
#        mask_pred, _ = self.edge_branch(x, temperature)
        mask_pred, mask_logits = self.edge_branch(x, temperature)

        feat = F.relu(self.conv_in(x))

        feat = self.down(feat)
        mask_low = F.interpolate(mask_pred, size=feat.shape[-2:], mode="nearest")

        feat = self.mamba1(feat)
        feat = self.sft1(feat, mask_low)

        feat = self.mamba2(feat)
        feat = self.sft2(feat, mask_low)

        feat = self.up(feat)

        out = self.conv_out(feat) + x
#        return out, mask_pred
        return out, mask_pred, mask_logits


#class SCILoss(nn.Module):
#    def __init__(self):
#        super().__init__()
#        self.mse = nn.MSELoss()
#
#    def polarization_loss(self, mask_pred):
#        return torch.mean(mask_pred * (1.0 - mask_pred))
#
#    def transition_penalty_loss(self, pred_img, gt_img, gt_mask):
#        local_max = F.max_pool2d(gt_img, kernel_size=3, stride=1, padding=1)
#        local_min = -F.max_pool2d(-gt_img, kernel_size=3, stride=1, padding=1)
#
#        dist_to_max = torch.abs(pred_img - local_max)
#        dist_to_min = torch.abs(pred_img - local_min)
#        min_dist = torch.min(dist_to_max, dist_to_min)
#
#        transition_loss = torch.sum(min_dist * gt_mask) / (torch.sum(gt_mask) + 1e-6)
#        return transition_loss
#
#    def forward(self, pred_img, gt_img, mask_pred, gt_mask):
#        loss_mse = self.mse(pred_img, gt_img)
#        loss_polar = self.polarization_loss(mask_pred)
#        loss_trans = self.transition_penalty_loss(pred_img, gt_img, gt_mask)
#
#        total_loss = loss_mse + 0.1 * loss_polar + 0.5 * loss_trans
#        return total_loss, loss_mse, loss_polar, loss_trans

class SCILoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss()

    def polarization_loss(self, mask_pred):
        return torch.mean(mask_pred * (1.0 - mask_pred))

    def transition_penalty_loss(self, pred_img, gt_img, gt_mask):
        local_max = F.max_pool2d(gt_img, kernel_size=3, stride=1, padding=1)
        local_min = -F.max_pool2d(-gt_img, kernel_size=3, stride=1, padding=1)

        dist_to_max = torch.abs(pred_img - local_max)
        dist_to_min = torch.abs(pred_img - local_min)
        min_dist = torch.min(dist_to_max, dist_to_min)

        return torch.sum(min_dist * gt_mask) / (torch.sum(gt_mask) + 1e-6)

    def forward(self, pred_img, gt_img, mask_pred, mask_logits, gt_mask,
                w_polar=0.01, w_trans=0.1, w_edge=0.1):
        loss_mse = self.mse(pred_img, gt_img)
        loss_polar = self.polarization_loss(mask_pred)
        loss_trans = self.transition_penalty_loss(pred_img, gt_img, gt_mask)
        loss_edge = self.bce(mask_logits, gt_mask)

        total_loss = loss_mse + w_polar * loss_polar + w_trans * loss_trans + w_edge * loss_edge
        return total_loss, loss_mse, loss_polar, loss_trans, loss_edge