#import math
#import torch
#import torch.nn as nn
#import torch.nn.functional as F
#from torch.utils.checkpoint import checkpoint # <--- 核心救星
#
#def selective_scan_seq(x, dt, A, B, C, D):
#    """
#    纯 PyTorch 实现的串行 SSM 扫描。
#    """
#    batch, seq_len, d_inner = x.shape
#    d_state = A.shape[-1]
#    
#    h = torch.zeros(batch, d_inner, d_state, device=x.device)
#    ys = []
#    
#    for t in range(seq_len):
#        dt_t = dt[:, t]
#        x_t = x[:, t]
#        B_t = B[:, t]
#        C_t = C[:, t]
#        
#        dA_t = torch.exp(torch.einsum('bd,dn->bdn', dt_t, A))
#        dB_t = torch.einsum('bd,bn->bdn', dt_t, B_t)
#        
#        h = dA_t * h + dB_t * x_t.unsqueeze(-1)
#        y_t = torch.einsum('bdn,bn->bd', h, C_t)
#        ys.append(y_t)
#        
#    y = torch.stack(ys, dim=1)
#    y = y + x * D.unsqueeze(0).unsqueeze(0)
#    return y
#
#class Mamba(nn.Module):
#    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
#        super().__init__()
#        self.d_model = d_model
#        self.d_inner = int(expand * d_model)
#        self.dt_rank = math.ceil(d_model / 16)
#        self.d_state = d_state
#        
#        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
#        
#        self.conv1d = nn.Conv1d(
#            in_channels=self.d_inner,
#            out_channels=self.d_inner,
#            bias=True,
#            kernel_size=d_conv,
#            groups=self.d_inner,
#            padding=d_conv - 1,
#        )
#        
#        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + d_state * 2, bias=False)
#        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
#        
#        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
#        self.A_log = nn.Parameter(torch.log(A))
#        self.D = nn.Parameter(torch.ones(self.d_inner))
#        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
#
#    def forward(self, x):
#        B, L, D = x.shape
#        x_proj = self.in_proj(x)
#        x_res, x_gate = x_proj.chunk(2, dim=-1)
#        
#        x_conv = x_res.transpose(1, 2)
#        x_conv = self.conv1d(x_conv)[:, :, :L]
#        x_conv = x_conv.transpose(1, 2)
#        x_conv = F.silu(x_conv)
#        
#        x_dbl = self.x_proj(x_conv) 
#        dt, B_param, C_param = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
#        
#        dt = F.softplus(self.dt_proj(dt))
#        A = -torch.exp(self.A_log)
#        
#        # ============================================================
#        # 【关键修改】使用 Checkpoint 显存优化
#        # 这会让 PyTorch 不保存循环中的中间变量，而是反向传播时重新计算。
#        # 显存占用将大幅下降！
#        # ============================================================
#        # use_reentrant=False 是新版 PyTorch 推荐的，更稳定
#        y = checkpoint(selective_scan_seq, x_conv, dt, A, B_param, C_param, self.D, use_reentrant=False)
#        
#        y = y * F.silu(x_gate)
#        out = self.out_proj(y)
#        return out


import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

def selective_scan_seq(x, dt, A, B, C, D):
    """
    纯 PyTorch 串行扫描（仍非 fused CUDA，但已做预分配优化）
    x: [B,L,d_inner]
    dt: [B,L,d_inner]
    A:  [d_inner,d_state]
    B,C:[B,L,d_state]
    D:  [d_inner]
    """
    batch, L, d_inner = x.shape
    d_state = A.shape[-1]

    h = x.new_zeros((batch, d_inner, d_state))
    y = x.new_empty((batch, L, d_inner))
    D_ = D.view(1, 1, -1)

    for t in range(L):
        dt_t = dt[:, t]          # [B,d_inner]
        x_t  = x[:, t]           # [B,d_inner]
        B_t  = B[:, t]           # [B,d_state]
        C_t  = C[:, t]           # [B,d_state]

        dA_t = torch.exp(dt_t.unsqueeze(-1) * A)                  # [B,d_inner,d_state]
        dB_t = (dt_t.unsqueeze(-1) * B_t.unsqueeze(1))            # [B,d_inner,d_state]

        h = dA_t * h + dB_t * x_t.unsqueeze(-1)
        y[:, t] = torch.einsum('bdn,bn->bd', h, C_t)

    return y + x * D_


class Mamba(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, use_checkpoint=False):
        super().__init__()
        self.d_model = d_model
        self.d_inner = int(expand * d_model)
        self.dt_rank = math.ceil(d_model / 16)
        self.d_state = d_state
        self.use_checkpoint = use_checkpoint

        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)

        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=True,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
        )

        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, x):
        B, L, D = x.shape

        x_proj = self.in_proj(x)
        x_res, x_gate = x_proj.chunk(2, dim=-1)

        x_conv = x_res.transpose(1, 2)
        x_conv = self.conv1d(x_conv)[:, :, :L]
        x_conv = x_conv.transpose(1, 2)
        x_conv = F.silu(x_conv)

        x_dbl = self.x_proj(x_conv)
        dt, B_param, C_param = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)

        dt = F.softplus(self.dt_proj(dt))
        A = -torch.exp(self.A_log)

        if self.use_checkpoint and self.training:
            y = checkpoint(selective_scan_seq, x_conv, dt, A, B_param, C_param, self.D, use_reentrant=False)
        else:
            y = selective_scan_seq(x_conv, dt, A, B_param, C_param, self.D)

        y = y * F.silu(x_gate)
        return self.out_proj(y)