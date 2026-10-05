from typing import Union
import logging
import torch
import torch.nn as nn
import einops
from einops.layers.torch import Rearrange

from training.model.base.conv1d_components import (
    Downsample1d, Upsample1d, Conv1dBlock
)
from training.model.base.positional_embedding import SinusoidalPosEmb

logger = logging.getLogger(__name__)

class ConditionalUnet1D(nn.Module):
    """
    条件1D U-Net：输入噪声动作序列，输出去噪后的动作序列
    条件注入：通过FiLM（Feature-wise Linear Modulation）将观察特征z融入每层
    """
    def __init__(self, 
                 input_dim=2,      # 动作维度（x, y）
                 global_cond_dim=256,  # 观察条件维度
                 diffusion_step_embed_dim=128,
                 down_dims=[256, 512, 1024],
                 kernel_size=5,
                 n_groups=8):
        super().__init__()

        self.down_dims = down_dims

        # 时间步编码
        self.diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(diffusion_step_embed_dim),
            nn.Linear(diffusion_step_embed_dim, diffusion_step_embed_dim * 4),
            nn.Mish(),
            nn.Linear(diffusion_step_embed_dim * 4, diffusion_step_embed_dim)
        )
        
        # 条件投影：将观察特征z映射到各层的FiLM参数
        all_dims = [input_dim] + list(down_dims)
        self.global_cond_encoder = nn.Sequential(
            nn.Linear(global_cond_dim + diffusion_step_embed_dim, 256),
            nn.Mish(),
            nn.Linear(256, sum(down_dims) * 2)  # scale + shift for each layer
        )
        
        # 下采样路径
        self.down_modules = nn.ModuleList()
        for i in range(len(down_dims)):
            self.down_modules.append(
                Conv1dBlock(all_dims[i], all_dims[i+1], kernel_size)
            )
        
        # 中间层
        self.mid_module = Conv1dBlock(down_dims[-1], down_dims[-1], kernel_size)
        
        # 上采样路径
        self.up_modules = nn.ModuleList()
        for i in reversed(range(len(down_dims[1:]))):
            self.up_modules.append(
                Conv1dBlock(all_dims[i+2] * 2, all_dims[i+1], kernel_size)  # *2 for skip connection
            )
        
        # 输出层
        self.final_conv = nn.Sequential(
            Conv1dBlock(down_dims[0], down_dims[0], kernel_size),
            nn.Conv1d(down_dims[0], input_dim, 1)
        )

        logger.info(
            "number of parameters: %e", sum(p.numel() for p in self.parameters())
        )
        
    def forward(self, noisy_actions, timestep, global_cond):
        """
        noisy_actions: (B, T, input_dim) 噪声动作序列，T=16
        timestep: (B,) 扩散时间步
        global_cond: (B, global_cond_dim) 观察条件特征
        return: (B, T, input_dim) 预测噪声
        """
        noisy_actions = einops.rearrange(noisy_actions, 'b t h -> b h t')
        B, _, T = noisy_actions.shape

        # 时间步预处理：确保 timestep 为 (B,) 的 1D 张量
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=noisy_actions.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(noisy_actions.device)
        timesteps = timesteps.expand(B)

        # 时间步编码
        t_emb = self.diffusion_step_encoder(timesteps)  # (B, diff_embed_dim)
        
        # 条件融合
        cond = torch.cat([t_emb, global_cond], dim=-1)
        cond_params = self.global_cond_encoder(cond)  # (B, sum(dims)*2)
        
        # 分割为各层的scale和shift
        film_params = torch.split(cond_params, [d*2 for d in self.down_dims], dim=-1)
        
        # 下采样
        x = noisy_actions
        skips = []
        for i, down_module in enumerate(self.down_modules):
            x = down_module(x)
            # FiLM调制
            scale, shift = torch.chunk(film_params[i], 2, dim=-1)
            scale = scale.view(B, -1, 1)
            shift = shift.view(B, -1, 1)
            x = x * (1 + scale) + shift
            skips.append(x)
        
        # 中间
        x = self.mid_module(x)
        
        # 上采样（带skip connection）
        for i, up_module in enumerate(self.up_modules):
            x = torch.cat([x, skips[-(i+1)]], dim=1)  # skip connection
            x = up_module(x)
        
        # 输出：预测噪声
        noise_pred = self.final_conv(x)
        return einops.rearrange(noise_pred, 'b h t -> b t h')