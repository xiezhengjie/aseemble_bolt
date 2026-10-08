import math
import torch
import numpy as np
import torch.nn.functional as F 
from typing import Dict
from pathlib import Path
from diffusers.optimization import get_scheduler
from training.model.gail.discriminator import DiscriminatorNN   
from training.policy.base_policy import BasePolicy

DISC_LOGIT_CLAMP = 20.0  # z clamp ±20 → softplus(−z)∈[≈0,20]；对外 r̃/20 ∈[0,1]

class Discriminator(BasePolicy):
    def __init__(self, state_dim, action_dim, hidden_dim,
                 optimizer: Dict, lr_scheduler: Dict, num_training_steps: int,
                 smoothing=0.1, grad_clip_norm=None,
                 ent_reg_scale=0.001, dropout=0.1):
        super().__init__()
        ent_reg_scale = float(ent_reg_scale)
        if not math.isfinite(ent_reg_scale) or ent_reg_scale < 0.0:
            raise ValueError("ent_reg_scale must be a finite non-negative number")

        self.state_dim = int(state_dim)
        self.smoothing = smoothing
        self.grad_clip_norm = grad_clip_norm
        self.ent_reg_scale = ent_reg_scale
        self.disc = DiscriminatorNN(state_dim, action_dim, hidden_dim, dropout=dropout).to(self.device)
        self.disc_optim = torch.optim.AdamW(
            self.disc.parameters(),
            lr=optimizer['lr'],
            betas=optimizer['betas'],
            eps=optimizer['eps'],
            weight_decay=optimizer['weight_decay'],
        )
        self.lr_scheduler = get_scheduler(
            lr_scheduler['name'],
            optimizer=self.disc_optim,
            num_warmup_steps=lr_scheduler['num_warmup_steps'],
            num_training_steps=num_training_steps,
        )


    def _as_2d(self, x):
        t = torch.as_tensor(np.asarray(x), dtype=torch.float32)
        if t.dim() == 1:  t = t.unsqueeze(0)
        elif t.dim() > 2: t = t.reshape(t.shape[0], -1)
        return t.to(self.device)

    def _states_to_batch(self, states):
        return self.normalize_obs(self._as_2d(states))

    def update(self,expert_states, expert_actions, gen_states, gen_actions,  log_info=True):
        expert_d = self.disc(self._states_to_batch(expert_states), self._as_2d(expert_actions))
        policy_d = self.disc(self._states_to_batch(gen_states), self._as_2d(gen_actions))

        # 计算专家数据和生成数据的损失
        expert_loss = F.binary_cross_entropy_with_logits(expert_d, torch.full_like(expert_d, self.smoothing))
        policy_loss = F.binary_cross_entropy_with_logits(policy_d, torch.full_like(policy_d, 1 - self.smoothing))
        bce_loss = expert_loss + policy_loss

        # 计算预测熵损失
        logits = torch.cat((expert_d, policy_d), dim=0)
        abs_logits = logits.abs()
        prediction_entropy = ( F.softplus(-abs_logits) + abs_logits * torch.sigmoid(-abs_logits)).mean()
        entropy_loss = -self.ent_reg_scale * prediction_entropy
        gail_loss = bce_loss + entropy_loss

        self.disc_optim.zero_grad()
        gail_loss.backward()
        if self.grad_clip_norm is not None:
            gn = torch.nn.utils.clip_grad_norm_(self.disc.parameters(), self.grad_clip_norm)
        else:
            gn = None
        self.disc_optim.step()
        self.lr_scheduler.step()

        if log_info:
            return {
                'loss': gail_loss.item(),
                'bce_loss': bce_loss.item(),
                'entropy': prediction_entropy.item(),
                'entropy_loss': entropy_loss.item(),
                'expert_value': torch.sigmoid(expert_d).mean().item(),
                'policy_value': torch.sigmoid(policy_d).mean().item(),
                'grad_norm': gn.item() if torch.is_tensor(gn) else
                (float(gn) if gn is not None else 0.0)}

        else:
            return {}
    
    def predict_policy_prob(self, states, actions, to_numpy=True):
        """D = P(policy|s,a) = sigmoid(ℓ)，与 expert_value / policy_value 同一尺度。"""
        with torch.no_grad():
            logit = self.disc(self._states_to_batch(states), self._as_2d(actions)).squeeze(-1)
            d = torch.sigmoid(logit)
            if to_numpy:
                return d.cpu().numpy()
            return d

    def predict_rewards(self, states, actions, to_numpy=True):
        """ r̃ = softplus(−z) """
        with torch.no_grad():
            logit = self.disc(self._states_to_batch(states), self._as_2d(actions)).squeeze(-1)
            z = torch.clamp(logit, -DISC_LOGIT_CLAMP, DISC_LOGIT_CLAMP)  # logit_clamp ∈ [0,1]（clamp=20）
            reward = F.softplus(-z) 
        if to_numpy:
            return reward.cpu().numpy()
        return reward

    def load_model(self, model_dir):
        _model_dir = Path(model_dir)
        _model_dir.mkdir(parents=True, exist_ok=True)
        self.disc.load_state_dict(torch.load(
            _model_dir/"discriminator_net.pt", map_location=self.device, weights_only=True,
        ))
        self._load_obs_normalizer(_model_dir)

    def save_model(self, model_dir):
        _model_dir = Path(model_dir)
        _model_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.disc.state_dict(), _model_dir/"discriminator_net.pt")
        self._save_obs_normalizer(_model_dir)
