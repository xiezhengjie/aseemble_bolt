import math
import torch
import numpy as np
import torch.nn.functional as F 
from typing import Dict
from pathlib import Path
from diffusers.optimization import get_scheduler
from training.model.gail.discriminator import DiscriminatorNN   
from training.policy.base_policy import BasePolicy
from training.common.rl_utils import discriminator_weights_init

DISC_LOGIT_CLAMP = 20.0  # z clamp ±20 → softplus(−z)∈[≈0,20]；对外 r̃/20 ∈[0,1]

class Discriminator(BasePolicy):
    def __init__(self, state_dim, action_dim, hidden_dim,
                 optimizer: Dict, lr_scheduler: Dict, num_training_steps: int,
                 smoothing=0.1, grad_clip_norm=None,
                 ent_reg_scale=0.001,
                 dropout_in=0.2,
                 dropout=0.2,
                 weight_init=True,
                 last_layer_scale=0.1,
                 gradient_penalty=False,
                 gradient_penalty_lambda=1.0,
                 gradient_penalty_k=0.0):
        super().__init__()
        ent_reg_scale = float(ent_reg_scale)
        if not math.isfinite(ent_reg_scale) or ent_reg_scale < 0.0:
            raise ValueError("ent_reg_scale must be a finite non-negative number")

        self.state_dim = int(state_dim)
        self.smoothing = smoothing
        self.grad_clip_norm = grad_clip_norm
        self.ent_reg_scale = ent_reg_scale
        self.gradient_penalty = bool(gradient_penalty)
        self.gradient_penalty_lambda = float(gradient_penalty_lambda)
        self.gradient_penalty_k = float(gradient_penalty_k)
        self.disc = DiscriminatorNN(state_dim, action_dim, hidden_dim, dropout_in=dropout_in, dropout=dropout).to(self.device)
        if weight_init:
            discriminator_weights_init(self.disc, last_layer_scale=last_layer_scale)
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

    def _gradient_penalty(self, expert_states, expert_actions, gen_states, gen_actions):
        expert = torch.cat([expert_states.detach(), expert_actions.detach()], dim=-1)
        generated = torch.cat([gen_states.detach(), gen_actions.detach()], dim=-1)
        mixing = torch.rand((expert.shape[0], 1), device=self.device)
        interpolated = (mixing * expert + (1.0 - mixing) * generated).requires_grad_(True)
        logits = self.disc(interpolated[:, :self.state_dim], interpolated[:, self.state_dim:])
        gradients = torch.autograd.grad(
            outputs=logits,
            inputs=interpolated,
            grad_outputs=torch.ones_like(logits),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        norms = gradients.norm(2, dim=-1)
        if self.gradient_penalty_k == 0.0:
            penalty = self.gradient_penalty_lambda * gradients.square().sum(dim=-1).mean()
        else:
            penalty = self.gradient_penalty_lambda * (norms - self.gradient_penalty_k).square().mean()
        return penalty, norms.detach().mean()

    def update(self,expert_states, expert_actions, gen_states, gen_actions,  log_info=True):
        expert_states = self._as_2d(expert_states)
        expert_actions = self._as_2d(expert_actions)
        gen_states = self._as_2d(gen_states)
        gen_actions = self._as_2d(gen_actions)
        expert_d = self.disc(expert_states, expert_actions)
        policy_d = self.disc(gen_states, gen_actions)

        # 计算专家数据和生成数据的损失
        expert_loss = F.binary_cross_entropy_with_logits(expert_d, torch.full_like(expert_d, self.smoothing))
        policy_loss = F.binary_cross_entropy_with_logits(policy_d, torch.full_like(policy_d, 1 - self.smoothing))
        bce_loss = expert_loss + policy_loss
        if self.gradient_penalty:
            gp_loss, gp_grad_norm = self._gradient_penalty(
                expert_states, expert_actions, gen_states, gen_actions,
            )
        else:
            gp_loss = torch.zeros((), device=self.device)
            gp_grad_norm = gp_loss

        # 计算预测熵损失
        logits = torch.cat((expert_d, policy_d), dim=0)
        abs_logits = logits.abs()
        prediction_entropy = ( F.softplus(-abs_logits) + abs_logits * torch.sigmoid(-abs_logits)).mean()
        entropy_loss = -self.ent_reg_scale * prediction_entropy
        gail_loss = bce_loss + entropy_loss + gp_loss

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
                'gp_loss': gp_loss.item(),
                'gp_grad_norm': gp_grad_norm.item(),
                'expert_value': torch.sigmoid(expert_d).mean().item(),
                'policy_value': torch.sigmoid(policy_d).mean().item(),
                'grad_norm': gn.item() if torch.is_tensor(gn) else
                (float(gn) if gn is not None else 0.0)}
        else:
            return {}
        
    
    def _predict_logits(self, states, actions):
        training = self.disc.training
        try:
            self.disc.eval()
            with torch.no_grad():
                return self.disc(
                    self._as_2d(states), self._as_2d(actions),
                ).squeeze(-1)
        finally:
            self.disc.train(training)

    def predict_policy_prob(self, states, actions, to_numpy=True):
        """D = P(policy|s,a) = sigmoid(ℓ)，与 expert_value / policy_value 同一尺度。"""
        d = torch.sigmoid(self._predict_logits(states, actions))
        return d.cpu().numpy() if to_numpy else d

    def predict_rewards(self, states, actions, to_numpy=True):
        """ r̃ = softplus(−z) """
        logit = self._predict_logits(states, actions)
        z = torch.clamp(logit, -DISC_LOGIT_CLAMP, DISC_LOGIT_CLAMP)
        reward = F.softplus(-z)
        return reward.cpu().numpy() if to_numpy else reward

    def load_model(self, model_dir):
        _model_dir = Path(model_dir)
        _model_dir.mkdir(parents=True, exist_ok=True)
        self.disc.load_state_dict(torch.load(
            _model_dir/"discriminator_net.pt", map_location=self.device, weights_only=True,
        ))

    def save_model(self, model_dir):
        _model_dir = Path(model_dir)
        _model_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.disc.state_dict(), _model_dir/"discriminator_net.pt")
