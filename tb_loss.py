from __future__ import annotations

from typing import Iterable, Sequence

import torch


def uniform_backward_log_prob(mask: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Compute log P_B under a uniform backward policy over valid actions."""
    if mask.dtype != torch.float32:
        mask = mask.float()
    count = mask.sum().clamp(min=1.0)
    return -torch.log(count + eps)


def _sum_log_probs(log_probs: Sequence[torch.Tensor], device: torch.device) -> torch.Tensor:
    if not log_probs:
        return torch.tensor(0.0, device=device)
    return torch.stack(log_probs).sum()


def trajectory_balance_loss(
    log_z: torch.Tensor,
    forward_log_probs: Sequence[torch.Tensor],
    backward_log_probs: Sequence[torch.Tensor],
    reward: torch.Tensor | float,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Compute the Trajectory Balance loss for a single trajectory."""
    reward_tensor = reward if isinstance(reward, torch.Tensor) else torch.tensor(reward)
    reward_tensor = reward_tensor.to(log_z.device)
    log_reward = torch.log(reward_tensor.clamp(min=eps))

    forward_sum = _sum_log_probs(forward_log_probs, log_z.device)
    backward_sum = _sum_log_probs(backward_log_probs, log_z.device)

    residual = log_z + forward_sum - log_reward - backward_sum
    return residual.pow(2)
