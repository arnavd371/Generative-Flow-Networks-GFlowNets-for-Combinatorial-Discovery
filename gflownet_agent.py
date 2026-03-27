from __future__ import annotations

import random
from typing import List, Sequence, Tuple, Union

import torch
from torch import nn

from mis_env import MISDAGEnv, MISState
from sorting_network_env import SortingNetworkDAGEnv, SortingNetworkState

Env = Union[MISDAGEnv, SortingNetworkDAGEnv]
State = Union[MISState, SortingNetworkState]


def infer_state_dim(env: Env) -> int:
    """Infer the flattened state dimension for the provided environment."""
    if isinstance(env, MISDAGEnv):
        return 2 * env.num_nodes + 2
    if isinstance(env, SortingNetworkDAGEnv):
        return env.max_layers * env.num_comparator_types + 2
    raise TypeError("Unsupported environment type for state dimension inference")


def _bitmask_to_list(mask: int, length: int) -> List[float]:
    return [1.0 if mask & (1 << idx) else 0.0 for idx in range(length)]


def encode_state(env: Env, state: State, device: torch.device | None = None) -> torch.Tensor:
    """Encode an environment state into a flat float tensor."""
    if isinstance(env, MISDAGEnv) and isinstance(state, MISState):
        selected = _bitmask_to_list(state.selected_mask, env.num_nodes)
        forbidden = _bitmask_to_list(state.forbidden_mask, env.num_nodes)
        features = selected + forbidden
        features.append(1.0 if state.halted else 0.0)
        features.append(state.set_size / max(env.num_nodes, 1))
        return torch.tensor(features, dtype=torch.float32, device=device)

    if isinstance(env, SortingNetworkDAGEnv) and isinstance(state, SortingNetworkState):
        features: List[float] = []
        for layer_mask in state.layer_masks:
            features.extend(_bitmask_to_list(layer_mask, env.num_comparator_types))
        features.append(1.0 if state.halted else 0.0)
        features.append(state.num_comparators / max(env.max_comparators, 1))
        return torch.tensor(features, dtype=torch.float32, device=device)

    raise TypeError("Environment/state pair is not supported for encoding")


class GFlowNetModel(nn.Module):
    """Simple MLP that predicts forward-transition logits and learns log Z."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
    ) -> None:
        super().__init__()
        if state_dim <= 0:
            raise ValueError("state_dim must be positive")
        if action_dim <= 0:
            raise ValueError("action_dim must be positive")
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")

        layers: List[nn.Module] = []
        in_dim = state_dim
        if num_layers == 1:
            layers.append(nn.Linear(in_dim, action_dim))
        else:
            for _ in range(num_layers - 1):
                layers.append(nn.Linear(in_dim, hidden_dim))
                layers.append(nn.ReLU())
                in_dim = hidden_dim
            layers.append(nn.Linear(in_dim, action_dim))

        self.network = nn.Sequential(*layers)
        self.log_z = nn.Parameter(torch.zeros(()))

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        if state.dim() == 1:
            state = state.unsqueeze(0)
        logits = self.network(state)
        return logits.squeeze(0)


class GFlowNetAgent:
    """Sampling helper with epsilon-greedy or temperature-scaled exploration."""

    def __init__(
        self,
        model: GFlowNetModel,
        temperature: float = 1.0,
        epsilon: float = 0.0,
        device: torch.device | None = None,
    ) -> None:
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if not 0.0 <= epsilon <= 1.0:
            raise ValueError("epsilon must be within [0, 1]")

        self.model = model
        self.temperature = float(temperature)
        self.epsilon = float(epsilon)
        self.device = device or next(model.parameters()).device

    def _mask_tensor(self, mask: Sequence[bool]) -> torch.Tensor:
        return torch.tensor(mask, dtype=torch.bool, device=self.device)

    def _masked_logits(self, logits: torch.Tensor, mask_tensor: torch.Tensor) -> torch.Tensor:
        return logits.masked_fill(~mask_tensor, -1e9)

    def compute_log_probs(
        self, env: Env, state: State, mask: Sequence[bool]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        state_tensor = encode_state(env, state, device=self.device)
        logits = self.model(state_tensor)
        mask_tensor = self._mask_tensor(mask)
        if mask_tensor.sum().item() == 0:
            raise RuntimeError("No valid actions available for sampling")
        masked_logits = self._masked_logits(logits, mask_tensor)
        log_probs = torch.log_softmax(masked_logits / self.temperature, dim=-1)
        return logits, masked_logits, log_probs

    def sample_action(
        self, env: Env, state: State, mask: Sequence[bool]
    ) -> Tuple[int, torch.Tensor, torch.Tensor]:
        _, masked_logits, log_probs = self.compute_log_probs(env, state, mask)
        if random.random() < self.epsilon:
            valid_indices = (
                torch.tensor(mask, dtype=torch.bool, device=self.device)
                .nonzero(as_tuple=False)
                .view(-1)
                .tolist()
            )
            action = int(random.choice(valid_indices))
        else:
            dist = torch.distributions.Categorical(logits=masked_logits / self.temperature)
            action = int(dist.sample().item())
        return action, log_probs[action], log_probs
