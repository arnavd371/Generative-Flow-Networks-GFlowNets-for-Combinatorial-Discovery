from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass(frozen=True)
class SortingNetworkState:
    """Compact immutable state for memoization-friendly DAG traversal."""

    layer_masks: Tuple[int, ...]
    halted: bool
    num_comparators: int


class SortingNetworkDAGEnv:
    """
    Gym-like combinatorial environment for sorting-network construction on a DAG.

    Forward actions monotonically add one comparator into one layer, represented as
    bitsets for low-overhead masking and transitions. A dedicated STOP action marks
    a terminal state.
    """

    def __init__(self, n_wires: int, max_layers: int, max_comparators: int):
        if n_wires < 2:
            raise ValueError("n_wires must be >= 2")
        if max_layers < 1:
            raise ValueError("max_layers must be >= 1")
        if max_comparators < 1:
            raise ValueError("max_comparators must be >= 1")

        self.n_wires = n_wires
        self.max_layers = max_layers
        self.max_comparators = max_comparators

        self.comparators: List[Tuple[int, int]] = []
        for i in range(n_wires):
            for j in range(i + 1, n_wires):
                self.comparators.append((i, j))

        self.num_comparator_types = len(self.comparators)
        self.num_non_stop_actions = self.max_layers * self.num_comparator_types
        self.stop_action = self.num_non_stop_actions
        self.action_dim = self.num_non_stop_actions + 1

        self._comparator_wire_masks = [
            (1 << i) | (1 << j) for i, j in self.comparators
        ]

        self.layer_masks: List[int]
        self.layer_wire_usage: List[int]
        self.halted: bool
        self.num_comparators: int
        self.reset()

    def reset(self) -> SortingNetworkState:
        self.layer_masks = [0] * self.max_layers
        self.layer_wire_usage = [0] * self.max_layers
        self.halted = False
        self.num_comparators = 0
        return self.get_state()

    def get_state(self) -> SortingNetworkState:
        return SortingNetworkState(
            layer_masks=tuple(self.layer_masks),
            halted=self.halted,
            num_comparators=self.num_comparators,
        )

    def _decode_action(self, action: int) -> Tuple[int, int]:
        if action < 0 or action >= self.num_non_stop_actions:
            raise ValueError("Action out of forward-comparator range")
        layer = action // self.num_comparator_types
        comp_idx = action % self.num_comparator_types
        return layer, comp_idx

    def _recompute_layer_wire_usage(self, layer: int) -> None:
        layer_mask = self.layer_masks[layer]
        used_wires = 0
        while layer_mask:
            lsb = layer_mask & -layer_mask
            comp_idx = lsb.bit_length() - 1
            used_wires |= self._comparator_wire_masks[comp_idx]
            layer_mask ^= lsb
        self.layer_wire_usage[layer] = used_wires

    def get_mask(self) -> List[bool]:
        """
        Forward legal-action mask (length = action_dim).
        `True` means action is legal; `False` means action is illegal.
        """

        mask = [False] * self.action_dim
        if self.halted:
            return mask

        # STOP always legal while not halted.
        mask[self.stop_action] = True

        if self.num_comparators >= self.max_comparators:
            return mask

        for layer in range(self.max_layers):
            layer_mask = self.layer_masks[layer]
            layer_wires = self.layer_wire_usage[layer]
            offset = layer * self.num_comparator_types
            for comp_idx in range(self.num_comparator_types):
                bit = 1 << comp_idx
                if layer_mask & bit:
                    continue
                if layer_wires & self._comparator_wire_masks[comp_idx]:
                    continue
                mask[offset + comp_idx] = True

        return mask

    def get_backward_mask(self) -> List[bool]:
        """
        Backward legal-action mask over the same action space.
        `True` means action is legal; `False` means action is illegal.
        - If halted, only STOP is legal (to unhalt).
        - Otherwise, only present comparators can be removed.
        """

        mask = [False] * self.action_dim
        if self.halted:
            mask[self.stop_action] = True
            return mask

        if self.num_comparators == 0:
            return mask

        for layer in range(self.max_layers):
            layer_mask = self.layer_masks[layer]
            if layer_mask == 0:
                continue
            offset = layer * self.num_comparator_types
            for comp_idx in range(self.num_comparator_types):
                if layer_mask & (1 << comp_idx):
                    mask[offset + comp_idx] = True

        return mask

    def step(self, action: int) -> Tuple[SortingNetworkState, float, bool, Dict[str, object]]:
        """Apply one forward action and return (state, reward, done, info)."""
        if self.halted:
            raise RuntimeError("Cannot call step() from a terminal state; call reset().")

        mask = self.get_mask()
        if action < 0 or action >= self.action_dim or not mask[action]:
            raise ValueError("Invalid forward action for current state")

        if action == self.stop_action:
            self.halted = True
            return self.get_state(), 0.0, True, {"action": "STOP"}

        layer, comp_idx = self._decode_action(action)
        bit = 1 << comp_idx
        self.layer_masks[layer] |= bit
        self.layer_wire_usage[layer] |= self._comparator_wire_masks[comp_idx]
        self.num_comparators += 1

        done = self.num_comparators >= self.max_comparators
        if done:
            self.halted = True
        return self.get_state(), 0.0, done, {"action": (layer, self.comparators[comp_idx])}

    def backward_step(
        self, action: int
    ) -> Tuple[SortingNetworkState, float, bool, Dict[str, object]]:
        """Reverse one legal action and return (state, reward, done, info)."""
        mask = self.get_backward_mask()
        if action < 0 or action >= self.action_dim or not mask[action]:
            raise ValueError("Invalid backward action for current state")

        if action == self.stop_action:
            self.halted = False
            return self.get_state(), 0.0, False, {"action": "UNSTOP"}

        layer, comp_idx = self._decode_action(action)
        bit = 1 << comp_idx
        self.layer_masks[layer] &= ~bit
        self._recompute_layer_wire_usage(layer)
        self.num_comparators -= 1
        self.halted = False
        return self.get_state(), 0.0, False, {"action": (layer, self.comparators[comp_idx])}
