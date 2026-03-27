from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Set, Tuple


@dataclass(frozen=True)
class MISState:
    """Compact immutable state for memoization-friendly DAG traversal."""

    selected_mask: int
    forbidden_mask: int
    halted: bool
    set_size: int


class MISDAGEnv:
    """
    Gym-like combinatorial environment for Maximum Independent Set construction.

    Forward actions monotonically add one node to the independent set. A dedicated
    STOP action marks a terminal state.
    """

    def __init__(self, adjacency: Dict[int, Iterable[int]]):
        if not adjacency:
            raise ValueError("adjacency must be non-empty")

        node_ids = sorted(adjacency.keys())
        if node_ids != list(range(len(node_ids))):
            raise ValueError("adjacency keys must be contiguous node ids: 0..n-1")

        self.num_nodes = len(node_ids)
        self.stop_action = self.num_nodes
        self.action_dim = self.num_nodes + 1

        self.adjacency_masks: List[int] = [0] * self.num_nodes
        for node in range(self.num_nodes):
            neighbors: Set[int] = set(adjacency[node])
            if node in neighbors:
                raise ValueError("self-loops are not supported for MIS environment")
            for nbr in neighbors:
                if nbr < 0 or nbr >= self.num_nodes:
                    raise ValueError("adjacency contains out-of-range neighbor id")
                self.adjacency_masks[node] |= 1 << nbr

        self.selected_mask: int
        self.forbidden_mask: int
        self.halted: bool
        self.set_size: int
        self.reset()

    def reset(self) -> MISState:
        self.selected_mask = 0
        self.forbidden_mask = 0
        self.halted = False
        self.set_size = 0
        return self.get_state()

    def get_state(self) -> MISState:
        return MISState(
            selected_mask=self.selected_mask,
            forbidden_mask=self.forbidden_mask,
            halted=self.halted,
            set_size=self.set_size,
        )

    def _recompute_forbidden_mask(self) -> None:
        selected = self.selected_mask
        forbidden = 0
        while selected:
            lsb = selected & -selected
            node = lsb.bit_length() - 1
            forbidden |= self.adjacency_masks[node]
            selected ^= lsb
        self.forbidden_mask = forbidden

    def get_mask(self) -> List[bool]:
        """
        Forward legal-action mask (length = action_dim).
        `True` means action is legal; `False` means action is illegal.
        """
        mask = [False] * self.action_dim
        if self.halted:
            return mask

        mask[self.stop_action] = True
        unavailable = self.selected_mask | self.forbidden_mask
        for node in range(self.num_nodes):
            if not (unavailable & (1 << node)):
                mask[node] = True
        return mask

    def get_backward_mask(self) -> List[bool]:
        """
        Backward legal-action mask over the same action space.
        `True` means action is legal; `False` means action is illegal.
        - If halted, only STOP is legal (to unhalt).
        - Otherwise, selected nodes can be removed.
        """
        mask = [False] * self.action_dim
        if self.halted:
            mask[self.stop_action] = True
            return mask

        selected = self.selected_mask
        while selected:
            lsb = selected & -selected
            node = lsb.bit_length() - 1
            mask[node] = True
            selected ^= lsb
        return mask

    def step(self, action: int) -> Tuple[MISState, float, bool, Dict[str, object]]:
        """Apply one forward action and return (state, reward, done, info)."""
        if self.halted:
            raise RuntimeError("Cannot call step() from a terminal state; call reset().")
        mask = self.get_mask()
        if action < 0 or action >= self.action_dim or not mask[action]:
            raise ValueError("Invalid forward action for current state")

        if action == self.stop_action:
            self.halted = True
            return self.get_state(), 0.0, True, {"action": "STOP"}

        node_bit = 1 << action
        self.selected_mask |= node_bit
        self.forbidden_mask |= self.adjacency_masks[action]
        self.set_size += 1
        return self.get_state(), 0.0, False, {"action": action}

    def backward_step(self, action: int) -> Tuple[MISState, float, bool, Dict[str, object]]:
        """Reverse one legal action and return (state, reward, done, info)."""
        mask = self.get_backward_mask()
        if action < 0 or action >= self.action_dim or not mask[action]:
            raise ValueError("Invalid backward action for current state")

        if action == self.stop_action:
            self.halted = False
            return self.get_state(), 0.0, False, {"action": "UNSTOP"}

        node_bit = 1 << action
        self.selected_mask &= ~node_bit
        self.set_size -= 1
        self._recompute_forbidden_mask()
        self.halted = False
        return self.get_state(), 0.0, False, {"action": action}
