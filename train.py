from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Union

import torch
from torch.optim import AdamW

from gflownet_agent import GFlowNetAgent, GFlowNetModel, encode_state, infer_state_dim
from mis_env import MISDAGEnv, MISState
from sorting_network_env import SortingNetworkDAGEnv, SortingNetworkState
from tb_loss import trajectory_balance_loss, uniform_backward_log_prob

Env = Union[MISDAGEnv, SortingNetworkDAGEnv]
State = Union[MISState, SortingNetworkState]


@dataclass
class Trajectory:
    states: List[State]
    actions: List[int]
    reward: float
    forward_log_probs: List[torch.Tensor]
    backward_log_probs: List[torch.Tensor]
    flow_values: List[float]


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def build_mis_env(num_nodes: int, edge_prob: float, rng: random.Random) -> MISDAGEnv:
    adjacency: Dict[int, List[int]] = {node: [] for node in range(num_nodes)}
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if rng.random() < edge_prob:
                adjacency[i].append(j)
                adjacency[j].append(i)
    return MISDAGEnv(adjacency)


def compute_reward(env: Env, state: State, reward_sharpness: float) -> float:
    if isinstance(env, MISDAGEnv) and isinstance(state, MISState):
        return float((state.set_size + 1) ** reward_sharpness)
    if isinstance(env, SortingNetworkDAGEnv) and isinstance(state, SortingNetworkState):
        return float((state.num_comparators + 1) ** reward_sharpness)
    raise TypeError("Unsupported environment/state pair for reward computation")


def compute_flow_value(
    agent: GFlowNetAgent,
    env: Env,
    state: State,
    mask: Sequence[bool],
    terminal_reward: float | None = None,
) -> float:
    mask_tensor = torch.tensor(mask, dtype=torch.bool, device=agent.device)
    if mask_tensor.sum().item() == 0:
        return float(terminal_reward or 0.0)
    state_tensor = encode_state(env, state, device=agent.device)
    logits = agent.model(state_tensor)
    masked_logits = logits.masked_fill(~mask_tensor, -1e9)
    log_flow = torch.logsumexp(masked_logits, dim=-1)
    return float(torch.exp(log_flow).item())


def collect_trajectory(
    env: Env,
    agent: GFlowNetAgent,
    reward_sharpness: float,
    max_steps: int | None = None,
    record_flow_values: bool = False,
) -> Trajectory:
    state = env.reset()
    done = False

    states = [state]
    actions: List[int] = []
    forward_log_probs: List[torch.Tensor] = []
    backward_log_probs: List[torch.Tensor] = []
    flow_values: List[float] = []

    mask = env.get_mask()
    if record_flow_values:
        flow_values.append(compute_flow_value(agent, env, state, mask))

    steps = 0
    while not done:
        mask = env.get_mask()
        if max_steps is not None and steps >= max_steps:
            action = env.stop_action
            _, _, log_probs = agent.compute_log_probs(env, state, mask)
            log_prob = log_probs[action]
        else:
            action, log_prob, _ = agent.sample_action(env, state, mask)

        next_state, _, done, _ = env.step(action)
        backward_mask = env.get_backward_mask()
        backward_mask_tensor = torch.tensor(
            backward_mask, dtype=torch.float32, device=agent.device
        )

        states.append(next_state)
        actions.append(action)
        forward_log_probs.append(log_prob)
        backward_log_probs.append(uniform_backward_log_prob(backward_mask_tensor))

        if record_flow_values:
            next_mask = env.get_mask()
            if done:
                flow_values.append(0.0)
            else:
                flow_values.append(compute_flow_value(agent, env, next_state, next_mask))

        state = next_state
        steps += 1

    reward = compute_reward(env, state, reward_sharpness)
    if record_flow_values and flow_values:
        terminal_mask = env.get_mask()
        flow_values[-1] = compute_flow_value(
            agent, env, state, terminal_mask, terminal_reward=reward
        )

    return Trajectory(
        states=states,
        actions=actions,
        reward=reward,
        forward_log_probs=forward_log_probs,
        backward_log_probs=backward_log_probs,
        flow_values=flow_values,
    )


def export_results(
    path: Path,
    temperature: float,
    reward_sharpness: float,
    trajectories: List[Trajectory],
    env: Env,
) -> None:
    nodes: List[Dict[str, object]] = []
    edges: List[Dict[str, object]] = []
    flow_values: Dict[str, float] = {}
    rewards: List[float] = []

    for traj_idx, traj in enumerate(trajectories):
        rewards.append(traj.reward)
        for step_idx, state in enumerate(traj.states):
            node_id = f"t{traj_idx}_s{step_idx}"
            if isinstance(env, MISDAGEnv) and isinstance(state, MISState):
                label = f"set={state.set_size}"
            elif isinstance(env, SortingNetworkDAGEnv) and isinstance(state, SortingNetworkState):
                label = f"comps={state.num_comparators}"
            else:
                label = "state"

            nodes.append(
                {
                    "id": node_id,
                    "label": label,
                    "terminal": bool(getattr(state, "halted", False)),
                }
            )

            if traj.flow_values:
                flow_values[node_id] = float(traj.flow_values[step_idx])

        for step_idx, action in enumerate(traj.actions):
            edges.append(
                {
                    "source": f"t{traj_idx}_s{step_idx}",
                    "target": f"t{traj_idx}_s{step_idx + 1}",
                    "action": int(action),
                }
            )

    payload = {
        "temperature": float(temperature),
        "reward_sharpness": float(reward_sharpness),
        "nodes": nodes,
        "edges": edges,
        "flow_values": flow_values,
        "rewards": rewards,
    }
    path.write_text(json.dumps(payload, indent=2))


def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)

    rng = random.Random(args.seed)
    if args.env == "mis":
        env = build_mis_env(args.num_nodes, args.edge_prob, rng)
    else:
        env = SortingNetworkDAGEnv(args.n_wires, args.max_layers, args.max_comparators)

    state_dim = infer_state_dim(env)
    model = GFlowNetModel(state_dim, env.action_dim, args.hidden_dim)
    agent = GFlowNetAgent(model, temperature=args.temperature, epsilon=args.epsilon)
    optimizer = AdamW(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses: List[torch.Tensor] = []
        for _ in range(args.batch_size):
            traj = collect_trajectory(
                env,
                agent,
                reward_sharpness=args.reward_sharpness,
                max_steps=args.max_steps,
            )
            loss = trajectory_balance_loss(
                model.log_z, traj.forward_log_probs, traj.backward_log_probs, traj.reward
            )
            losses.append(loss)

        batch_loss = torch.stack(losses).mean()
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()

        if epoch % args.export_interval == 0:
            model.eval()
            validation_agent = GFlowNetAgent(
                model, temperature=args.temperature, epsilon=0.0
            )
            with torch.no_grad():
                trajectories = [
                    collect_trajectory(
                        env,
                        validation_agent,
                        reward_sharpness=args.reward_sharpness,
                        max_steps=args.max_steps,
                        record_flow_values=True,
                    )
                    for _ in range(args.validation_trajectories)
                ]
            export_results(
                Path(args.results_path),
                args.temperature,
                args.reward_sharpness,
                trajectories,
                env,
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a GFlowNet with TB loss")
    parser.add_argument("--env", choices=["mis", "sorting"], default="mis")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--reward-sharpness", type=float, default=1.4)
    parser.add_argument("--export-interval", type=int, default=25)
    parser.add_argument("--validation-trajectories", type=int, default=6)
    parser.add_argument("--results-path", type=str, default="results.json")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=17)

    parser.add_argument("--num-nodes", type=int, default=8)
    parser.add_argument("--edge-prob", type=float, default=0.25)

    parser.add_argument("--n-wires", type=int, default=6)
    parser.add_argument("--max-layers", type=int, default=5)
    parser.add_argument("--max-comparators", type=int, default=12)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
