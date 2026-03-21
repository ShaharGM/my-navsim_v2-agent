"""
Generic NN memory-bank query validation for ActionDiffusion retrieval.

This script validates that query results from _PerceptionTrajectoryMemory remain
consistent when you change bank generation or query logic.

Checks performed:
1) Random subset self-query check (query vectors sampled from the bank)
2) Optional full-bank self-query sweep
3) Optional perturbation robustness check

For each metric (cosine/l2), the script reports:
- Returned-trajectory consistency with computed nearest indices
- Source-index recovery rate (exact same row id recovered)
- Source-trajectory exact match rate

Usage examples:
  python test_nn_memory_query.py \
    --bank_path ../bev_navmini_perc_traj_pairs.pt \
    --samples 100 \
    --metrics cosine l2

  python test_nn_memory_query.py \
    --bank_path ../bev_navmini_perc_traj_pairs.pt \
    --full_sweep \
    --perturb_scale 1e-4
"""

import argparse
import random
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn.functional as F

from navsim.agents.action_diffusion_agent.ad_config import ActionDiffusionConfig
from navsim.agents.action_diffusion_agent.ad_model import _PerceptionTrajectoryMemory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate NN retrieval behavior on a memory bank.")
    parser.add_argument(
        "--bank_path",
        type=str,
        required=True,
        help="Path to .pt memory bank payload with perception_vectors and gt_trajectories.",
    )
    parser.add_argument(
        "--metrics",
        type=str,
        nargs="+",
        default=["cosine", "l2"],
        choices=["cosine", "l2"],
        help="Distance metrics to validate.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=100,
        help="Number of random bank rows to self-query.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=123,
        help="Random seed for row sampling and perturbation.",
    )
    parser.add_argument(
        "--full_sweep",
        action="store_true",
        help="Also run checks on every row in the bank.",
    )
    parser.add_argument(
        "--perturb_scale",
        type=float,
        default=0.0,
        help="If > 0, add gaussian noise to sampled query vectors with this std.",
    )
    return parser.parse_args()


def _compute_nn_indices(
    query: torch.Tensor,
    bank_perception: torch.Tensor,
    metric: str,
) -> torch.Tensor:
    if metric == "cosine":
        q = F.normalize(query, dim=-1)
        p = F.normalize(bank_perception, dim=-1)
        return (q @ p.t()).argmax(dim=-1)

    q_sq = (query ** 2).sum(dim=-1, keepdim=True)
    p_sq = (bank_perception ** 2).sum(dim=-1).unsqueeze(0)
    dist = q_sq + p_sq - 2.0 * (query @ bank_perception.t())
    return dist.argmin(dim=-1)


def _run_eval(
    memory: _PerceptionTrajectoryMemory,
    bank_perception: torch.Tensor,
    bank_trajectory: torch.Tensor,
    query: torch.Tensor,
    source_idx: torch.Tensor,
    metric: str,
) -> Dict[str, int]:
    out = memory.query(query)
    nn_idx = _compute_nn_indices(query, bank_perception, metric)
    expected_from_nn = bank_trajectory[nn_idx]
    expected_from_source = bank_trajectory[source_idx]

    # Sanity: returned trajectories should match trajectories indexed by NN ids.
    consistent = (out == expected_from_nn).all(dim=(1, 2))
    recovered_source_idx = nn_idx.eq(source_idx)
    recovered_source_traj = (out == expected_from_source).all(dim=(1, 2))

    return {
        "total": int(query.shape[0]),
        "consistent": int(consistent.sum().item()),
        "source_idx_match": int(recovered_source_idx.sum().item()),
        "source_traj_match": int(recovered_source_traj.sum().item()),
    }


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    bank_path = Path(args.bank_path).expanduser().resolve()
    if not bank_path.exists():
        raise FileNotFoundError(f"Bank file not found: {bank_path}")

    payload = torch.load(bank_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise TypeError("Bank payload must be a dict with perception/trajectory keys.")

    if "perception_vectors" not in payload or "gt_trajectories" not in payload:
        raise KeyError("Bank payload must include keys: perception_vectors, gt_trajectories")

    perception = payload["perception_vectors"].float().contiguous().cpu()
    trajectories = payload["gt_trajectories"].float().contiguous().cpu()

    if perception.ndim != 2:
        raise ValueError(f"Expected perception rank-2 [N, C], got {tuple(perception.shape)}")
    if trajectories.ndim != 3:
        raise ValueError(f"Expected trajectory rank-3 [N, T, D], got {tuple(trajectories.shape)}")
    if perception.shape[0] != trajectories.shape[0]:
        raise ValueError(
            f"Bank size mismatch: perception N={perception.shape[0]} vs trajectories N={trajectories.shape[0]}"
        )

    n_rows = int(perception.shape[0])
    k = min(max(args.samples, 1), n_rows)
    sample_ids = random.sample(range(n_rows), k)

    query = perception[sample_ids].clone()
    if args.perturb_scale > 0:
        query = query + args.perturb_scale * torch.randn_like(query)

    sample_ids_tensor = torch.tensor(sample_ids, dtype=torch.long)

    print(f"bank_path: {bank_path}")
    print(f"bank_perception_shape: {tuple(perception.shape)}")
    print(f"bank_trajectory_shape: {tuple(trajectories.shape)}")
    print(f"random_subset_size: {k}")
    print(f"perturb_scale: {args.perturb_scale}")

    for metric in args.metrics:
        cfg = ActionDiffusionConfig(
            use_nn_trajectory_context=True,
            nn_memory_path=str(bank_path),
            nn_memory_metric=metric,
            nn_trajectory_steps=int(trajectories.shape[1]),
            nn_trajectory_dim=int(trajectories.shape[2]),
        )
        memory = _PerceptionTrajectoryMemory(cfg)

        subset_stats = _run_eval(
            memory=memory,
            bank_perception=perception,
            bank_trajectory=trajectories,
            query=query,
            source_idx=sample_ids_tensor,
            metric=metric,
        )

        print(f"\n[{metric}] random subset")
        print(
            "  consistency_with_computed_nn: "
            f"{subset_stats['consistent']}/{subset_stats['total']}"
        )
        print(
            "  source_index_recovered: "
            f"{subset_stats['source_idx_match']}/{subset_stats['total']}"
        )
        print(
            "  source_trajectory_recovered: "
            f"{subset_stats['source_traj_match']}/{subset_stats['total']}"
        )

        if args.full_sweep:
            full_ids = torch.arange(n_rows, dtype=torch.long)
            full_query = perception.clone()
            full_stats = _run_eval(
                memory=memory,
                bank_perception=perception,
                bank_trajectory=trajectories,
                query=full_query,
                source_idx=full_ids,
                metric=metric,
            )
            print(f"[{metric}] full sweep")
            print(
                "  consistency_with_computed_nn: "
                f"{full_stats['consistent']}/{full_stats['total']}"
            )
            print(
                "  source_index_recovered: "
                f"{full_stats['source_idx_match']}/{full_stats['total']}"
            )
            print(
                "  source_trajectory_recovered: "
                f"{full_stats['source_traj_match']}/{full_stats['total']}"
            )


if __name__ == "__main__":
    main()
