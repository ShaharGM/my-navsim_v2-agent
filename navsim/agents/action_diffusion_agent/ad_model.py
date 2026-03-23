"""
ActionDiffusionModel — main nn.Module that wires together:

    BackboneBase subclass   →  img_proj  →  image tokens (B, N_img, D)
    STATUS ENCODER                        →  status token (B,   1,   D)
    concatenate + positional bias         →  context KV   (B, N_img+1, D)
    ActionDiffusionHead                   →  trajectories / loss

The active backbone is selected by `config.backbone_type`:
  'timm' / 'vov'  —  PoolBackbone (CNN + adaptive pooling)
  'bev'           —  BEVBackbone  (VoVNet + cross-attention BEV fusion)

The backbone owns all image extraction, so the model is agnostic to
how many cameras are used or how tokens are produced.

Freezing
--------
`freeze_backbone=True` freezes only the backbone encoder.
All other parameters (img_proj, status_encoder, diffusion_head) are always
trainable.

Checkpoint loading
------------------
When `pretrained_ckpt` is set, weights are loaded for any key that matches
(diffusion_head is skipped so it always trains from scratch).  Freezing is
applied afterwards based solely on `freeze_backbone`.
"""

from typing import Any, Callable, Dict, Optional, Tuple

import faiss
import torch
import torch.nn as nn
import torch.nn.functional as F

from navsim.agents.action_diffusion_agent.ad_config import ActionDiffusionConfig
from navsim.agents.action_diffusion_agent.backbones import build_backbone
from navsim.agents.action_diffusion_agent.ad_diffusion_head import ActionDiffusionHead


class _PerceptionTrajectoryMemory:
    """Offline nearest-neighbor memory bank for perception -> GT trajectory retrieval.

    Expected payload format (recommended):
        {
            <perception_key>: Tensor[N, D],
            <trajectory_key>: Tensor[N, T, 3],
        }
    where trajectory channels are (x, y, heading).
    """

    def __init__(self, config: ActionDiffusionConfig) -> None:
        if not config.nn_memory_path:
            raise ValueError("nn_memory_path must be set when use_nn_trajectory_context=True")

        self._metric = config.nn_memory_metric.lower()
        if self._metric not in {"cosine", "l2"}:
            raise ValueError(
                f"Unsupported nn_memory_metric={config.nn_memory_metric!r}; expected 'cosine' or 'l2'."
            )

        payload = torch.load(config.nn_memory_path, map_location="cpu", weights_only=False)
        perception, trajectories = self._extract(
            payload,
            config.nn_memory_perception_key,
            config.nn_memory_trajectory_key,
        )
        perception = perception.float().contiguous().cpu()
        trajectories = self._format_trajectories(
            trajectories,
            config.nn_trajectory_steps,
            config.nn_trajectory_dim,
        ).float().contiguous().cpu()

        if perception.ndim != 2:
            raise ValueError(
                f"Perception tensor must be rank-2 [M, C], got shape {tuple(perception.shape)}"
            )
        if perception.shape[0] != trajectories.shape[0]:
            raise ValueError(
                "Perception and trajectory bank sizes must match; got "
                f"{perception.shape[0]} vs {trajectories.shape[0]}."
            )

        self._perception = perception
        self._trajectories = trajectories
        self._use_faiss = config.use_faiss_retrieval
        self._faiss_index = None

        if self._use_faiss:
            # For cosine: normalize vectors so L2 distance approximates cosine distance.
            if self._metric == "cosine":
                perception_for_index = F.normalize(perception, dim=-1).numpy()
            else:
                perception_for_index = perception.numpy()

            # FAISS HNSW is CPU-native; keep this path intentionally simple.
            dim = self._perception.shape[1]
            self._faiss_index = faiss.IndexHNSWFlat(dim, 16)
            if hasattr(self._faiss_index, "hnsw") and hasattr(self._faiss_index.hnsw, "efSearch"):
                self._faiss_index.hnsw.efSearch = 40
            elif hasattr(self._faiss_index, "efSearch"):
                self._faiss_index.efSearch = 40

            self._faiss_index.add(perception_for_index)
            print(
                "[_PerceptionTrajectoryMemory] "
                f"FAISS CPU HNSW index built, metric={self._metric}, "
                f"size={len(self._perception)}"
            )
        else:
            # Pre-compute for brute-force methods
            if self._metric == "cosine":
                self._perception_norm = F.normalize(self._perception, dim=-1)
                self._perception_sq = None
            else:
                self._perception_norm = None
                self._perception_sq = (self._perception ** 2).sum(dim=-1)
            print(
                "[_PerceptionTrajectoryMemory] "
                f"Brute-force retrieval setup, metric={self._metric}, "
                f"size={len(self._perception)}"
            )

    @staticmethod
    def _extract(
        payload: Any,
        perception_key: str,
        trajectory_key: str,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if isinstance(payload, dict):
            if perception_key in payload and trajectory_key in payload:
                return torch.as_tensor(payload[perception_key]), torch.as_tensor(payload[trajectory_key])
            raise KeyError(
                "Retrieval payload must contain configured bank keys "
                f"{perception_key!r} and {trajectory_key!r}."
            )

        if isinstance(payload, (list, tuple)) and len(payload) == 2:
            return torch.as_tensor(payload[0]), torch.as_tensor(payload[1])

        raise TypeError(
            "Unsupported retrieval bank format. Expected either: "
            "dict with configured keys, or 2-item tuple/list "
            "(perception_bank, trajectory_bank)."
        )

    @staticmethod
    def _format_trajectories(
        trajectories: torch.Tensor,
        steps: int,
        dim: int,
    ) -> torch.Tensor:
        if trajectories.ndim == 3:
            if tuple(trajectories.shape[1:]) != (steps, dim):
                raise ValueError(
                    f"Trajectory shape mismatch: expected [M, {steps}, {dim}], got {tuple(trajectories.shape)}"
                )
            if dim != 3:
                raise ValueError(
                    f"Expected trajectory dim=3 for (x, y, heading), got dim={dim}."
                )
            return trajectories
        if trajectories.ndim == 2:
            if trajectories.shape[1] != steps * dim:
                raise ValueError(
                    f"Flattened trajectory dim mismatch: expected {steps * dim}, got {trajectories.shape[1]}"
                )
            return trajectories.view(-1, steps, dim)
        raise ValueError(
            "Trajectory tensor must be rank-2 [M, steps*dim] or rank-3 [M, steps, dim], "
            f"got {tuple(trajectories.shape)}"
        )

    @torch.no_grad()
    def query(self, perception_query: torch.Tensor, k: int = 1) -> torch.Tensor:
        query = perception_query.detach().float().cpu()
        if query.ndim != 2:
            raise ValueError(
                f"Perception query must be rank-2 [B, C], got shape {tuple(query.shape)}"
            )
        if query.shape[1] != self._perception.shape[1]:
            raise ValueError(
                "Perception query dim mismatch: "
                f"query C={query.shape[1]}, bank C={self._perception.shape[1]}."
            )

        if self._use_faiss:
            # FAISS search
            if self._metric == "cosine":
                query_for_search = F.normalize(query, dim=-1).numpy()
            else:
                query_for_search = query.numpy()
            _, nn_idx = self._faiss_index.search(query_for_search, k=k)
        else:
            # Brute-force search
            if self._metric == "cosine":
                query_norm = F.normalize(query, dim=-1)
                similarities = query_norm @ self._perception_norm.t()  # (B, M)
                _, nn_idx = similarities.topk(k, dim=-1)
            else:
                q_sq = (query ** 2).sum(dim=-1, keepdim=True)
                dist = q_sq + self._perception_sq.unsqueeze(0) - 2.0 * (query @ self._perception.t())
                _, nn_idx = dist.topk(k, dim=-1, largest=False)

        return self._trajectories[nn_idx]  # (B, k, T, 3)


class ActionDiffusionModel(nn.Module):

    def __init__(self, config: ActionDiffusionConfig) -> None:
        super().__init__()
        self.config = config
        self._use_nn_trajectory_context = bool(config.use_nn_trajectory_context)

        # ── Perception backbone ───────────────────────────────────────────────
        self.backbone = build_backbone(config)

        # num_tokens already accounts for back-view doubling (PoolBackbone)
        # or BEV grid size (BEVBackbone) — no manual adjustment needed here.
        num_img_tokens = self.backbone.num_tokens

        # ── Feature projection: backbone channels → model_dim ────────────────
        self.img_proj = nn.Linear(self.backbone.out_channels, config.model_dim)

        # ── Ego-status encoder ────────────────────────────────────────────────
        # [driving_command(4), vx(1), vy(1), ax(1), ay(1)] → (B, 1, D)
        self.status_encoder = nn.Sequential(
            nn.Linear(config.ego_status_dim, config.model_dim),
            nn.LayerNorm(config.model_dim),
        )

        self._nn_memory: Optional[_PerceptionTrajectoryMemory] = None
        if self._use_nn_trajectory_context:
            self._nn_memory = _PerceptionTrajectoryMemory(config)
            self.nn_traj_encoder = nn.Sequential(
                nn.Linear(config.nn_trajectory_steps * config.nn_trajectory_dim, config.model_dim),
                nn.LayerNorm(config.model_dim),
            )

        # ── Learnable positional embedding ────────────────────────────────────
        extra_tokens = 1 if self._use_nn_trajectory_context else 0
        self._context_len = num_img_tokens + 1 + extra_tokens
        self.pos_embedding = nn.Embedding(self._context_len, config.model_dim)

        # ── Diffusion head ────────────────────────────────────────────────────
        self.diffusion_head = ActionDiffusionHead(
            config=config, context_len=self._context_len
        )

        # ── Optional weight loading ───────────────────────────────────────────
        if config.pretrained_ckpt:
            self._load_pretrained(config.pretrained_ckpt)

        # ── Backbone freezing (independent of checkpoint) ─────────────────────
        if config.freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            print("[ActionDiffusionModel] Backbone frozen.")

    # ─────────────────────────────────────────────── checkpoint loading ───────

    def _load_pretrained(self, ckpt_path: str) -> None:
        """Load matching weights from a checkpoint. Diffusion head is always skipped."""
        print(f"[ActionDiffusionModel] Loading pretrained weights from {ckpt_path} …")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        raw_state: dict = ckpt.get("state_dict", ckpt)

        cleaned: dict = {}
        for k, v in raw_state.items():
            name = k.replace("agent.model.", "").replace("model.", "")
            if "diffusion_head" in name:
                continue
            cleaned[name] = v

        msg = self.load_state_dict(cleaned, strict=False)
        print(
            f"[ActionDiffusionModel] Loaded. "
            f"Missing: {len(msg.missing_keys)}, Unexpected: {len(msg.unexpected_keys)}"
        )

    # ─────────────────────────────────────────────── train() override ────────

    def train(self, mode: bool = True) -> "ActionDiffusionModel":
        """Keep the frozen backbone in eval mode so BN stats don't drift."""
        super().train(mode)
        if mode and self.config.freeze_backbone:
            self.backbone.eval()
        return self

    # ────────────────────────────────────────────────── context builder ───────

    def _build_context(
        self, features: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Build the (B, context_len, D) context KV tensor consumed by the
        diffusion head.

        Steps:
          1. Backbone extracts images from the features dict and returns tokens.
          2. Project backbone tokens to model_dim.
          3. Encode ego-status as a single token.
          4. Concatenate image tokens + status token.
          5. Add learnable positional bias.
        """
        cfg = self.config

        # The backbone owns all image extraction; it receives the full
        # features dict and returns (B, num_tokens, out_channels).
        backbone_tokens = self.backbone(features)                 # (B, N, C)
        tokens = self.img_proj(backbone_tokens)                   # (B, N, D)

        # --- Ego-status token ---
        status = features["status_feature"][:, : cfg.ego_status_dim]  # (B, 8)
        status_tok = self.status_encoder(status).unsqueeze(1)          # (B, 1, D)

        context_tokens = [tokens, status_tok]

        if self._use_nn_trajectory_context:
            assert self._nn_memory is not None
            perception_vec = backbone_tokens.mean(dim=1)                  # (B, C)
            nn_traj_topk = self._nn_memory.query(perception_vec).to(
                device=tokens.device,
                dtype=tokens.dtype,
            )                                                             # (B, k, S, 3)
            nn_traj = nn_traj_topk[:, 0]                                  # (B, S, 3)
            nn_tok = self.nn_traj_encoder(nn_traj.flatten(start_dim=1)).unsqueeze(1)
            context_tokens.append(nn_tok)                                 # (B, 1, D)

        # --- Assemble context ---
        context = torch.cat(context_tokens, dim=1)                # (B, N+1(+1), D)
        context = context + self.pos_embedding.weight[None, :]    # broadcast positional bias
        return context

    # ──────────────────────────────────────────────────────── forward ────────

    def forward(
        self,
        features: Dict[str, torch.Tensor],
        scorer_guidance_fn: Optional[Callable[[torch.Tensor, int, int], torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Full forward pass.

        Training
        --------
        Returns {'context_kv': (B, L, D), 'trajectory': zeros placeholder}.
        The actual diffusion loss is computed separately in ActionDiffusionAgent
        via `diffusion_head.compute_loss`.

        Inference
        ---------
        Returns {
            'context_kv':   (B, L, D),
            'dp_pred':      (B, N, 8, 3),   – N trajectory proposals
            'pred_actions': (B, N, 40, 2),
            'trajectory':   (B, 8, 3),      – one randomly selected proposal
        }
        """
        B = features["status_feature"].shape[0]
        device = features["status_feature"].device

        context = self._build_context(features)  # (B, L, D)
        out: Dict[str, torch.Tensor] = {"context_kv": context}

        if self.training:
            # During training the caller computes the loss externally.
            # Return a zero placeholder so the framework doesn't break.
            out["trajectory"] = torch.zeros(B, 8, 3, device=device)
        else:
            head_out = self.diffusion_head(
                context,
                features,
                scorer_guidance_fn=scorer_guidance_fn,
            )
            out.update(head_out)

            # Pick one proposal at random as the "committed" trajectory.
            N = head_out["dp_pred"].shape[1]
            idx = torch.randint(0, N, (1,)).item()
            out["trajectory"] = head_out["dp_pred"][:, idx]  # (B, 8, 3)

        return out
