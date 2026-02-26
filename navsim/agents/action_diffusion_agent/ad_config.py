"""
ActionDiffusionConfig — standalone dataclass configuration for the
ActionDiffusionAgent.

Each section is labelled so the fields' purpose is immediately obvious.
All fields have sensible defaults; only `pretrained_ckpt` (and `vov_ckpt`
when using the VoV backbone) typically need to be overridden from the
Hydra YAML.
"""
from dataclasses import dataclass, field

from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling


@dataclass
class ActionDiffusionConfig:
    # -------------------------------------------------------------------
    # Trajectory output
    # -------------------------------------------------------------------
    # Sparse trajectory returned by the agent (used by the NAVSIM evaluator)
    trajectory_sampling: TrajectorySampling = field(
        default_factory=lambda: TrajectorySampling(time_horizon=4, interval_length=0.5)
    )

    # -------------------------------------------------------------------
    # Camera input
    # -------------------------------------------------------------------
    # Resolution of the stitched panoramic images fed to the backbone.
    camera_width: int = 2048
    camera_height: int = 512
    # Number of consecutive historical frames to load (most recent used for
    # the backbone; older frames are only kept in history for the physics model).
    seq_len: int = 1
    # Include the rear stitched view (l2 + b0 + r2) in addition to the front.
    use_back_view: bool = True

    # -------------------------------------------------------------------
    # Perception backbone
    # -------------------------------------------------------------------
    # Select the visual backbone:
    #   'timm'  →  any model identifier accepted by `timm.create_model`,
    #               e.g. "resnet50", "convnext_small_in22ft1k",
    #                    "vit_base_patch16_224"            (default)
    #   'vov'   →  VoVNet V-99-eSE (used in the GTRS pipeline); requires
    #               a pretrained checkpoint via `vov_ckpt`
    #   'bev'   →  BEVBackbone: VoVNet + cross-attention BEV fusion.
    #               Always consumes front + back cameras internally and
    #               returns 64 BEV tokens (8×8 query grid) at 1024 channels.
    #               Requires `vov_ckpt`.  Optionally set `bev_ckpt` to load
    #               the full backbone state (fusion transformer, BEV queries).
    backbone_type: str = "timm"
    timm_model_name: str = "resnet50"
    timm_pretrained: bool = True          # load ImageNet weights via timm hub
    vov_ckpt: str = ""                    # path to VoV checkpoint (vov + bev)
    bev_ckpt: str = ""                    # path to full BEVBackbone checkpoint (bev only)

    # When True all parameters inside self.backbone are frozen.
    # Only the ego-status encoder, positional embedding, and diffusion head
    # are updated.  When False every parameter is trainable.
    freeze_backbone: bool = True

    # -------------------------------------------------------------------
    # BEV backbone settings  (only used when backbone_type='bev')
    # -------------------------------------------------------------------
    bev_fusion_layers: int = 3        # TransformerEncoder depth

    # -------------------------------------------------------------------
    # Spatial pooling — defines how many image tokens are produced
    # -------------------------------------------------------------------
    # Used by ALL backbone types.  The backbone's last feature map is
    # pooled to (img_vert_anchors × img_horz_anchors) spatial positions.
    #   • timm / vov  — these become the output tokens directly.
    #   • bev         — these are the intermediate image tokens fed INTO
    #                   the fusion transformer; the BEV output is always
    #                   8×8 = 64 tokens (controlled by BEVBackbone._BEV_H/W).
    # Defaults match gtrs_dp.ckpt: pos_emb shape = 2×(16×64)+64 = 2112.
    img_vert_anchors: int = 16
    img_horz_anchors: int = 64

    # -------------------------------------------------------------------
    # Core model dimensions
    # -------------------------------------------------------------------
    model_dim: int = 256      # embedding / projection dimension (D)
    ffn_dim: int = 1024       # feed-forward inner dimension in the transformer
    num_heads: int = 8        # attention heads in the denoising transformer

    # -------------------------------------------------------------------
    # Ego-status encoding
    # -------------------------------------------------------------------
    # Default layout (matches the feature builder):
    #   driving_command (4) + ego_velocity (2) + ego_acceleration (2) = 8
    ego_status_dim: int = 8

    # -------------------------------------------------------------------
    # Diffusion head
    # -------------------------------------------------------------------
    num_diffusion_layers: int = 4      # TransformerDecoder layers in denoiser
    num_diffusion_timesteps: int = 100  # DDPM training timesteps
    # Number of independent trajectory proposals generated at inference time.
    num_inference_proposals: int = 64

    # -------------------------------------------------------------------
    # Internal action / trajectory representation
    # (unicycle kinematic model)
    # -------------------------------------------------------------------
    # The diffusion head operates in action space (accel, curvature) with a
    # fine-grained 0.1 s time step that integrates to the full 4 s horizon.
    internal_dt: float = 0.1          # integration dt [s]
    internal_horizon: int = 40        # steps = time_horizon / internal_dt
    action_dim: int = 2               # (acceleration, curvature)

    # Z-score normalisation statistics for actions.
    # Computed from the mini training split; adjust if re-training on
    # a different dataset splits.
    accel_mean: float = 0.05671
    accel_std: float = 1.12148
    curv_mean: float = 0.00180
    curv_std: float = 0.02297

    # -------------------------------------------------------------------
    # Pretrained checkpoint loading
    # -------------------------------------------------------------------
    # When non-empty, load *backbone + projection* weights from this file
    # (stripping "agent.model." / "model." prefixes) and freeze them.
    # The diffusion head is always trained from scratch.
    # Set to "" to train everything end-to-end from random (or timm)
    # initialisation.
    pretrained_ckpt: str = ""

    # -------------------------------------------------------------------
    # Noise scheduler type
    # -------------------------------------------------------------------
    # 'ddpm' : DDPM Markov-chain noise schedule (epsilon-prediction, original)
    # 'flow' : Flow matching — linear probability paths, velocity prediction,
    #          continuous time t ∈ [0, 1].  Training and inference are both
    #          rewritten; the denoising transformer architecture is unchanged.
    noise_type: str = "ddpm"
    # Number of Euler integration steps used at inference time when
    # noise_type='flow'.  Has no effect for noise_type='ddpm'.
    # Fewer steps = faster inference; 20 is a good default.
    num_flow_steps: int = 20

    # -------------------------------------------------------------------
    # Training hyper-parameters
    # -------------------------------------------------------------------
    weight_decay: float = 0.0
    dp_loss_weight: float = 10.0
