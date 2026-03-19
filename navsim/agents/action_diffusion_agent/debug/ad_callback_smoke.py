import argparse
from pathlib import Path

import numpy as np

from navsim.agents.action_diffusion_agent.ad_callback import ActionDiffusionCallback
from navsim.agents.action_diffusion_agent.ad_config import ActionDiffusionConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke test for AD callback uncertainty plots.")
    parser.add_argument(
        "--mode",
        default="gaussian",
        choices=["gaussian", "kde", "boxplot"],
        help="Uncertainty visualization mode to render.",
    )
    args = parser.parse_args()

    config = ActionDiffusionConfig()
    callback = ActionDiffusionCallback(
        config=config,
        num_samples=3,
        uncertainty_viz_mode=args.mode,
    )

    batch_size = 3
    num_proposals = 96
    num_waypoints = 8

    # Build smooth GT trajectories with slight variation per sample.
    t = np.linspace(0.0, 1.0, num_waypoints)
    gt = np.zeros((batch_size, num_waypoints, 3), dtype=np.float32)
    for b in range(batch_size):
        gt[b, :, 0] = 2.0 + 25.0 * t + 0.8 * b  # longitudinal x
        gt[b, :, 1] = -2.0 + 4.0 * np.sin(1.4 * np.pi * t + 0.2 * b)  # lateral y

    # Add proposal noise that increases by waypoint index.
    rng = np.random.default_rng(123)
    pred = np.repeat(gt[:, None, :, :], num_proposals, axis=1)
    wp_std = np.linspace(0.2, 2.0, num_waypoints, dtype=np.float32)

    noise_x = rng.normal(loc=0.0, scale=wp_std[None, None, :], size=(batch_size, num_proposals, num_waypoints)).astype(np.float32)
    noise_y = rng.normal(loc=0.0, scale=(0.6 * wp_std)[None, None, :], size=(batch_size, num_proposals, num_waypoints)).astype(np.float32)
    pred[:, :, :, 0] += noise_x
    pred[:, :, :, 1] += noise_y

    fig = callback._plot_trajectories(gt_batch=gt, pred_batch=pred, epoch=0)

    script_dir = Path(__file__).parent
    out_path = script_dir / f"ad_callback_smoke_{args.mode}.png"
    fig.savefig(out_path, dpi=160)

    x_std = np.std(pred[0, :, :, 0], axis=0)
    y_std = np.std(pred[0, :, :, 1], axis=0)

    print(f"mode: {args.mode}")
    print(f"saved: {out_path}")
    print("sample-0 waypoint std longitudinal:", np.array2string(x_std, precision=3))
    print("sample-0 waypoint std lateral:", np.array2string(y_std, precision=3))


if __name__ == "__main__":
    main()
