"""
count_agent_params.py
---------------------
Inspect, tabulate, and compare parameter counts for any AbstractAgent.

Usage examples
--------------
# ── List parameters for a single agent ──────────────────────────────────────
python scripts/count_agent_params.py

# The script instantiates ActionDiffusionAgent as a self-contained demo.
# To plug in your own agent, edit the `build_agents()` function at the bottom.

Output
------
• Per-module parameter table (trainable / frozen / total for every submodule)
• Leaf-layer detail table (optional, controlled by SHOW_LEAF_DETAIL flag)
• Summary footer with grand totals
• Side-by-side diff when two agents are compared
"""

from __future__ import annotations

import argparse
import sys
from collections import OrderedDict
from itertools import zip_longest
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn


# ─────────────────────────────────────────────────────────────────────────────
# Core utilities
# ─────────────────────────────────────────────────────────────────────────────

def _param_counts(module: nn.Module) -> Tuple[int, int]:
    """Return (trainable_params, frozen_params) for `module` (all descendants included)."""
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in module.parameters() if not p.requires_grad)
    return trainable, frozen


def _direct_param_counts(module: nn.Module) -> Tuple[int, int]:
    """Return param counts for parameters registered *directly* on `module` (no children)."""
    trainable = sum(p.numel() for p in module._parameters.values() if p is not None and p.requires_grad)
    frozen = sum(p.numel() for p in module._parameters.values() if p is not None and not p.requires_grad)
    return trainable, frozen


def build_param_table(
    model: nn.Module,
    max_depth: int = 3,
) -> List[Dict]:
    """
    Walk the module tree up to `max_depth` and collect param stats for each node.

    Returns a list of dicts with keys:
        path, module_type, trainable, frozen, total, depth
    """
    rows: List[Dict] = []

    def _walk(mod: nn.Module, path: str, depth: int) -> None:
        if depth > max_depth:
            return
        trainable, frozen = _param_counts(mod)
        rows.append(
            {
                "path": path if path else "(root)",
                "module_type": type(mod).__name__,
                "trainable": trainable,
                "frozen": frozen,
                "total": trainable + frozen,
                "depth": depth,
            }
        )
        for name, child in mod.named_children():
            child_path = f"{path}.{name}" if path else name
            _walk(child, child_path, depth + 1)

    _walk(model, "", 0)
    return rows


def build_leaf_table(model: nn.Module) -> List[Dict]:
    """
    Return one row per *leaf* module (no children) that owns at least one parameter.
    """
    rows: List[Dict] = []
    for path, mod in model.named_modules():
        if len(list(mod.children())) == 0:  # leaf
            trainable, frozen = _direct_param_counts(mod)
            if trainable + frozen == 0:
                continue
            rows.append(
                {
                    "path": path,
                    "module_type": type(mod).__name__,
                    "trainable": trainable,
                    "frozen": frozen,
                    "total": trainable + frozen,
                }
            )
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Formatting helpers
# ─────────────────────────────────────────────────────────────────────────────

def _fmt(n: int) -> str:
    """Format an integer with thousands separators."""
    return f"{n:,}"


def _pct(part: int, total: int) -> str:
    if total == 0:
        return "  0.0 %"
    return f"{100 * part / total:5.1f} %"


COL_SEP = "  "

def _print_table(
    rows: List[Dict],
    title: str,
    indent_by_depth: bool = True,
) -> None:
    """Pretty-print a parameter table."""
    # ── column widths ────────────────────────────────────────────────────────
    path_w = max(len(r["path"]) + (r.get("depth", 0) * 2 if indent_by_depth else 0) for r in rows)
    path_w = max(path_w, 6)
    type_w = max(len(r["module_type"]) for r in rows)
    type_w = max(type_w, 11)
    num_w = 14  # wide enough for "1,234,567,890"

    header = (
        f"{'Module':<{path_w}}{COL_SEP}"
        f"{'Type':<{type_w}}{COL_SEP}"
        f"{'Trainable':>{num_w}}{COL_SEP}"
        f"{'Frozen':>{num_w}}{COL_SEP}"
        f"{'Total':>{num_w}}{COL_SEP}"
        f"{'Train %':>8}"
    )
    sep = "─" * len(header)

    print(f"\n{'═' * len(header)}")
    print(f"  {title}")
    print(f"{'═' * len(header)}")
    print(header)
    print(sep)

    for r in rows:
        indent = "  " * r.get("depth", 0) if indent_by_depth else ""
        padded_path = indent + r["path"]
        trainable, frozen, total = r["trainable"], r["frozen"], r["total"]
        print(
            f"{padded_path:<{path_w}}{COL_SEP}"
            f"{r['module_type']:<{type_w}}{COL_SEP}"
            f"{_fmt(trainable):>{num_w}}{COL_SEP}"
            f"{_fmt(frozen):>{num_w}}{COL_SEP}"
            f"{_fmt(total):>{num_w}}{COL_SEP}"
            f"{_pct(trainable, total):>8}"
        )

    print(sep)

    # ── footer totals ────────────────────────────────────────────────────────
    root = rows[0]  # first row is the root module
    t, f, tot = root["trainable"], root["frozen"], root["total"]
    print(
        f"{'TOTAL':<{path_w}}{COL_SEP}"
        f"{''::<{type_w}}{COL_SEP}"
        f"{_fmt(t):>{num_w}}{COL_SEP}"
        f"{_fmt(f):>{num_w}}{COL_SEP}"
        f"{_fmt(tot):>{num_w}}{COL_SEP}"
        f"{_pct(t, tot):>8}"
    )
    print(f"{'═' * len(header)}\n")


def print_agent_params(
    model: nn.Module,
    name: str = "Agent",
    max_depth: int = 3,
    show_leaf_detail: bool = False,
) -> None:
    """
    Full parameter report for a single model.

    Args:
        model: Any nn.Module (e.g. an AbstractAgent or its inner model).
        name: Display name used in table titles.
        max_depth: How many levels of the module tree to expand.
        show_leaf_detail: If True, also print a flat table of every leaf layer.
    """
    rows = build_param_table(model, max_depth=max_depth)
    _print_table(rows, title=f"Parameter breakdown — {name}", indent_by_depth=True)

    if show_leaf_detail:
        leaf_rows = build_leaf_table(model)
        _print_table(leaf_rows, title=f"Leaf-layer detail — {name}", indent_by_depth=False)


# ─────────────────────────────────────────────────────────────────────────────
# Side-by-side diff
# ─────────────────────────────────────────────────────────────────────────────

def compare_agents(
    model_a: nn.Module,
    name_a: str,
    model_b: nn.Module,
    name_b: str,
    max_depth: int = 3,
) -> None:
    """
    Print a side-by-side diff of the top-level submodule param counts for two models.
    Rows where both totals match are highlighted with '='; differences with '≠'.
    """
    def _top_level(model: nn.Module) -> "OrderedDict[str, Dict]":
        result: OrderedDict = OrderedDict()
        for name, child in model.named_children():
            t, f = _param_counts(child)
            result[name] = {"type": type(child).__name__, "trainable": t, "frozen": f, "total": t + f}
        # add root entry
        t, f = _param_counts(model)
        result["(root total)"] = {"type": type(model).__name__, "trainable": t, "frozen": f, "total": t + f}
        return result

    a_stats = _top_level(model_a)
    b_stats = _top_level(model_b)

    all_keys = list(OrderedDict.fromkeys(list(a_stats.keys()) + list(b_stats.keys())))

    col_w = 14
    key_w = max(max(len(k) for k in all_keys), 10)
    type_w = 20

    sep_half = "─" * (key_w + type_w + col_w * 2 + len(COL_SEP) * 4)
    header_line = (
        f"{'Module':<{key_w}}{COL_SEP}"
        f"{'':>{2}}{COL_SEP}"  # diff marker
        f"{name_a[:type_w + col_w * 2]:^{type_w + col_w * 2}}{COL_SEP}"
        f"{name_b[:type_w + col_w * 2]:^{type_w + col_w * 2}}"
    )
    sub_header = (
        f"{'':>{key_w}}{COL_SEP}"
        f"{'':>{2}}{COL_SEP}"
        f"{'Type':<{type_w}}{COL_SEP}{'Total':>{col_w}}{COL_SEP}"
        f"{'Type':<{type_w}}{COL_SEP}{'Total':>{col_w}}"
    )

    full_sep = "═" * len(sub_header)
    print(f"\n{full_sep}")
    print(f"  Side-by-side comparison: {name_a}  vs  {name_b}")
    print(full_sep)
    print(header_line)
    print(sub_header)
    print("─" * len(sub_header))

    for key in all_keys:
        a = a_stats.get(key)
        b = b_stats.get(key)

        a_type = a["type"] if a else "—"
        a_total = _fmt(a["total"]) if a else "—"
        b_type = b["type"] if b else "—"
        b_total = _fmt(b["total"]) if b else "—"

        a_val = a["total"] if a else None
        b_val = b["total"] if b else None
        marker = "=" if a_val == b_val else "≠"

        print(
            f"{key:<{key_w}}{COL_SEP}"
            f"{marker:>{2}}{COL_SEP}"
            f"{a_type:<{type_w}}{COL_SEP}{a_total:>{col_w}}{COL_SEP}"
            f"{b_type:<{type_w}}{COL_SEP}{b_total:>{col_w}}"
        )

    print(f"{'═' * len(sub_header)}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Demo — ActionDiffusionAgent
# ─────────────────────────────────────────────────────────────────────────────

def build_demo_agents() -> List[Tuple[nn.Module, str]]:
    """
    Build one or more agents for the demo run.
    Edit this function to swap in your own agents.
    Returns a list of (model, display_name) tuples.
    """
    import sys
    import os

    # Make the navsim package importable when running from the repo root
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    from navsim.agents.action_diffusion_agent.ad_config import ActionDiffusionConfig
    from navsim.agents.action_diffusion_agent.ad_agent import ActionDiffusionAgent

    # ── Agent A: default config (frozen ResNet-50 backbone) ─────────────────
    cfg_a = ActionDiffusionConfig(
        timm_model_name="resnet50",
        freeze_backbone=True,
        timm_pretrained=False,   # skip hub download in demo
    )
    agent_a = ActionDiffusionAgent(config=cfg_a)
    agent_a.initialize()

    # ── Agent B: unfrozen, smaller backbone — change freely ─────────────────
    cfg_b = ActionDiffusionConfig(
        timm_model_name="resnet50",
        freeze_backbone=False,
        timm_pretrained=False,
    )
    agent_b = ActionDiffusionAgent(config=cfg_b)
    agent_b.initialize()

    return [
        (agent_a, "AD-Agent (frozen backbone)"),
        (agent_b, "AD-Agent (unfrozen backbone)"),
    ]


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="List and compare parameter counts for AbstractAgent subclasses.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--depth", type=int, default=3,
        help="How many levels of the module hierarchy to expand in the table.",
    )
    p.add_argument(
        "--leaf", action="store_true",
        help="Also print a flat table of every leaf layer.",
    )
    p.add_argument(
        "--no-compare", action="store_true",
        help="Skip the side-by-side comparison when multiple agents are built.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    agents = build_demo_agents()

    # ── Per-agent report ─────────────────────────────────────────────────────
    for model, name in agents:
        print_agent_params(
            model,
            name=name,
            max_depth=args.depth,
            show_leaf_detail=args.leaf,
        )

    # ── Side-by-side comparison (first two agents) ───────────────────────────
    if len(agents) >= 2 and not args.no_compare:
        compare_agents(
            agents[0][0], agents[0][1],
            agents[1][0], agents[1][1],
        )


if __name__ == "__main__":
    main()
