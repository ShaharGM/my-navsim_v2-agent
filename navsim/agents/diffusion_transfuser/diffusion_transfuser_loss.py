from typing import Dict

import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from navsim.agents.diffusion_transfuser.diffusion_transfuser_config import DiffusionTransfuserConfig
from navsim.agents.diffusion_transfuser.diffusion_transfuser_model import DiffusionTransfuserModel
from navsim.agents.transfuser.transfuser_features import BoundingBox2DIndex


def diffusion_transfuser_loss(targets: Dict[str, torch.Tensor], predictions: Dict[str, torch.Tensor], config: DiffusionTransfuserConfig, model: DiffusionTransfuserModel):
    """
    Helper function calculating complete loss of Diffusion Transfuser.
    Combines diffusion training loss (for trajectory generation) with auxiliary losses.
    
    :param targets: dictionary of target tensors
    :param predictions: dictionary of prediction tensors  
    :param config: global Diffusion Transfuser config
    :param model: the diffusion transfuser model
    :return: combined loss value
    """

    # trajectory_loss = F.l1_loss(predictions["trajectory"], targets["trajectory"])
    agent_class_loss, agent_box_loss = _agent_loss(targets, predictions, config)
    bev_semantic_loss = F.cross_entropy(predictions["bev_semantic_map"], targets["bev_semantic_map"].long())
    # loss = (
    #     config.trajectory_weight * trajectory_loss
    #     + config.agent_class_weight * agent_class_loss
    #     + config.agent_box_weight * agent_box_loss
    #     + config.bev_semantic_weight * bev_semantic_loss
    # )
    loss = (
        config.agent_class_weight * agent_class_loss
        + config.agent_box_weight * agent_box_loss
        + config.bev_semantic_weight * bev_semantic_loss
    )
    return loss


def _agent_loss(targets: Dict[str, torch.Tensor], predictions: Dict[str, torch.Tensor], config: DiffusionTransfuserConfig):
    """
    Hungarian matching loss for agent detection
    :param targets: dictionary of name tensor pairings
    :param predictions: dictionary of name tensor pairings
    :param config: global Diffusion Transfuser config
    :return: detection loss
    """

    gt_states, gt_valid = targets["agent_states"], targets["agent_labels"]
    pred_states, pred_logits = predictions["agent_states"], predictions["agent_labels"]

    if config.latent:
        rad_to_ego = torch.arctan2(
            gt_states[..., BoundingBox2DIndex.Y],
            gt_states[..., BoundingBox2DIndex.X],
        )

        in_latent_rad_thresh = torch.logical_and(
            rad_to_ego > -config.latent_rad_thresh,
            rad_to_ego < config.latent_rad_thresh,
        )

        gt_valid = torch.logical_and(gt_valid, in_latent_rad_thresh)

    # classification loss
    class_loss = F.binary_cross_entropy_with_logits(pred_logits, gt_valid.float())

    # regression loss
    pred_states = pred_states[gt_valid.bool()]
    gt_states = gt_states[gt_valid.bool()]

    if len(pred_states) == 0:
        box_loss = torch.tensor(0.0, device=pred_logits.device)
    else:
        box_loss = F.l1_loss(pred_states, gt_states)

    return class_loss, box_loss