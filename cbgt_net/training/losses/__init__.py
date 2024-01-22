# -*- coding: utf-8 -*-
"""
.. module:: training.losses
   :platform: Linux, Windows, OSX
   :synopsis: Submodule containing loss definitions for CBGT_Net experiments

.. moduleauthor:: AARTLab <danahugh@andrew.cmu.edu>

The training submodule contains implementations of loss functions for training
CBGT_Net instances.
"""

from .reinforce_loss import REINFORCE_Loss
from .composite_loss import CompositeLoss
from .entropy_loss import EntropyLoss
from .threshold_loss import ThresholdLoss
from .threshold_loss_Dana import ThresholdLossDana
from .tanh_reg_loss import TanhRegularLoss
from .tanh_reg_loss2 import TanhRegularLoss2
from .ce_loss import CE_Loss

__all__ = ["REINFORCE_Loss", "CompositeLoss", "EntropyLoss", "ThresholdLoss", "ThresholdLossDana", "TanhRegularLoss", "TanhRegularLoss2", "CE_Loss"]
