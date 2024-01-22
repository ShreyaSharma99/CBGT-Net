# -*- coding: utf-8 -*-
"""
.. module:: training
   :platform: Linux, Windows, OSX
   :synopsis: Submodule containing reward definitions for CBGT_Net experiments

.. moduleauthor:: AARTLab <danahugh@andrew.cmu.edu>

The training submodule contains implementations of reward functions for training
CBGT_Net instances.
"""

from .simple_categorical_reward import SimpleCategoricalReward
from .composite_reward import CompositeReward

__all__ = ["SimpleCategoricalReward", "CompositeReward"]
