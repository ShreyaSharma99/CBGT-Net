# -*- coding: utf-8 -*-
"""
.. module:: training.datatypes
   :platform: Linux, Windows, OSX
   :synopsis: Collection of datatypes to use during training

.. moduleauthor:: AARTLab <danahugh@andrew.cmu.edu>

A simple collection of datatypes that can be used to summarize episodes during
training.
"""

from .episode_batch import EpisodeBatch
from .reward_batch import RewardBatch
from .loss_batch import LossBatch