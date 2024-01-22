# -*- coding: utf-8 -*-
"""
.. module:: cbgt_net.components
   :platform: Linux, Windows, OSX
   :synopsis: Definition of architecture components for CBGT_Net

.. moduleauthor:: AARTLab <danahugh@andrew.cmu.edu>

This module defines components used to construct a CBGT_Net, including the base
CBGT_Net model, and pre-defined layers for thresholding and accumulation.
"""

from .cbgt_net import CBGT_Net
from .evidence_mlp import EvidenceMLPModule
from .simple_accumulator import SimpleAccumulatorModule
from .fixed_decay_accumulator import FixedDecayAccumulatorModule
from .fixed_threshold import FixedDecisionThresholdModule
from .evidence_image_encoder import EvidenceImageEncoder
from .evidence_shape_encoder import EvidenceShapeEncoder
from .evidence_shape_patch_encoder import EvidenceShapePatchEncoder
from .probabilistic_threshold import ProbabDecisionThresholdModule
from .dynamic_threshold import DynamicDecisionThresholdModule
from .evidence_shape_patch_encoder_pretrained import EvidenceShapePatchEncoderPretrained
from .evidence_shape_patch_encoder_cifar import EvidenceShapePatchEncoderCIFAR
from .resnet import *

__all__ = ['CBGT_Net', 
           'EvidenceMLPModule', 
           'EvidenceShapeEncoder',
           'SimpleAccumulatorModule',
           'FixedDecayAccumulatorModule', 
           'FixedDecisionThresholdModule',
           'EvidenceImageEncoder',
           'EvidenceShapePatchEncoder',
           'EvidenceShapePatchEncoderPretrained',
           'ProbabDecisionThresholdModule',
           'DynamicDecisionThresholdModule',
           'EvidenceShapePatchEncoderCIFAR']
