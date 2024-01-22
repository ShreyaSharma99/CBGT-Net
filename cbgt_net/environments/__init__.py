# -*- coding: utf-8 -*-
"""
.. module:: cbgt_net.environments
   :platform: Linux, Windows, OSX
   :synopsis: Environments used to evaluate CBGT_Net architectures

.. moduleauthor:: AARTLab <danahugh@andrew.cmu.edu>

This module defines a set of environments used to evaluate the CBGT_Net
architecture.
"""

from .environment import Environment
from .simple_categorical import SimpleCategoricalEnvironment
from .shape_categorical import ShapeCategoricalEnvironment
from .voc_objectDetection import VOC_ObjDetect
from .vectorized_voc_environment import Vectorized_VOC_ObjDetect
from .mnist_largeData_patch_environment import MNIST_LargeData_Patch_Environment
from .vectorize import Vectorize
from .mnist_categorical import MNISTCategoricalEnvironment
from .mnist_categorical_padded import MNISTCategoricalEnvironmentPadded
from .cifar_categorical import CIFAR_CategoricalEnvironment
from .cifar_categorical_time_parallel import CIFAR_CategoricalEnvironment_Parallel
from .miniworld_bldg_fire_categorical import Miniworld_bldg_fire_CategoricalEnvironment

__all__ = ['SimpleCategoricalEnvironment', 'ShapeCategoricalEnvironment', 'VOC_ObjDetect', 'Vectorize', 'Vectorized_VOC_ObjDetect', 
         'MNIST_LargeData_Patch_Environment', 'MNISTCategoricalEnvironment', 'MNISTCategoricalEnvironmentPadded',
         'CIFAR_CategoricalEnvironment', 'CIFAR_CategoricalEnvironment_Parallel',
         'Miniworld_bldg_fire_CategoricalEnvironment']