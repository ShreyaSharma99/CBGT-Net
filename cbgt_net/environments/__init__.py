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
from .vectorize import Vectorize
from .mnist_categorical import MNISTCategoricalEnvironment
from .cifar_categorical import CIFAR_CategoricalEnvironment

__all__ = ['SimpleCategoricalEnvironment', 'Vectorize', 'MNISTCategoricalEnvironment', 'CIFAR_CategoricalEnvironment']