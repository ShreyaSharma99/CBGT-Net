# -*- coding: utf-8 -*-
"""
.. module:: cbgt_net
   :platform: Linux, Windows, OSX
   :synopsis: Definition of model and algorithms for CGBT-Net.

.. moduleauthor:: AARTLab <danahugh@andrew.cmu.edu>

This top-level module defines the model, components, and training algorithms
for a neural network architecture inspired by the Cortico-Basal 
Ganglia-Thalamic (CBGT) loop in the human brain.
"""

from importlib.metadata import version, PackageNotFoundError

from .components import CBGT_Net
from .experiment import Experiment

# Get the version number from the package install
try:
	__version__ = version(__name__)
except PackageNotFoundError:
	__version__ = "0.0.0"

__all__ = ['CBGT_Net', 'Experiment']