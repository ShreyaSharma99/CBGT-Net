# -*- coding: utf-8 -*-
"""
.. module:: training.measures
   :platform: Linux, Windows, OSX
   :synopsis: Submodule containing measures functions

.. moduleauthor:: AARTLab <danahugh@andrew.cmu.edu>

The measures submodule contains classes to calculate various measure for 
evaluation of episodes and/or models after each training step.
"""

from .accuracy_measures import AccuracyMeasure

__all__ = ["AccuracyMeasure"]
