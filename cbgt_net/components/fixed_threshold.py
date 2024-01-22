# -*- coding: utf-8 -*-
"""
.. module:: fixed_threshold
    :platform: Linux, Windows, OSX
    :synopsis: Definition of a fixed decision threshold module

.. moduleauthor:: AARTLab <danahugh@andrew.cmu.edu>

A fixed decision threshold module is used to determine whether a decision is
ready to be made solely on whether the current accumulator values exceed a
pre-defined threshold.


Requirements
------------
Tensorflow 2.8
"""

import tensorflow as tf
from tensorflow import keras

import numpy as np

import logging

class FixedDecisionThresholdModule(keras.layers.Layer):
    """
    A FixedDecisionThresholdModule will indicate that a choice is to be made
    whenever an accumulator value exceeds a given pre-defined threshold.
    """

    def __init__(self, **kwargs):
        """
        Keyword Arguments
        -----------------
        decision_threshold : int
            Minimum threshold needed for a decision to be made        
        name : string, default="FixedDecisionThresholdModule"
            String name for this module
        """

        super(FixedDecisionThresholdModule, self).__init__()

        # Get the name of this block and a logger for it
        self._name = kwargs.get("name", self.__class__.__name__)
        self._logger = logging.getLogger(self._name)
        self._logger.setLevel(kwargs.get("log_level", logging.WARNING))

        # Check to see that the decision threshold is non-negative, and set to
        # zero if it is
        self._decision_threshold = kwargs.get("decision_threshold", 0)

        if self._decision_threshold < 0:
            self._logger.warning("%s:  Decision threshold less than zero.  Setting to zero." % self)

        # Batch size will not be known until the first call
        self._batch_size = kwargs.get("batch_size", None)

        # Convert the decision threshold to a tf.constant
        self._decision_threshold = tf.constant(self._decision_threshold, dtype=tf.float32)


    @property
    def decision_threshold(self):
        return self._decision_threshold
    

    def __str__(self):
        """
        String representation of the block
        """

        return self._name


    def reset(self):
        """
        Does nothing
        """

        pass


    def call(self, evidence, accumulator):
        """
        Determine if a decision should be made given the provided evidence and
        accumulator vectors.

        Arguments
        ---------
        evidence : tf.Tensor
            Evidence tensor
        accumulator : tf.Tensor
            Accumulator values

        Returns
        -------
        tf.Tensor
            Boolean tensor indicating if a decision should be made
        """

        # Return 1.0 when any accumulated value exceeds the threshold, 0.0
        # otherwise.
        decision_gate = tf.greater(tf.reduce_max(accumulator, axis=1), self.decision_threshold)
        # decision_gate = tf.reduce_any(tf.greater(accumulator, self._decision_threshold), axis=1)
        decision_gate = tf.cast(decision_gate, tf.float32)

        return tf.reshape(decision_gate, (-1,1))