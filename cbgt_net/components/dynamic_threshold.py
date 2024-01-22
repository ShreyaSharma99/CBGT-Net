# -*- coding: utf-8 -*-
"""
.. module:: dynamic_threshold
    :platform: Linux, Windows, OSX
    :synopsis: Definition of a dynamic decision threshold module

.. moduleauthor:: AARTLab <danahugh@andrew.cmu.edu>

A dynamic decision threshold module allows for training of a single threshold
parameter.  The decision probability will be calculated as the sigmoid of the
difference between the maximum accumulator value, and the current decision
threshold.

Requirements
------------
Tensorflow 2.8
"""

import tensorflow as tf
from tensorflow import keras

import numpy as np

import logging

class DynamicDecisionThresholdModule(keras.layers.Layer):
    """
    A DynamicDecisionThresholdModule will indicate that a choice is to be made
    whenever an accumulator value - dynamic threshold > 0 value.
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

        super(DynamicDecisionThresholdModule, self).__init__()

        # Get the name of this block and a logger for it
        self._name = kwargs.get("name", self.__class__.__name__)
        self._logger = logging.getLogger(self._name)
        self._logger.setLevel(kwargs.get("log_level", logging.WARNING))

        # Check to see that the decision threshold is non-negative, and set to
        # zero if it is
        self._decision_threshold = tf.Variable(kwargs.get("decision_threshold", 0), 
                                               dtype=tf.float32, trainable=True)

        # Batch size will not be known until the first call
        self._batch_size = None


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
            Tensor indicating the likelihood of making a decision
        """

        # accumulator_max = tf.reduce_max(accumulator, axis=1)

        # decision_probability = tf.keras.activations.sigmoid(accumulator_max - self._decision_threshold)
        
        # return tf.reshape(decision_probability, (-1,1))

        threshold_vector = tf.keras.activations.sigmoid(accumulator - self._decision_threshold)
        # tf.reduce_max(accumulator, axis=1)

        decision_gate = tf.greater(tf.reduce_max(threshold_vector, axis=1), 0.5)
        decision_gate = tf.cast(decision_gate, tf.float32)
        
        return tf.reshape(decision_gate, (-1,1)), threshold_vector