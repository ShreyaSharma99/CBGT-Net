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

class ProbabDecisionThresholdModule(keras.layers.Layer):
    """
    A ProbabDecisionThresholdModule will indicate that a choice is to be made
    whenever an accumulator value passed through a dense layer give > 0.5 value.
    """

    def __init__(self, num_categories, **kwargs):
        """
        Keyword Arguments
        -----------------
        decision_threshold : int
            Minimum threshold needed for a decision to be made        
        name : string, default="FixedDecisionThresholdModule"
            String name for this module
        """

        super(ProbabDecisionThresholdModule, self).__init__()

        # Get the name of this block and a logger for it
        self._name = kwargs.get("name", self.__class__.__name__)
        self._logger = logging.getLogger(self._name)
        self._logger.setLevel(kwargs.get("log_level", logging.WARNING))

        try:
            self._num_categories = int(num_categories)
        except Exception as e:
            self.logger.error("%s: Number of categories cannot be cast as int: %s", self, str(num_categories))
            raise e

        if self._num_categories <= 0:
            self.logger.error("%s: Number of categories must be greater than 0: %d", self, self._num_categories)
            raise ValueError("Non-positive number of categories: %d" % self._num_categories)

        self._num_categories = num_categories

        # Check to see that the decision threshold is non-negative, and set to
        # zero if it is
        self._decision_threshold = kwargs.get("decision_threshold", 0)
        # self._encoder = tf.keras.Sequential(keras.layers.Dense(self._num_categories, activation='sigmoid'))
        self._encoder = tf.keras.Sequential(keras.layers.Dense(1, activation='sigmoid'))

        if self._decision_threshold < 0:
            self._logger.warning("%s:  Decision threshold less than zero.  Setting to zero." % self)

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

        # Indicate that the batch size is now unknown until the next call
        self._batch_size = None


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

        # Determine the batch size, if necessary
        if self._batch_size is None:
            self._batch_size = evidence.get_shape()[0]

        # print(N"Accumulator shape = ", accumulator.shape)
        # print("threshold_vector shape = ", threshold_vector.shape)
        # print("threshold = ", self.decision_threshold)

        threshold_vector = self._encoder(accumulator)
#        decision_gate = tf.greater(threshold_vector, 0.5)
        # decision_gate = tf.greater(tf.reduce_max(threshold_vector, axis=1), 0.5)
        # print("Threshold_vector = ", threshold_vector)
#        return tf.reshape(decision_gate, (-1,1)), threshold_vector
        return threshold_vector