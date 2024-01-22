# -*- coding: utf-8 -*-
"""
.. module:: simple_accumulator
    :platform: Linux, Windows, OSX
    :synopsis: Definition of a simple accumulator module


.. moduleauthor:: AARTLab <danahugh@andrew.cmu.edu>


Requirements
------------
Tensorflow 2.8

"""

import tensorflow as tf
from tensorflow import keras

import numpy as np

import logging

class SimpleAccumulatorNewModule(keras.layers.Layer):
    """
    A simple module to accumulate evidence.
    """

    def __init__(self, num_categories, **kwargs):
        """
        Arguments
        ---------
        num_categories : int
            Number of unique categories for the task

        Keyword Arguments
        -----------------
        name : string, default="SimpleAccumulatorModule"
            String name for this module
        """

        super(SimpleAccumulatorNewModule, self).__init__()

        # Get the name of this block and a logger for it
        self._name = kwargs.get("name", self.__class__.__name__)
        self._logger = logging.getLogger(self._name)
        self._logger.setLevel(kwargs.get("log_level", logging.WARNING))

        # What is the expected batch size
        self._batch_size = kwargs.get("batch_size", 1)

        # Check that the number of categories is valid
        try:
            self._num_categories = int(num_categories)
        except Exception as e:
            self.logger.error("%s: Number of categories cannot be cast as int: %s", self, str(num_categories))
            raise e

        if self._num_categories <= 0:
            self.logger.error("%s: Number of categories must be greater than 0: %d", self, self._num_categories)
            raise ValueError("Non-positive number of categories: %d" % self._num_categories)

        self._num_categories = num_categories
        self._evidence_dim = kwargs.get("evidence_dim", 64)
        self._accumulator = tf.Variable(np.zeros((self._batch_size, self._num_categories)), 
                                        dtype=tf.float32, trainable=False, 
                                        name="%s/accumulator"%self)

        self._deconder = tf.keras.Sequential([keras.layers.Dense(self._num_categories, activation="softmax")])


    @property
    def num_categories(self):
        return self._num_categories


    @property
    def accumulator(self):
        return self._accumulator

    @property
    def accumulator_value(self):
        """
        Provide the current contents of the accumulator
        """
        return self._accumulator.value()
    

    def __str__(self):
        """
        String representation of the block
        """

        return self._name


    def reset(self):
        """
        Reset the accumulator value to zero
        """

        # Indicate that the batch size is no longer known, so that when the 
        # next call is made, a new accumulator vector will be constructed.e
        self._accumulator = tf.Variable(np.zeros((self._batch_size, self._num_categories)),
                                        dtype=tf.float32, trainable=False,
                                        name="%s/accumulator"%self)

        
        # self._decoder = tf.keras.Sequential([keras.layers.Dense(self._num_categories, activation="softmax")])
        # print("Batch size in simple accum set to- ", self._batch_size)
        # self._batch_size = None

#        self._accumulator.assign(np.zeros((1,self._num_categories)))


    def call(self, evidence, accumulator_values=None):
        """
        Run the provided evidence through the accumulator, and return the
        updated accumulator output.

        If no accumulator_values are provided, the module will use the current
        value of the internal accumulator, and update the accumulator value
        with the evidence vector.  Otherwise, a forward pass of the module will
        be performed without any internal update. 

        Arguments
        ---------
        evidence : tf.Tensor
            Evidence tensor
        accumulator_values : tf.Tensor, default=None
            Accumulator values

        Returns
        -------
        tf.Tensor
            Tensor containing the accumulated evidence
        """

        # If needed, determine the batch size and set the accumulator 
        if self._batch_size is None:
            self._batch_size = evidence.get_shape()[0]
            self._accumulator = tf.Variable(np.zeros((self._batch_size,self._num_categories)), 
                                            dtype=tf.float32, trainable=False, 
                                            name="%s/accumulator"%self)

        # Check if accumulator values were provided.  If so, the internal 
        # accumulator tensor should not be updated.
        if accumulator_values is not None:
            # Accumulator values were provided, so simply calculate the
            # accumulated evidence and do not update the internal value
            accumulated_evidence = evidence + accumulator_values

        else:
            # Accumulator values were not provided, so use the internal
            # accumulator value and update with evidence
            accumulated_evidence = self._deconder(evidence) + self._accumulator.value()
            self._accumulator.assign(accumulated_evidence)
        # print("accumulated_evidence - ", accumulated_evidence)
        return accumulated_evidence