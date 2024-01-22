# -*- coding: utf-8 -*-
"""
.. module:: evidence_mlp
    :platform: Linux, Windows, OSX
    :synopsis: Definition of a simple MLP evidence module.

.. moduleauthor:: AARTLab <danahugh@andrew.cmu.edu>

The EvidenceMLPModule is a simple class defining an MLP to map observations to
evidence vectors.

Requirements
------------
Tensorflow 2.8
"""

import tensorflow as tf
from tensorflow import keras

import logging

class EvidenceMLPModule(keras.layers.Layer):
    """
    A simple evidence module that uses an MLP to map observations to an
    evidence vector.  The evidence vector is a distribution over possible
    categories.

    Attributes
    ----------
    num_categories : int
        Number of categories / evidence vector size
    """

    def __init__(self, num_categories, **kwargs):
        """
        Arguments
        ---------
        num_categories : int
            Number of unique categories / choices for the task

        Keyword Arguments
        -----------------
        name : string, default="EvidenceMLPModule"
            String name for this block
        num_hidden_units : int, default=25
            Number of hidden units in the hidden layer of the MLP
        hidden_activation : string, default="tanh"
            Activation function for the hidden layer
        """

        super(EvidenceMLPModule, self).__init__()

        # Get the name of this block and a logger for it
        self._name = kwargs.get("name", self.__class__.__name__)
        self._logger = logging.getLogger(self._name)
        self._logger.setLevel(kwargs.get("log_level", logging.WARNING))

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


        # Store the details needed to construct the block
        self._batch_size = kwargs.get("batch_size", None)
        self._input_shape = None
        self._num_hidden_units = kwargs.get("num_hidden_units", 25)
        self._hidden_activation = kwargs.get("hidden_activation", "tanh")

        # Construct the MLP
        self._input_to_hidden = keras.layers.Dense(self._num_hidden_units, activation=self._hidden_activation)
        self._hidden_to_output = keras.layers.Dense(self._num_categories)
        self._output_softmax = keras.layers.Softmax()


    @property
    def num_categories(self):
        return self._num_categories


    @property
    def output_shape(self):
        return (self._num_categories,)
    

    def __str__(self):
        """
        String representation of the block
        """

        return self._name


    def reset(self):
        """
        Does nothing, as there's no internal state to reset in this module
        """

        pass


    def build(self, input_shape):
        """
        Construct the MLP, called upon the initial call to `call`

        Arguments
        ---------
        input_shape
            Shape of the initial input
        """

        self._input_shape = input_shape
#        self._batch_size = input_shape[0]


    def call(self, observation):
        """
        Perform a forward pass of the evidence block with the given observaiton
        as input.  Returns an evidence vector in the form of distribution over
        

        Arguments
        ---------
        observation : tf.Tensor
            Observation tensor

        Returns
        -------
        tf.Tensor
            Evidence vector
        """

#        # Figure out what the batch size is, if needed
#        if self._batch_size is None:
#            self._batch_size = observation.get_shape()[0]

        # Run the observation through the MLP
        hidden = self._input_to_hidden(observation)
        output = self._hidden_to_output(hidden)
        return self._output_softmax(output)
