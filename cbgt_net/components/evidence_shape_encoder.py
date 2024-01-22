# -*- coding: utf-8 -*-
"""
.. module:: evidence_image_encoder
    :platform: Linux, Windows, OSX
    :synopsis: Definition of an image evidence encoder module.

.. moduleauthor:: AARTLab <danahugh@andrew.cmu.edu>

The EvidenceImageEncoder is a simple image encoder to map a patch of image of dim (patch_size_1 X patch_size_2 x 3) to a 
(latent_dim X 1) evidence vectors.

Requirements
------------
Tensorflow 2.8
"""

import tensorflow as tf
from tensorflow import keras

import logging

class EvidenceShapeEncoder(keras.layers.Layer):
    """
    A simple shapes image encoder to map an image of shape of dim (32 x 32 x 3) 
    to a (n_categories X 1) evidence vectors.


    Attributes
    ----------
    patch_size : [int, int]
        Size of the sampled image patch
    latent_dim : int
        Latent dimension to which we want to encode the image patch
    """

    def __init__(self, num_categories, **kwargs):
        """
        Arguments
        ---------
        num_categories : int
            Number of unique categories / choices for the task

        input_shape : list of int
            Shape of input patch image

        Keyword Arguments
        -----------------
        name : string, default="EvidenceMLPModule"
            String name for this block
        num_hidden_units : int, default=25
            Number of hidden units in the hidden layer of the MLP
        hidden_activation : string, default="tanh"
            Activation function for the hidden layer
        """

        super(EvidenceShapeEncoder, self).__init__()

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
        self._batch_size = None
        self._input_shape = kwargs.get("input_shape", (32, 32, 3))

        # self._patch_size = kwargs.get("patch_size", [20, 20])
        self._latent_dim = kwargs.get("latent_dim", 64)
        self._hidden_activation = kwargs.get("hidden_activation", "relu")

        ##model building
        # self._encoder = tf.keras.Sequential()
        # self._encoder.add(keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=self._input_shape))
        # self._encoder.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
        # self._encoder.add(keras.layers.MaxPooling2D((2, 2)))
        # self._encoder.add(keras.layers.Dropout(0.25))
        # self._encoder.add(keras.layers.Flatten())
        # self._encoder.add(keras.layers.Dense(128, activation='relu'))
        # self._encoder.add(keras.layers.Dropout(0.25))
        # self._encoder.add(keras.layers.Dense(self._num_categories, activation='softmax'))

        # LeNet Architecture
        self._encoder = tf.keras.Sequential()
        self._encoder.add(keras.layers.Conv2D(6, kernel_size=(5, 5), activation='relu', input_shape=self._input_shape))
        self._encoder.add(keras.layers.MaxPooling2D((2, 2)))
        self._encoder.add(keras.layers.Conv2D(16, (5, 5), activation='relu'))
        self._encoder.add(keras.layers.MaxPooling2D((2, 2)))
        self._encoder.add(keras.layers.Flatten())
        self._encoder.add(keras.layers.Dense(120, activation='relu'))
        self._encoder.add(keras.layers.Dense(84, activation='relu'))
        self._encoder.add(keras.layers.Dense(self._num_categories, activation='softmax'))

        # self.scratch_model = tf.keras.models.load_model('/home/shreya/cbgt_net-develop/cbgt_net/pretrained_shape_10_32x32/')
        # self._output_softmax = tf.keras.layers.Softmax()

        # # Construct the MLP
        # self._encoder = tf.keras.Sequential([keras.layers.Flatten(), 
        #     keras.layers.Dense(self._latent_dim, activation=self._hidden_activation),
        # ])

        # self._hidden_to_output = keras.layers.Dense(self._num_categories)
        # self._output_softmax = keras.layers.Softmax()
        

    @property
    def num_categories(self):
        return self._num_categories
    
    @property
    def latent_dim(self):
        return self._latent_dim

    @property
    def hidden_activation(self):
        return self._hidden_activation
    

    def __str__(self):
        """
        String representation of the block
        """

        return self._name


    def reset(self):
        """
        Does nothing, as there's no internal state to reset in this module
        """

        # Indicate that the batch size is no longer known
        self._batch_size = None


    def build(self, input_shape):
        """
        Construct the Encoder, called upon the initial call to `call`

        Arguments
        ---------
        input_shape
            Shape of the initial input
        """

        self._input_shape = input_shape


    def call(self, observation):
        """
        Perform a forward pass of the evidence block with the given observaiton
        as input.  Returns an encoded vector in the form of [latent_dim, 1]
        

        Arguments
        ---------
        observation : tf.Tensor
            Observation tensor

        Returns
        -------
        tf.Tensor
            Evidence vector
        """

        # Figure out what the batch size is, if needed
        if self._batch_size is None:
            self._batch_size = observation.get_shape()[0]


        # Run the observation through the enoder layer
        # print("observation.shape ", observation.shape)

        output = self._encoder(observation) 
        return output

        # self.scratch_model.trainable = True
        # output = self.scratch_model(observation)
        # output = self._output_softmax(output)

        # # print("Encoder output - ", output)
        # return output