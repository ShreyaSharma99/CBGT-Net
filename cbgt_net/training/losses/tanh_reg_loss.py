# -*- coding: utf-8 -*-
"""
.. module:: entropy_loss.py
   :platform: Linux, Windows, OSX
   :synopsis: Loss function that aims to maximize entropy of a categorical
              distribution

.. moduleauthor:: AARTLab <danahugh@andrew.cmu.edu>

An implementation of a max entropy loss function. 
"""

import tensorflow as tf
from ..datatypes import LossBatch

class TanhRegularLoss:
	"""
	EntropyLoss generates a loss that aims to maximize the entropy of a
	categorical distribution.
	"""

	def __init__(self, **kwargs):
		"""
		Keyword Arguments
		-----------------
		attribute_name : string, default="evidence"
			Attribute whose entropy should be maximized
		"""

		self._entropy_attribute_name = kwargs.get("attribute_name", "evidence")
		# self._is_a2c = kwargs.get("is_a2c", 0)
		# self._tanh_activation = kwargs.get("tanh_activation", 0)
		# self.softmax_layer = tf.keras.layers.Softmax()


	def __repr__(self):
		"""
		Produce a string representation of the reward, including the class name
		and the important attributes.
		"""

		return f'{self.__class__.__name__}'


	@tf.function
	def __call__(self, episode, model=None, attribute_name="losses"):
		"""
		Create a tensor calculating the max entropy loss for the attribute 
		identified in the episode

		Arguments
		---------
		episode : dict
			Dictionary of episode data
		"""

		# Get the attribute to calculate the entropy for
		attribute = episode.evidence
		# attribute = episode[self._entropy_attribute_name]

		# tanh_reg_loss = tf.reduce_mean(tf.reduce_sum(tf.pow(attribute, 4), axis=2)*tf.squeeze(1.0 - tf.cast(episode.decision_masks, tf.float32)), axis=0)
		tanh_reg_loss = tf.reduce_mean(tf.reduce_sum(tf.math.pow(attribute, 2), axis=2)*tf.squeeze(1.0 - tf.cast(episode.decision_masks, tf.float32)), axis=0)
		
		# # Add the loss to the episode, if an attribute name has been provided
		# if attribute_name is not None and attribute_name != "":
		# 	episode.add_attribute(attribute_name, tanh_reg_loss)

		# return tanh_reg_loss
		return LossBatch(tanh_reg_loss)
