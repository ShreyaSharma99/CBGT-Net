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
import numpy as np

class EntropyLoss:
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
		self._is_a2c = kwargs.get("is_a2c", 0)
		self._tanh_activation = kwargs.get("tanh_activation", 0)
		self.softmax_layer = tf.keras.layers.Softmax()


	def __repr__(self):
		"""
		Produce a string representation of the reward, including the class name
		and the important attributes.
		"""

		return f'{self.__class__.__name__}'


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
		entropy_attribute = episode[self._entropy_attribute_name]

		# # Calculate the mean entropy over the duration of the episode.
		# print("Attribute shape - ", tf.reduce_sum(attribute * tf.math.log(attribute+1e-32), axis=2).shape)
		# print("deision mask - ", episode.decision_masks.shape)
		# print("d_i = ", decision_index.shape)

		if self._tanh_activation:
			print("Applied SOftmax!! --------vff bfg  --------")
			attribute = self.softmax_layer(entropy_attribute)
		else:
			attribute = entropy_attribute

		####### Type 1 - all entropy terms till the point the decision is made
		if self._is_a2c:
			entropy = tf.reduce_sum(tf.reduce_sum(attribute * tf.math.log(attribute+1e-32), axis=2)*tf.squeeze(1.0 - tf.cast(episode.decision_masks, tf.float32)))/(episode.returns.shape[0]*episode.returns.shape[1])
		else:
			entropy = tf.reduce_sum(tf.reduce_sum(attribute * tf.math.log(attribute+1e-32), axis=2)*tf.squeeze(1.0 - tf.cast(episode.decision_masks, tf.float32)), axis=0)

		######## Type 2 - only at decision instance entropy term
		# decision_index = (tf.math.argmax(tf.squeeze(episode.decision_masks), axis=0) - 1).numpy().tolist()
		# d_mask = np.zeros(shape = (episode.decision_masks.shape[0], episode.decision_masks.shape[1]))
		# batch_index = np.arange(episode.decision_masks.shape[1]).tolist()
		# # print("d_mask shape = ", d_mask.shape)
		# d_mask[decision_index, batch_index] = 1
		# d_mask = tf.convert_to_tensor(d_mask)
		# entropy = tf.reduce_sum(tf.reduce_sum(attribute * tf.math.log(attribute+1e-32), axis=2)*tf.cast(d_mask, tf.float32), axis=0)

		# ####### Type 3  -all entropy terms in evidence output till the point the decision is made
		# choice_distribution = episode.decision_probabilities
		# print("choice_distribution - ", choice_distribution.shape)
		# entropy = tf.reduce_sum(tf.reduce_sum(choice_distribution * tf.math.log(choice_distribution+1e-32), axis=2)*tf.squeeze(1.0 - tf.cast(episode.decision_masks, tf.float32)), axis=0)

		# Add the loss to the episode, if an attribute name has been provided
		if attribute_name is not None and attribute_name != "":
			episode.add_attribute(attribute_name, entropy)

		return entropy
