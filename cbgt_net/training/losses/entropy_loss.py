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

class EntropyLoss:
	"""
	EntropyLoss generates a loss that aims to maximize the entropy of a
	categorical distribution.  This can be used to encourage entropy of the 
	output of the evidence module, or the final CBGT predictions.
	"""

	def __init__(self, **kwargs):
		"""
		Keyword Arguments
		-----------------
		use_evidence_output : boolean, default=False
			Used to indicate if the entropy loss be applied to the output of the
			evidence module (True), or the output of the CBGT network decision
			head (False)
		"""

		# The use_evidence_output flag is used to indicate if the entropy loss be
		# applied to the evidence or decision_probability streams.
		self._use_evidence_output = tf.constant(kwargs.get("use_evidence_output", True), dtype=tf.bool)



	def __repr__(self):
		"""
		Produce a string representation of the reward, including the class name
		and the important attributes.
		"""

		return f'{self.__class__.__name__}'


	@tf.function
	def __call__(self, episode, model=None):
		"""
		Create a tensor calculating the max entropy loss for the attribute 
		identified in the episode

		Arguments
		---------
		episode : dict
			Dictionary of episode data
		# """
		attribute = episode.evidence
		entropy =  tf.reduce_sum(tf.reduce_sum(attribute * tf.math.log(attribute+1e-32), axis=2)*tf.squeeze(1.0 - tf.cast(episode.decision_masks, tf.float32)), axis=0)

		# # If the entropy loss should be calculated for the evidence module, then
		# # Calculate the _mean_ of the entropy over all time steps
		# if self._use_evidence_output:
		# 	distribution = episode.evidence
		# 	entropy = tf.reduce_mean(tf.reduce_sum(distribution*tf.math.log(distribution+1e-32), axis=2), axis=0)
			
		# # Otherwise, need to calculate the entropy of the CBGT output when 
		# # decisions were made
		# else:
		# 	num_timesteps, batch_size = episode.decision_probabilities.shape[:2]

		# 	# Need to determine the time that decisions were made.  Generate a tensor
		# 	# of index values to use to gather distributions
		# 	decision_idx = num_timesteps - tf.reduce_sum(tf.cast(episode.decision_masks, tf.int32), axis=[0,2]) - 1
		# 	batch_idx = tf.range(batch_size, dtype=tf.int32)
		# 	_idx = tf.stack([decision_idx, batch_idx], axis=1)

		# 	distribution = tf.gather_nd(episode.decision_probabilities, _idx)
		# 	entropy = tf.reduce_sum(distribution * tf.math.log(distribution), axis=1)

		# # Return a LossBatch with the entropy as the loss term
		return LossBatch(entropy)