# -*- coding: utf-8 -*-
"""
.. module:: threshold_loss.py
   :platform: Linux, Windows, OSX
   :synopsis: Loss function 

.. moduleauthor:: AARTLab <danahugh@andrew.cmu.edu>

"""

import tensorflow as tf

from ..datatypes import LossBatch

class ThresholdLossDana:
	"""
	The REINFORCE_Loss class describes a function used by the REINFORCE
	algorithm.  In essence, the loss function is the negative log likelihood
	of the choices times the return of the choice.
	"""

	def __init__(self, **kwargs):
		"""
		Keyword Arguments
		-----------------
		correct_rate : float, default = -1
			Rate at which to reduce threshold when correct
		incorrect_rate : float, default = 1
			Rate at which to increase threshold when incorrect
		no_decision_rate : float, default = -1
			Rate at which to decreate the threshold when no decision is made
		"""

		self._correct_rate = tf.constant(kwargs.get('correct_rate', -1), dtype=tf.float32)
		self._incorrect_rate = tf.constant(kwargs.get('incorrect_rate', 1), dtype=tf.float32)
		self._no_decision_rate = tf.constant(kwargs.get('no_decision_rate', -1), dtype=tf.float32)



	def __repr__(self):
		"""
		Produce a string representation of the reward, including the class name
		and the important attributes.
		"""

		return f'{self.__class__.__name__}'


	@tf.function
	def _create_indices(self, num_timesteps, batch_size):
		"""
		Helper to create index tensors along the timestep and batch axes.  This
		method assumes that the self._timeteps and self._num_batches values are
		valid.
		"""

		# Create a tensor updating the timestep along the first axis, and a
		# second tensor with the batch number in each row.  The final shape of
		# the tensors needs to be (num_timesteps, batch_size, 1)
		time_idx = tf.reshape(tf.range(0, num_timesteps), (-1,1,1))
		time_idx = tf.tile(time_idx, [1, batch_size, 1])
		batch_idx = tf.reshape(tf.range(0, batch_size), (1,-1,1))
		batch_idx = tf.tile(batch_idx, [num_timesteps, 1, 1])

		return time_idx, batch_idx


	@tf.function
	def __call__(self, episode, model=None):
		"""
		Create a tensor calculating the loss for the choices made in the 
		episode.  The episode is assumed to have an attribute named `returns`
		containing the return at each timestep for each batch. 

		Arguments
		---------
		episode : dict
			Dictionary of episode data
		"""

		if model is None:
			return

		# Extract the decision threshold variable from the model
		threshold_variable = model.threshold_module.decision_threshold

		# Determine the number of timesteps and batch size from the episode data,
		# and create indices to access the time that a decision was made
		num_timesteps, batch_size = episode.decision_probabilities.shape[:2]

		decision_idx = num_timesteps - tf.reduce_sum(tf.cast(episode.decision_masks, tf.int32), axis=[0,2]) - 1
		batch_idx = tf.range(batch_size, dtype=tf.int32)
		_idx = tf.stack([decision_idx, batch_idx], axis=1)

		# Get the decisions, targets, and accumulators
		decisions = tf.gather_nd(episode.decisions, _idx)
		targets = tf.gather_nd(episode.targets, _idx)
		accumulators = tf.reduce_max(tf.gather_nd(episode.accumulators, _idx), axis=1)

		# Determine when the decisions were correct
		correct = tf.cast(decisions == targets, tf.float32)
		no_decision = tf.reduce_sum(tf.cast(episode.did_decide, tf.float32), axis=0)
		no_decision = 1.0 - tf.math.minimum(no_decision, 1.0)

		# Calculate the loss based on whether the decision was correct, and how
		# far from the max accumulator the threshold is
		correct_loss = (1.0 - no_decision) * correct * (accumulators - threshold_variable) * self._correct_rate
		incorrect_loss = (1.0 - no_decision) * (1-correct) * (accumulators - threshold_variable) * self._incorrect_rate
		no_decision_loss = no_decision * self._no_decision_rate * threshold_variable

		return LossBatch(correct_loss + incorrect_loss + no_decision_loss)

