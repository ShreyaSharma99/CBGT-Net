# -*- coding: utf-8 -*-
"""
.. module:: simple_categorical_reward.py
   :platform: Linux, Windows, OSX
   :synopsis: Reward function for selecting the correct category

.. moduleauthor:: AARTLab <danahugh@andrew.cmu.edu>

A simple reward function for selecting the correct category.
"""


import tensorflow as tf

from ..datatypes import RewardBatch


class SimpleCategoricalReward:
	"""
	Simple reward class used to calculate reward associated with guessing 
	the correct category.

	The reward function is parameterized by five variables, and is defined as

	r_t = R_0                 t <= T_max; no guess
	      R_1 - k * (t - 1)   t <= T_max; correct guess
	      R_2                 t <= T_max; incorrect guess
	      R_3                 t > T_max

	The variables are labeled as

	R_0 : no_guess_reward
	R_1 : correct_guess_reward
	R_2 : incorrect_guess_reward
	R_3 : timeout_reward
	k   : tardiness_rate
	"""

	def __init__(self, **kwargs):
		"""
		Keyword Arguments
		-----------------
		correct_guess_reward : float, default=0
		    Reward for making a correct guess (R_1)
		incorrect_guess_reward : float, default=0
		    Reward for making an incorrect guess (R_2)
		timeout_reward : float, default=0
		    Reward for reaching timeout / terminal state (R_3)
		no_guess_reward : float, default=0
		    Reward for not guessing (R_0)
		tardiness_rate : float, default=0
		    Reward for each time step taken to guess(k)
		"""

		self._correct_guess_reward = kwargs.get("correct_guess_reward", 0)
		self._incorrect_guess_reward = kwargs.get("incorrect_guess_reward", 0)
		self._timeout_reward = kwargs.get("timeout_reward", 0)
		self._no_guess_reward = kwargs.get("no_guess_reward", 0)
		self._tardiness_rate = kwargs.get("tardiness_rate", 0)


	def reset(self):
		"""
		Reset the reward
		"""

		pass


	def __repr__(self):
		"""
		Produce a string representation of the reward, including the class name
		and the important attributes.
		"""

		return f'{self.__class__.__name__}(R0={self._no_guess_reward}, R1={self._correct_guess_reward}, R2={self._incorrect_guess_reward}, R3={self._timeout_reward}, k={self._tardiness_rate})'


	@tf.function
	def __call__(self, episode, attribute_name="rewards"):
		"""
		Calculate the reward for a given step based on whether a choice was made,
		if the choice was correct, or if the state is terminal.

		Arguments
		---------
		episode : training.datatypes.EpisodeBatch
			Episode to compute the reward for and add
		attribute_name : string or None, default="reward"
			If provided, attempts to add an attribute with the provided name to
			the episode.  If None, then an attribute will not be added.

		Returns
		-------
		reward : training.datatypes.RewardBatch
			ExtendedType instance containing reward information 
		"""

		# How many timesteps and batches in the data?
		num_timesteps = episode.decisions.shape[0]
		batch_size = episode.decisions.shape[1]

		# Create some needed tensors from the episode, in right format (float32):
		# 1.  Calculate whether decisions were correct or not (0.0 or 1.0)
		# 2.  Create a tensor indicating that the last step is a timeout
		# 3.  Cast the did_decide tensor appropriately
		was_correct = tf.cast(episode.targets == episode.decisions, tf.float32)
		did_decide = tf.cast(episode.did_decide, tf.float32)
		is_timeout_start = tf.zeros((num_timesteps-1, batch_size, 1))
		is_timeout_end = tf.ones((1, batch_size, 1))
		is_timeout = tf.concat([is_timeout_start, is_timeout_end], 0)


		# Reward at timeout if a decision was not made at that timestep
		timeout_reward = is_timeout * (1.0-did_decide) * self._timeout_reward

		# Reward for not guessing at each timestep
		no_guess_reward = (1.0-is_timeout) * (1.0-did_decide) * self._no_guess_reward

		# Tardiness at each timestep, used for correct guess reward
		tardiness = tf.reshape(tf.range(0, num_timesteps), (-1,1,1))
		tardiness = tf.cast(tf.tile(tardiness, (1,batch_size,1)), tf.float32)

		# Correct and incorrect guess rewards
		correct_guess_reward = (1.0-is_timeout) * did_decide * was_correct * \
		                       (self._correct_guess_reward - self._tardiness_rate * tardiness)

		# Incorrect guess reward
		incorrect_guess_reward = (1.0-is_timeout) * did_decide * \
		                         (1.0-was_correct) * self._incorrect_guess_reward

		# Add up the components to get the final reward
		reward = timeout_reward + no_guess_reward + \
		         correct_guess_reward + incorrect_guess_reward

		# Rewards after the decision should be masked out by the decision mask
		reward = (1.0 - tf.cast(episode.decision_masks, tf.float32)) * reward

		return RewardBatch(reward)