# -*- coding: utf-8 -*-
"""
.. module:: reinforce_loss.py
   :platform: Linux, Windows, OSX
   :synopsis: Reward function for selecting the correct category

.. moduleauthor:: AARTLab <danahugh@andrew.cmu.edu>

An implementation of the loss function used by REINFORCE. 
"""

import tensorflow as tf

from ..datatypes import LossBatch

class REINFORCE_Loss:
	"""
	The REINFORCE_Loss class describes a function used by the REINFORCE
	algorithm.  In essence, the loss function is the negative log likelihood
	of the choices times the return of the choice.
	"""

	def __init__(self, **kwargs):
		"""
		Keyword Arguments
		-----------------
		reward_name : Optional[str], default=None
			Name of the subreward to calculate the return from.  If not provided,
			it will use the rewards attribute in the episode
		discount_factor : float, default=1.0
			Discount factor used to calculate the return
		"""

		# The loss can be based on either 1) a named return attribute in the
		# episode, or 2) a named reward attribute and discount factor in the
		# episode.
		self._reward_name = kwargs.get("reward_name", None)
		self._discount_factor = tf.constant(kwargs.get("discount_factor", 1.0), 
			                                 dtype=tf.float32)


	def __repr__(self):
		"""
		Produce a string representation of the reward, including the class name
		and the important attributes.
		"""

		return f'{self.__class__.__name__}'


	def __default_return(self, episode):
		"""
		Private function that simply returns the default return of the episode.

		Arguments
		---------
		episode : EpisodeBatch
		"""

		return episode.returns


	def __named_return(self, episode):
		"""
		Private function that returns a named return attribute of the episode

		Arguments
		---------
		episode : EpisodeBatch
		"""

		return getattr(episode, self._return_name)


	@tf.function
	def _calculate_return(self, rewards):
		"""
		Private function that calculates a return based on a named reward and
		local discount factor

		Arguments
		---------
		episode : EpisodeBatch
		"""

		# Create a TensorArray to store the returns in while calculating, with an
		# initial cummulative reward of 0 at the final timestep
		returns = tf.TensorArray(tf.float32, size=rewards.shape[0] + 1, dynamic_size=False, clear_after_read=False)
		returns = returns.write(rewards.shape[0], tf.zeros_like(rewards[0], dtype=tf.float32))

		# Work backward from the final step, accumulating the reward
		for t in tf.range(rewards.shape[0]-1, -1, -1):
			returns = returns.write(t, rewards[t] + self._discount_factor*returns.read(t+1))

		# Convert the return to a Tensor and remove the "dummy" value at the end
		return returns.stack()[:-1]


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

		# Determine the number of timesteps and batch size from the episode data
		num_timesteps, batch_size = episode.decision_probabilities.shape[:2]


		# Get the returns, depending on the keyword arguments provided during
		# creation. --> Will need to test this now that @tf.function is used!
		if self._reward_name is None:
			rewards = episode.rewards.rewards
		else:
			rewards = episode.rewards.subrewards[self._reward_name].rewards


		returns = self._calculate_return(rewards)


		# Create an index tensor to determine the probability of the choice taken
		# at each timestep for each batch
		time_idx, batch_idx = self._create_indices(num_timesteps, batch_size)
		choice_idx = episode.decisions
		indices = tf.concat([time_idx, batch_idx, choice_idx], 2)

		# print("Indices shape = ", indices.shape, " episode.decision_probabilities shape = ", episode.decision_probabilities.shape)
		# Get the log probability of each choice taken
		choice_probabilities = tf.gather_nd(episode.decision_probabilities, indices)
		choice_probabilities = tf.expand_dims(choice_probabilities, -1)
		choice_loss = -tf.math.reduce_sum(tf.math.log(choice_probabilities) * returns, [0,2])

		return LossBatch(choice_loss)
