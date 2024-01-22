# -*- coding: utf-8 -*-
"""
.. module:: reinforce_loss.py
   :platform: Linux, Windows, OSX
   :synopsis: Reward function for selecting the correct category

.. moduleauthor:: AARTLab <danahugh@andrew.cmu.edu>

An implementation of the loss function used by REINFORCE. 
"""

import tensorflow as tf

class REINFORCE_Loss_Common:
	"""
	The REINFORCE_Loss class describes a function used by the REINFORCE
	algorithm.  In essence, the loss function is the negative log likelihood
	of the choices times the return of the choice.
	"""

	def __init__(self, **kwargs):
		"""
		Keyword Arguments
		-----------------
		timesteps : int, default=None
			Number of timesteps per episode
		num_batches : int, default=None
			Number of batches per episode
		"""

		# Get the number of timesteps and batch size
		self._timesteps = kwargs.get("timesteps", None)
		self._num_batches = kwargs.get("num_batches", None)

		# Create the time and batch index tensors, if values were provided.
		# Otherwise, leave them as None
		self._time_idx = None
		self._batch_idx = None

		# If both of the keyword arguments were provided, create an index
		# tensor for use when called
		if self._timesteps is not None and self._num_batches is not None:
			self.__create_indices()


	def __repr__(self):
		"""
		Produce a string representation of the reward, including the class name
		and the important attributes.
		"""

		return f'{self.__class__.__name__}'


	def __create_indices(self):
		"""
		Helper to create index tensors along the timestep and batch axes.  This
		method assumes that the self._timeteps and self._num_batches values are
		valid.
		"""

		# Create a tensor updating the timestep along the first axis, and a
		# second tensor with the batch number in each row.  The final shape of
		# the tensors needs to be (num_timesteps, batch_size, 1)
		time_idx = tf.reshape(tf.range(0, self._timesteps), (-1,1,1))
		self._time_idx = tf.tile(time_idx, [1, self._num_batches, 1])
		batch_idx = tf.reshape(tf.range(0, self._num_batches), (1,-1,1))
		self._batch_idx = tf.tile(batch_idx, [self._timesteps, 1, 1])


	def __call__(self, episode, model=None, attribute_name="losses"):
		"""
		Create a tensor calculating the loss for the choices made in the 
		episode.  The episode is assumed to have an attribute named `returns`
		containing the return at each timestep for each batch. 

		Arguments
		---------
		episode : dict
			Dictionary of episode data
		"""

		# Get a tensor of the probabilities of the choices taken.  Stack the
		# time indices with the choice indices at each time to select the
		# probability of each choice
		choice_distribution = episode.decision_probabilities

		# Check to see if the time and batch indices were created, and
		# construct if not.  Also, check to see if the shape of the decision
		# probabilities match the number of timesteps and batch size, and 
		# recreate the indices if needed.
		if self._time_idx is None or self._batch_idx is None or \
		   self._timesteps != choice_distribution.shape[0] or \
		   self._num_batches != choice_distribution.shape[1]:
			# Get the dimensions needed from the choice distribution, and 
			# create the indices
			self._timesteps = choice_distribution.shape[0]
			self._num_batches = choice_distribution.shape[1]
			self.__create_indices()

		choice_idx = episode.decisions
		indices = tf.concat([self._time_idx, self._batch_idx, choice_idx], 2)

		# print("indices shape = ", indices.shape)
		# print("choice_distribution = ", choice_distribution.shape)
		# # Get the log probability of each choice taken
		choice_probabilities = tf.gather_nd(choice_distribution, indices)
		choice_probabilities = tf.reshape(choice_probabilities, choice_probabilities.shape + [1])
		choice_log_probabilities = tf.math.log(choice_probabilities+1e-32)

		# print("choice_probabilities = ", choice_probabilities.shape)
		# print("episode.returns = ", tf.math.reduce_sum(episode.returns[30:, ...]))
		choice_loss = -tf.math.reduce_sum(choice_log_probabilities * episode.returns, [0,2])

		# Add the reward to the episode, if an attribute name has been provided
		if attribute_name is not None and attribute_name != "":
			episode.add_attribute(attribute_name, choice_loss)

		return choice_loss
