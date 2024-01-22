# -*- coding: utf-8 -*-
"""
.. module:: reinforce.py
   :platform: Linux, Windows, OSX
   :synopsis: REINFORCE training algorithm

.. moduleauthor:: AARTLab <danahugh@andrew.cmu.edu>

Implmentation of REINFORCE algorithm for training CBGT_Net instances
"""

import tensorflow as tf

from tensorflow.keras.optimizers.schedules import ExponentialDecay

from .datatypes import EpisodeBatch
from .losses import REINFORCE_Loss
from ..utils import Loggable,Observable
import matplotlib.pyplot as plt
import numpy as np


class REINFORCE_Trainer(Loggable,Observable):
	"""
	Class that trains a model for a given environment using the REINFORCE 
	algorithm.

	Arguments
	---------
	model : CBGT_Net
		Network model being trained
	environment : Environment
		Environment the model is trained on
	"""

	def __init__(self, model, environment, reward, loss, optimizer=None, **kwargs):
		"""
		Arguments
		---------
		model : CBGT_Net
			Network model to train
		environment : Environment
			Environment the model is trained on
		reward : Reward
			Reward function instance
		loss : Loss, default=REINFORCE_Loss
			Loss function instance

		Keyword Arguments
		-----------------
		discount_factor : float, default=0.9
			Amount to discount reward for each observation
		"""

		Loggable.__init__(self, **kwargs)
		Observable.__init__(self, **kwargs)


		self._model = model
		self._environment = environment
		self._reward = reward
		self._loss = loss


		# Create an ADAM optimizer if needed, otherwise, use the provided one
		if optimizer is None:
			initial_learning_rate = 0.001
			lr_schedule = ExponentialDecay(
				initial_learning_rate, decay_steps=100000, decay_rate=0.9, staircase=True
			)
			self._optimizer = tf.keras.optimizers.Adam(learning_rate=kwargs.get("learning_rate", 0.0005))  #latest = 0.001


		else:
			self._optimizer = optimizer

		self._discount_factor = tf.constant(kwargs.get("discount_factor", 0.95), dtype=tf.float32)
		self._max_steps_per_episode = tf.constant(kwargs.get("max_steps_per_episode", 30), dtype=tf.int32)
		self.dynamic_thresh_flag = kwargs.get("dynamic_thresh_flag", False)
		self._ce_loss = kwargs.get("ce_loss", False)

		# Measures to perform after training and evaluation
		self._post_training_measures = []
		self._post_evaluation_measures = []


	@property
	def model(self):
		return self._model


	@property
	def environment(self):
		return self._environment


	@property
	def optimizer(self):
		return self._optimizer
	

	@property
	def reward(self):
		return self._reward


	@property
	def loss(self):
		return self._loss


	def add_post_training_measure(self, measure):
		"""
		Add a measure to perform on episodes generated in each training step.

		Arguments
		---------
		measure : callable
			Measure to perform
		"""

		self._post_training_measures.append(measure)


	def add_post_evaluation_measure(self, measure):
		"""
		Add a measure to perform on episodes generated during evaluation.

		Arguments
		---------
		measure : callable
			Measure to perform
		"""

		self._post_evaluation_measures.append(measure)


	@tf.function
	def _run_episode(self, training=tf.constant(True)):
		"""
		Run a single episode to collect training data.

		Arguments
		---------
		training : tf.Tensor
			Boolean tensor indicating if the episode is training or evaluation

		Returns
		-------
		"""

		# print("ENTERED reinforce! ")
		

		# Reset the model and environment
		self.environment.reset(training=training)
		self.model.reset()

		# Setup tensor arrays to return once the function is complete
		observations = tf.TensorArray(tf.float32, size=self._max_steps_per_episode, dynamic_size=False, clear_after_read=False)
		targets = tf.TensorArray(tf.int32, size=self._max_steps_per_episode, dynamic_size=False, clear_after_read=False)
		evidence = tf.TensorArray(tf.float32, size=self._max_steps_per_episode, dynamic_size=False, clear_after_read=False)
		accumulators = tf.TensorArray(tf.float32, size=self._max_steps_per_episode, dynamic_size=False, clear_after_read=False)
		decision_distributions = tf.TensorArray(tf.float32, size=self._max_steps_per_episode, dynamic_size=False, clear_after_read=False)
		did_decide_probabilities = tf.TensorArray(tf.float32, size=self._max_steps_per_episode, dynamic_size=False, clear_after_read=False)
		did_decides = tf.TensorArray(tf.bool, size=self._max_steps_per_episode, dynamic_size=False, clear_after_read=False)
		decisions = tf.TensorArray(tf.int32, size=self._max_steps_per_episode, dynamic_size=False, clear_after_read=False)
		decision_masks = tf.TensorArray(tf.bool, size=self._max_steps_per_episode+1, dynamic_size=False, clear_after_read=False)

		threshold_vectors = tf.TensorArray(tf.float32, size=self._max_steps_per_episode, dynamic_size=False, clear_after_read=False)

		# Set up the first decision mask to False
		decision_masks = decision_masks.write(0, tf.cast(tf.zeros((self.environment.batch_size, 1)), tf.bool))

		# Run for the maximum number of steps
		for t in tf.range(self._max_steps_per_episode):

			# Get an observation from the environment, and add the observation and
			# target value to the trajectory.
			# observation = self.environment.observe(time_step=t)
			observation = self.environment.observe(training=training, time_step=t)
		
			# Store values of the environment at this time step, and the value of
			# the accumulator prior to execution
			observations = observations.write(t, observation)#.mark_used()
			targets = targets.write(t, self.environment.target_index)#.mark_used()
			accumulators = accumulators.write(t, self.model.accumulator)#.mark_used()

			# Do a forward pass on the model
			decision_distribution, threshold_output = self.model(observation)
			
			# print("self.dynamic_thresh_flag", self.dynamic_thresh_flag)
			if self.dynamic_thresh_flag:
				did_decide_probability, threshold_vector = threshold_output
				threshold_vectors  = threshold_vectors.write(t, threshold_vector)
			else: did_decide_probability = threshold_output


			# Store model parameters and outputs for this time step
			evidence = evidence.write(t, self.model.evidence_tensor)#.mark_used()
			decision_distributions = decision_distributions.write(t, decision_distribution)#.mark_used()
			did_decide_probabilities = did_decide_probabilities.write(t, did_decide_probability)#.mark_used()


			# NOTE:  It may be quicker to split this into two functions.  Will
			#        need to explore and benchmark
			if training and not self._ce_loss:
				decision = tf.random.categorical(tf.math.log(decision_distribution+1e-32), 1, dtype=tf.int32)
				did_decide = tf.math.less(tf.random.uniform(did_decide_probability.shape), did_decide_probability)
			else:
				decision = tf.reshape(tf.cast(tf.math.argmax(decision_distribution+1e-32, 1), tf.int32), (-1,1))
				did_decide = tf.math.greater(did_decide_probability, 0.5)

			if t == self._max_steps_per_episode-1:
				did_decide = tf.constant(True, dtype=None, shape=did_decide.shape)

			did_decides = did_decides.write(t, did_decide)#.mark_used()

			decisions = decisions.write(t, decision)#.mark_used()

			decision_masks = decision_masks.write(t+1, tf.math.logical_or(decision_masks.read(t), did_decide))

		# Collect the individual Tensors into an EpisodeBatch instance
		episode = EpisodeBatch(observations = observations.stack(),
			                    targets = targets.stack(),
			                    evidence = evidence.stack(),
			                    accumulators = accumulators.stack(),
			                    decision_probabilities = decision_distributions.stack(),
			                    did_decide_probabilities = did_decide_probabilities.stack(),
			                    did_decide = did_decides.stack(),
			                    decisions = decisions.stack(),
			                    decision_masks = decision_masks.stack()[:-1],
			                    )
		
		if self.dynamic_thresh_flag:
			episode = episode.add_attribute("threshold_vectors", threshold_vectors.stack())
		return episode


	@tf.function
	def train_step(self):
		"""
		Perforn the training step part of training, _not_ any post neasyres
		"""

		self.reward.reset()

		with tf.GradientTape() as tape:

			episode = self._run_episode()
			# episode = episode.add_attribute("epoch", epoch)

			rewards = self._reward(episode)
			episode = episode.set_rewards(rewards)
		
			losses = self._loss(episode, self._model)
			episode = episode.set_losses(losses)

			average_loss = tf.reduce_mean(tf.cast(episode.losses.losses, tf.float32))

		# print("Model summary - ", self.model.summary())
		# Run the optimization step
		gradients = tape.gradient(average_loss, self.model.trainable_variables)
		self._optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

		return episode


	def train(self):
		"""
		Train the model for the given number of episodes.
		"""

		episode = self.train_step()

		# Perform all post-training measures.  These will be done outside of 
		# tf.function, and are assumed to be reasonably quick operations
		for measure in self._post_training_measures:
			episode = measure(episode, self.model)

		return episode


	def evaluate(self, total_num_episodes):
		"""

		Calculate the average return of the model over the given number of
		episodes
		"""

		# Will need to accumulate episodes until the total number of episodes
		# is satisfied
		episode = self._run_episode(training=tf.constant(False))
		# TODO:  Implement the __add__ method in the new EpisodeBatch type

		# Add the rewards, returns, and losses into the episode
		self.reward.reset()

		rewards = self._reward(episode)
		episode = episode.set_rewards(rewards)

		losses = self._loss(episode, self._model)
		episode = episode.set_losses(losses)

		# Augment the evaluation episodes with accuracy measures
		for measure in self._post_evaluation_measures:
			episode = measure(episode, self.model)

		return episode
