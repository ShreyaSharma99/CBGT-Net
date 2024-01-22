# -*- coding: utf-8 -*-
"""
.. module:: tensorboard_monitor.py
   :platform: Linux, Windows, OSX
   :synopsis: Observer used to provide data to Tensorboard

.. moduleauthor:: AARTLab <danahugh@andrew.cmu.edu>

A simple observer to provide episode data to Tensorboard for logging
"""

import tensorflow as tf
import numpy as np
import datetime

class TensorboardMonitor:
	"""
	TensorboardMonitor is a simple observer used to log episode data to
	Tensorboard.
	"""

	def __init__(self, experiment, **kwargs):
		"""d
		Arguments
		---------
		experiment : Experiment
			Experiment being profiled
		"""

		self._experiment = experiment

		# Initialize the Tensorboard summary writter
		log_dir = kwargs.get("log_path", "logs")
		self.train_summary_writer = tf.summary.create_file_writer(log_dir)
		self.test_summary_writer = tf.summary.create_file_writer(log_dir + "_eval")


		# When to log information.  By default, this should be done after every
		# training step
		self._num_episodes = 0
		self._training_progress_frequency = kwargs.get("training_progress_frequency", 1)
		self._next_training_evaluation = self._training_progress_frequency

		self.dynamic_thresh_flag = kwargs.get("dynamic_thresh_flag", 0)
		self.is_tanh = kwargs.get("is_tanh", False)

		self.early_stop = False
		self.early_stop_flag = kwargs.get("early_stop_flag", False)
		self.loss_list_test = []
		self.last_k = 5000


	def on_experiment_start(self):
		"""
		Callback when the experiment is started
		"""

		pass


	def on_training_step(self, episode):
		"""
		Callback after a training step has been performed.

		Arguments
		---------
		episode : dict
		    Dictionary containing episode details
		"""

		self._num_episodes += episode.batch_size

		# Check to see if we should log the training info
		if self._num_episodes >= self._next_training_evaluation:

			# Store when to log the next training results
			self._next_training_evaluation += self._training_progress_frequency
			self._next_training_evaluation = max(self._next_training_evaluation, self._num_episodes)

			# Log the training results
			with self.train_summary_writer.as_default():
				tf.summary.scalar('Mean Loss', np.mean(episode.losses.losses.numpy()), step=self._num_episodes)
				tf.summary.scalar('Reinforce Loss', np.mean(episode.losses.sublosses['reinforce'].losses.numpy()), step=self._num_episodes)
				
				if self.is_tanh: 
					tf.summary.scalar('Tanh Reg Loss', np.mean(episode.losses.sublosses['tanh'].losses.numpy()), step=self._num_episodes)
				else: 
					tf.summary.scalar('Entropy Loss', np.mean(episode.losses.sublosses['entropy'].losses.numpy()), step=self._num_episodes)
				
				if self.dynamic_thresh_flag:
					tf.summary.scalar('Threshold Loss', np.mean(episode.losses.sublosses['threshold'].losses.numpy()), step=self._num_episodes)
					tf.summary.scalar('Dynamic Threshold', self._experiment.model.threshold_module.decision_threshold.numpy(), step=self._num_episodes)
				tf.summary.scalar('Mean Reward', np.mean(episode.rewards.rewards.numpy()), step=self._num_episodes)
				tf.summary.scalar('Accuracy', (100*episode.attributes["accuracy"].numpy()), step=self._num_episodes)
				tf.summary.scalar('Average Decision Time', np.mean(episode.attributes["decision_times"].numpy()), step=self._num_episodes)
				tf.summary.scalar('Encoder Accuracy', (100*episode.attributes["evidence_encoder_acc"].numpy()), step=self._num_episodes)
				


	def on_evaluation(self, episode):
		"""
		Callback after an evaluation step has been performed.  This method does
		nothing, and is simply for conforming to observer API.

		Arguments
		---------
		episode : dict
			Dictionary containing episode details
		"""

		# pass
		with self.test_summary_writer.as_default():
			self.loss_list_test.append(np.mean(episode.losses.losses.numpy()))
			if len(self.loss_list_test) > self.last_k:
				self.loss_list_test = self.loss_list_test[-self.last_k:]

			tf.summary.scalar('Eval Mean Loss', np.mean(episode.losses.losses.numpy()), step=self._num_episodes)
			if self.dynamic_thresh_flag:
				tf.summary.scalar('Eval Dynamic Threshold', self._experiment.model.threshold_module.decision_threshold.numpy(), step=self._num_episodes)

			tf.summary.scalar('Eval Cumulative Reward', np.mean(episode.rewards.rewards.numpy()), step=self._num_episodes)
			tf.summary.scalar('Eval Accuracy', (100*episode.attributes["accuracy"].numpy()), step=self._num_episodes)
			tf.summary.scalar('Eval Average Decision Time', np.mean(episode.attributes["decision_times"].numpy()), step=self._num_episodes)
			
		if self.early_stop_flag:
			if len(self.loss_list_test) == self.last_k and self.loss_list_test[-1] > np.mean(self.loss_list_test):
						print("EARLY STOPPIBG AT")
						self.early_stop = True

	def on_experiment_complete(self, exception=None):
		"""
		Callback when the experiment is completed.
		"""

		pass



