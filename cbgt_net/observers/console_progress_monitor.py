# -*- coding: utf-8 -*-
"""
.. module:: console_progress_monitor.py
   :platform: Linux, Windows, OSX
   :synopsis: REINFORCE training algorithm

.. moduleauthor:: AARTLab <danahugh@andrew.cmu.edu>

A simple observer to monitor progress of a trainer in a console.

Requires
--------
tqdm
"""

import cbgt_net

from tqdm import tqdm
import numpy as np
import tensorflow as tf
import datetime

class ConsoleProgressMonitor:
	"""
	A ConsoleProgressMonitor prints out training progress and evaluation
	statistics as a model is trained.
	"""

	def __init__(self, experiment, **kwargs):
		"""
		"""

		self._experiment = experiment
		self._progress_bar = None

		# Keep track of how many episodes have been trained upon
		self._num_episodes = 0

		# When to print out the training information
		self._training_progress_frequency = kwargs.get("training_progress_frequency", 250)
		self._next_training_evaluation = self._training_progress_frequency


	def on_experiment_start(self):
		"""
		Callback when the experiment is started
		"""

		# Print out relevant experiment information
		print("--===================== Experiment Properties =====================--")
		print("  CBGT-Net Version:  %s" % cbgt_net.__version__)
		print("  Model: %s" % self._experiment.model)
		print("    Evidence Module: %s" % self._experiment.model.evidence_module)
		print("    Accumulator Module: %s" % self._experiment.model.accumulator_module)
		print("    Threshold Module: %s" % self._experiment.model.threshold_module)
		print("  Environment: %s" % self._experiment.environment)
		print("  Trainer: %s" % self._experiment.trainer)
		print("  Experiment ID: %s" % self._experiment.id)
		print("--=================================================================--")

		# Create a progress bar
		self._progress_bar = tqdm(total=self._experiment.config["num_episodes"])


	def __print_rewards(self, reward_batch, prefix="", reward_name="Reward"):
		"""
		Pretty-print rewards, recursing into subrewards
		"""

		if reward_batch is None:
			return

		mean_reward = np.mean(np.sum(reward_batch.rewards.numpy(), axis=0))

		print(f"{prefix}{reward_name} = {mean_reward:.3f}")

		# Pretty-print subrewards
		for name, batch in reward_batch.subrewards.items():
			self.__print_rewards(batch, prefix+"  ", name)


	def __print_losses(self, loss_batch, prefix="", loss_name="Loss"):
		"""
		Pretty-print rewards, recursing into subrewards
		"""

		if loss_batch is None:
			return()

		mean_loss = np.mean(loss_batch.losses.numpy())

		print(f"{prefix}{loss_name} = {mean_loss:.4f}")

		# Pretty-print subrewards
		for name, batch in loss_batch.sublosses.items():
			self.__print_losses(batch, prefix+"  ", name)


	def __print_stats(self, episode):
		"""
		Pretty-print stats of interest in the episode.
		"""

		self.__print_rewards(episode.rewards, prefix="    ", reward_name="Average Reward")
		self.__print_losses(episode.losses, prefix="    ", loss_name="Training Loss")

		accuracy = episode.attributes.get("accuracy", None)
		decision_times = episode.attributes.get("decision_times", None)
		evidence_encoder_acc = episode.attributes.get("evidence_encoder_acc", None)
		
		if accuracy is not None:
			print("    Accuracy = %0.3f%%" % (100*accuracy.numpy()))
		if decision_times is not None:
			print("    Average Decision Time %0.2f" % np.mean(decision_times.numpy()))
		print("    Decision Threshold: %0.3f" % self._experiment.model.threshold_module.decision_threshold.numpy())
		# print("    Encoder Accuracy: %0.3f" % np.mean(evidence_encoder_acc.numpy()))
		print("    Encoder Accuracy: %0.3f" % (evidence_encoder_acc.numpy()))


	def on_training_step(self, episode):
		"""
		Callback after a training step has been performed.

		Arguments
		---------
		episode : dict
		    Dictionary containing episode details
		"""

		# Update the number of episodes and progress bar
		if self._progress_bar is not None:
			self._progress_bar.update(episode.batch_size)

		self._num_episodes += episode.batch_size

		# Check to see if we should print out training info
		if self._num_episodes >= self._next_training_evaluation:

			# Store when to print the next training results
			self._next_training_evaluation += self._training_progress_frequency

			print()
			print("Training Episode %d:" % self._num_episodes)
			self.__print_stats(episode)


	def on_evaluation(self, episode):
		"""
		Callback after a batch of evaluation episodes have been run.

		Arguments
		---------
		episode : EpisodeBatch
			Evaluation episode
		"""

		# Clear out the progress bar
		if self._progress_bar is not None:
			self._progress_bar.clear()

		# Print out progress
		print()
		print("Evaluation @ Episode %d:" % self._num_episodes)
		self.__print_stats(episode)


	def on_experiment_complete(self, exception=None):
		"""
		Callback when the experiment is completed
		"""

		# Close the progress bar
		if self._progress_bar is not None:
			self._progress_bar.clear()
			self._progress_bar.close()

		if exception is None:
			print("--======================= End of Experiment =======================--")
		else:
			print("--==================== Experiment Interrupted =====================--")

