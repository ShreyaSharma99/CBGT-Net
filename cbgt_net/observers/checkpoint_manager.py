# -*- coding: utf-8 -*-
"""
.. module:: checkpoint_manager.py
   :platform: Linux, Windows, OSX
   :synopsis: Observer responsible for managing experiment checkpoints

.. moduleauthor:: AARTLab <danahugh@andrew.cmu.edu>

A simple observer to manage checkpoints during experiment execution.

Requires
--------
"""

import os
import pathlib

import tensorflow as tf

class CheckpointManager:
	"""
	The CheckpointManager class is an observer used to manage checkpoints during
	experiment execution.  
	"""

	def __init__(self, experiment, **kwargs):
		"""
		Keyword Arguments
		-----------------
		checkpoint_path : string, default="checkpoints"
		    Path to write checkpoints to.  Expects to map to a directory
		file_prefix : string, default="ckpt"
		    String template for each checkpoint file.
		frequency : int, default=1000
		    Frequency (in number of episodes) to produce checkpoints.
		max_to_keep : int, default=10
		    Number of checkpoints to keep
		"""

		self._experiment = experiment

		# Where to write checkpoints to, and the file format.
		self._checkpoint_path = kwargs.get("checkpoint_path", "./checkpoint_folder_new")
		self._file_prefix = kwargs.get("file_prefix", "ckpt")
		self._checkpoint_prefix = os.path.join(self._checkpoint_path,
		                                       self._file_prefix)
		self._max_to_keep = kwargs.get("max_to_keep", 10)
		self._frequency = kwargs.get("frequency", 1000)
		self._next_checkpoint_step = 0
		self._num_episodes = 0

		

	def on_experiment_start(self):
		"""
		Callback when the experiment is started
		"""

		# Create the checkpoint directory, if it doesn't already exist
		pathlib.Path(self._checkpoint_path).mkdir(parents=True, exist_ok=True)


		# Create a Checkpoint and CheckpointManager instance.  The Checkpoint
		# instance will store the state of the model and optimizer
		self._checkpoint = tf.train.Checkpoint(model=self._experiment.model,
			                                    optimizer=self._experiment.trainer.optimizer,
			                                    step=tf.Variable(1))
		self._manager = tf.train.CheckpointManager(self._checkpoint,
			                                        self._checkpoint_path,
			                                        max_to_keep=self._max_to_keep)

		# When should the next checkpoint be saved
		self._next_checkpoint_step = self._frequency


	def on_training_step(self, episode):
		"""
		Callback after a training step has been performed.

		Arguments
		---------
		episode : dict
		    Dictionary containing episode details
		"""

		# return

		self._num_episodes += episode.batch_size

		self._checkpoint.step.assign(self._num_episodes)


		# Check to see if we should log the training info
		if self._num_episodes >= self._next_checkpoint_step:
		# Get the training step from the episode, and determine if a checkpoint
		# needs to be saved
		# if episode.attributes["episode_number"] >= self._next_checkpoint_step:
			save_path = self._manager.save()
			print("Saved Checkpoint for episode number {}: {}".format(self._num_episodes, save_path))
			self._next_checkpoint_step += self._frequency


	def on_evaluation(self, episode):
		"""
		Callback after an evaluation has been performed.

		Arguments
		---------
		episode : EpisodeBatch
			Batch of evaluation episodes
		"""

		pass


	def on_experiment_complete(self, exception=None):
		"""
		Callback when the experiment is completed
		"""

		pass