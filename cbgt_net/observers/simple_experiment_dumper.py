# -*- coding: utf-8 -*-
"""
.. module:: simple_experiment_dumper.py
   :platform: Linux, Windows, OSX
   :synopsis: Simple observer to dump training results to a file

.. moduleauthor:: AARTLab <danahugh@andrew.cmu.edu>

A simple observer that writes experiment configuration and training results to
a JSON file.
"""

import json
import pathlib

import cbgt_net

import tensorflow as tf
import numpy as np


class NumpyListEncoder(json.JSONEncoder):
	"""
	A simple encoder that converts numpy arrays to lists.  This is needed to 
	encode numpy arrays as JSON when dumping experiment results to a file.
	"""

	def default(self, obj):
		"""
		If the input object is an ndarray, convert it to a list.  Otherwise,
		simply use the default encoding
		"""

		if isinstance(obj, np.ndarray):
			return obj.tolist()

		if isinstance(obj, tf.Tensor):
			return obj.numpy().tolist()

		super(NumpyListEncoder, self).default(obj)


class SimpleExperimentDumper:
	"""
	A SimpleExperimentDumper produces a JSON file containing the experiment
	configuration and the episode / evaluation results of teach training step.
	"""

	def __init__(self, experiment, **kwargs):
		"""
		Keyword Arguments
		-----------------
		output_path : string
		    Path to write experiment results to
		include_training_fields : list
		    List of fields in training episode data to include
		include_evaluation_fields : list
		    List of fields in evaluation results to include
		"""

		self._experiment = experiment

		# Where to write the resulting JSON file to
		self._output_path = kwargs.get("results_path", "./experiment_results.json")
		self._output_file = None

		# Store fields to ignore in the training and evaluation data
		self._include_training_fields = kwargs.get("include_training_fields", [])
		self._include_evaluation_fields = kwargs.get("include_evaluation_fields", [])

		# Set up a dictionary to dump data into
		self._data = { 'cbgt_net_version': cbgt_net.__version__,
		               'experiment_config': {},
		               'training': [],
		               'evaluation': []
		             }


	def on_experiment_start(self):
		"""
		Callback when the experiment is started
		"""

		# Store a copy of the experiment configuration.  Append the version of
		# CBGT_Net beting used
		self._data['experiment_config'] = self._experiment.config.copy()

		# Try to open the file, creating directories as needed.  If the file
		# cannot be opened, an exception should be thrown
		json_path = pathlib.Path(self._output_path)
		json_path.parent.mkdir(parents=True, exist_ok=True)

		# Check if the file exists
		if json_path.exists():
			# TODO:  How exactly do we want to handle this?
			pass

		# Try to open the file.  If an error occurs, it will be thrown here
		self._output_file = open(json_path, "w")


	def on_training_step(self, episode):
		"""
		Callback after a training step has been performed.

		Arguments
		---------
		episode : dict
		    Dictionary containing episode details
		"""

		# Create a dictionary containing only the fields to include in the
		# episode
		self._data['training'].append({field: episode[field] for field in self._include_training_fields})


	def on_evaluation(self, episode):
		"""
		Callback after a batch of evaluation episodes have been run.

		Arguments
		---------
		episode : EpisodeBatch
			Evaluation episode
		"""

		# Create a dictionary containing only the fields to include in the
		# episode
		self._data['evaluation'].append({field: episode[field] for field in self._include_training_fields})


	def on_experiment_complete(self, exception=None):
		"""
		Callback when the experiment is completed
		"""

		# Make sure the the output file exists, and is open
		if self._output_file is None:
			# TODO: Do something here...
			return

		if self._output_file.closed:
			# TODO: Do something here
			return

		# Dump the json data to the file, and close.  Use the NumpyListEncoder
		# to make sure that numpy arrays are converted to lists when
		# encountered
		try:
			json.dump(self._data, self._output_file, cls=NumpyListEncoder)
		finally:
			self._output_file.close()