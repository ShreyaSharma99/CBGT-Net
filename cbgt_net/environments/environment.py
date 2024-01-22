# -*- coding: utf-8 -*-
"""
.. module:: environment.py
   :platform: Linux, Windows, OSX
   :synopsis: Definition of an abstract environment class

.. moduleauthor:: AARTLab <danahugh@andrew.cmu.edu>

This module defines an abstract environment class, defining core interface that
training algorithms and observers will use to interact with the environment.

Requirements
------------
Tensorflow 2.8

"""

import abc
import pathlib
import json

from ..utils import Loggable,Observable


class Environment(Loggable,Observable):
	"""
	The Enviornment class defines abstract methods common to all environments.
	These methods are used by training algorithms and observers to extract
	needed information from environments.

	Attributes
	----------
	name : string
	    Name of the environment
	"""

	def __init__(self, **kwargs):
		"""
		Keyword Arguments
		-----------------
		name : string
		    Name of the environment
		log_level : int, default=logging.WARNING
		    Log level that the logger should report
		"""

		Loggable.__init__(self, **kwargs)
		Observable.__init__(self, **kwargs)

		# What is the batch size of the environment?
		self._batch_size = kwargs.get("batch_size", 1)


	@classmethod
	@abc.abstractmethod
	def _build(cls, config):
		"""
		A private method used by concrete subclasses to create instances of the
		Environment.  This method is called by the public `build` method, once
		the contents of the json data is either passed through or loaded from
		a file.

		Arguments
		---------
		config : dict
			A dictionary containing parameters needed to construct the
			environment
		"""

		raise NotImplementedError


	@classmethod
	def build(cls, json_data):
		"""
		Construct an instance of the environment from json data.  The method 
		accepts either a dictionary as an argument (where the dictionary 
		represents the pre-loaded JSON data), or a string or pathlib.Path,
		representing the path to a JSON file. 

		Arguments
		---------
		json_data : dict, string, or pathlib.Path
			A dictionary containing loaded JSON data, or a path to a JSON file
		"""

		# If `json_data` is already in a dictionary format, then simply pass
		# to _build
		if isinstance(json_data, dict):
			return cls._build(json_data)

		# `json_data` was not a dictionary, so assume that the provided
		# argument is a path.  If it's a string, convert it to a pathlib.Path
		if type(json_data) is str:
			json_data = pathlib.Path(json_data)

		# Check to see if `json_data` refers to a path.  If not, raise an error
		if not isinstance(json_data, pathlib.Path):
			raise TypeError("Provided argument is not string, dictionary, or pathlib.Path: %s" % str(json_data))

		# Check to see if the provided path exists, and if it's a file
		if not json_data.exists():
			raise FileNotFoundError(str(json_data))
		if not json_data.is_file():
			raise IsADirectoryError("Provided path is not a file: %s" % str(json_data))

		# Try to load the contents of the file as a JSON string
		try:
			with open(json_data) as json_file:
				config = json.load(json_file)
			return cls._build(config)

		# The contents of the JSON file were not parsable for some reason
		except Exception as e:
			raise e


	@property
	def batch_size(self):
		return self._batch_size


	@property
	@abc.abstractmethod
	def observation_shape(self):
		"""
		Returns the shape of a single observation (i.e., not including batch size
		in the dimenstions).
		"""

		raise NotImplementedError


	@property
	@abc.abstractmethod
	def target_shape(self):
		"""
		Returns the shape of the target (not including batch size).
		"""

		raise NotImplementedError


	@abc.abstractmethod
	def properties(self):
		"""
		Returns a dictionary representation of an instance of the environment.
		The dictionary is expected to contain, at a minimum, the following
		fields:

		* name : the name of the environment
		* class : a fully-qualified class name of the environment

		The reported properties should contain sufficient information to 
		reconstruct the state of the environment.

		Returns
		-------
		dict
		    Dictionary containing the representation of the environment.
		"""

		raise NotImplementedError


	@abc.abstractmethod
	def reset(self):
		"""
		Resets the target to an initial state.  The `reset` method is expected 
		to be called prior to running a training or evaluation episode.
		"""

		raise NotImplementedError


	@abc.abstractmethod
	def observe(self):
		"""
		Generate an observation from the environment.  Subclasses should ensure
		that the `observe` method does not have any _required_ arguments.

		Returns
		-------
		tf.Tensor
			Tensor containin a batch of observations.  The first dimension of the
			tensor should be the batch size.
		"""

		raise NotImplementedError