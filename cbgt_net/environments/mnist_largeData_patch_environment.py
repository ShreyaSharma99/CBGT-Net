# -*- coding: utf-8 -*-
"""
.. module:: simple_categorical
   :platform: Linux, Windows, OSX
   :synopsis: Simple environment that produces a category as observations

.. moduleauthor:: AARTLab <danahugh@andrew.cmu.edu>

The simple categorical environment produces one of a pre-defined set of
categories when queried for an observation.  Instances of the environment have
a _noise_ attribute:  when an observation is requested, the environment will
produce a random, non-target category with probability _noise_, and the target
category otherwise.
"""

import numpy as np
import tensorflow as tf
import pickle
import matplotlib.pyplot as plt

from .environment import Environment


class MNIST_LargeData_Patch_Environment(Environment):
	"""
	Simple environment that produces evidence for a target value, corrupted by 
	some noise.  

	Attributes
	----------
	num_categories : int
		Number of categories that can be observed
	num_observations : int
		Number of observations that were made
	targets : int
		Value of the current target category
	noise : float
		Current noise level
	"""


	@classmethod
	def _build(cls, config):
		"""
		Builder method for creating a SimpleCategoricalEnvironment instance
		from a configuration dictionary.  The provided config file should have

		the following keys:

		* categories : int, or list
		    A list of category values, or an integer.  If an integer is given,
		    then the categories will be a list of integer values from 0 to
		    one less than the provided integer.
		* target : int or category, default=None
		    The initial target value for the environment.  If `None` is given,
		    then a random target is selected upon creation.
		* noise : float, default=0.1
		    The noise level of the environment
		* observation_mode : string
		    The observation mode to use, which must be one of "category",
		    "index", or "one_hot".  Observation mode is case insensitive.

		Arguments
		---------
		config : dict
		    Dictionary representation of the environment.
		"""

		# The `categories` key is the only requrement in the config file.
		# Make sure that it exists, and raise an error if it doesn't
		if not "categories" in config:
			raise AttributeError("Missing 'categories' in config dictionary.")
		# Extract the observation mode, which should be a string, and convert
		# to the enumerated value.

		env = cls(config["categories"], 
			       target=config.get("target", None),
			       noise=config.get("noise", 0.1),
				   batch_size=config.get("batch_size", 32),
				   patch_size=config.get("patch_size", [8,8,3]),
				   max_steps_per_episode=config.get("max_steps_per_episode", None),
				   images_per_class_train=config.get("images_per_class_train", None),
				   images_per_class_test=config.get("images_per_class_test", None),
				   image_shape=config.get("image_shape", None)
		)

		env = cls.OneHotWrapper(env)
		return env



	class OneHotWrapper(Environment):
		"""
		A wrapper for SimpleCategoricalEnvironment that returns observations as
		one-hot representations.
		"""

		def __init__(self, base_env):
			"""
			Arguments
			---------
			base_env : SimpleCategoricalEnvironment
			   Base environment to wrap
			"""

			self._base_env = base_env

			Environment.__init__(self, batch_size=base_env.batch_size)

			# Create a one-hot representation of the categories, and set the
			# target value according to the base environment

			self._categories = tf.Variable(tf.eye(self._base_env._num_categories, dtype=np.float32), trainable=False)
			self._target = tf.Variable(tf.gather(self._categories, self._base_env._target), trainable=False)
			self._target_index = self._base_env._target


		def __str__(self):
			"""
			String representation of the environment
			"""

			return "%s[%s]" % (self.__class__.__name__, str(self._base_env))


		# @property
		# def observation_shape(self):
		# 	return (self.num_categories.numpy(),)

		# @property
		# def target_shape(self):
		# 	return (1,)


		@property
		def categories(self):
			return self._categories


		@property
		def num_categories(self):
			return self._base_env._num_categories


		@property
		def num_observations(self):
			return self._base_env._num_observations

		
		@property
		def target(self):
			return self._target

	
		@property
		def target_value(self):
			return self._target

	
		@property
		def target_index(self):
			return tf.expand_dims(self._target_index, -1)

			
		@property
		def noise(self):
			return self._noise


		@tf.function
		def reset(self, target=None, training=True):
			"""
			Reset the environment.  Uses the target if it is given, otherwise, picks
			a target at random.
			"""

			self._base_env.reset(target, training=training)
			self._target.assign(tf.gather(self._categories, self._base_env.target))


		@tf.function	
		def observe(self, observation_mode=None, time_step=None):
			"""
			Generate an observation.
			"""

			observations = self._base_env.observe(observation_mode=observation_mode, time_step=time_step)
			# return tf.gather(self._categories, observations)	
			return observations


		def properties(self):
			"""			Returns a dictionary representation of an instance of the environment,
			containing the following fields:
	
			* name : the name of the environment
			* class : a fully-qualified class name of the environment
			* categories : a list of categorical values that can be observed
			* num_categories : the number of categories in the environment
			* observation_mode : the representation of observations
			* noise : the noise level of the environment
			* target_value : the current target's value
			* target_index : the current index of the target
	
			Returns
			-------
			dict
			    Dictionary containing the representation of the environment state.			
			"""

			# Required fields
			name = str(self)
			class_ = ".".join([self.__class__.__module__, self.__class__.__name__])
	
			# Construct the properties dictionary and return
			return { "name": name,
			         "class": class_,
			         "categories": self.categories,
			         "num_categories": self.num_categories,
			         "observation_mode": self.observation_mode.name,
			         "noise": self.noise,
			         "target_value": self.target_value,
			         "target_index": self.target_index
			       }



	def __init__(self, num_categories, training=True, **kwargs):
		"""
		Arguments
		---------
		num_categories : int
			Number of categories that an observation can take

		Keyword Arguments
		-----------------
		noise : float, default=0.2
			Noise level of the environment (change of producing incorrect 
			observation for target)
		target : int, default=None
		    Initial category index or value to use as the target value
		"""

		Environment.__init__(self, **kwargs)

		# Create the list of categories from the number of categories.  If the
		# number of categories provided is invalid, raise an exception
		if num_categories <= 0:
			raise TypeError("Number of categories must be strictly positive")

		noise = kwargs.get("noise", 0.2)
		if noise < 0.0 or noise > 1.0:
			self.logger.warning("%s:  Invalid noise level: %0.2f.  Noise must be in range [0,1].  Noise value is capped.", self, noise)
			noise = min(max(self._noise, 0.0), 1.0)


		# Store the keyword arguments as constants, and set up remaining variables
		self._categories = tf.constant(list(range(num_categories)), dtype=tf.int32)
		self._num_categories = tf.constant(len(self._categories), dtype=tf.int32)
		self._batch_size = tf.constant(self._batch_size, dtype=tf.int32)
		self._noise = tf.constant(noise, dtype=tf.float32)
		
		self._num_observations = tf.Variable(0, trainable=False)
		self._target = tf.Variable(tf.zeros((self._batch_size,), dtype=tf.int32), trainable=False)
		self._non_target_categories = tf.Variable(tf.zeros((self._batch_size, self._num_categories-1), dtype=tf.int32), trainable=False)

		self._image_shape = kwargs.get("image_shape", None)
		self._images_per_class_train = kwargs.get("images_per_class_train", None)
		self._images_per_class_test = kwargs.get("images_per_class_test", None)
		self._max_steps_per_episode = kwargs.get("max_steps_per_episode", None)	
		self._patch_size = kwargs.get("patch_size", [8,8,3])
		
		self._observed_patches = tf.Variable(tf.zeros((self._batch_size,self._max_steps_per_episode,self._patch_size[0], self._patch_size[1], self._patch_size[2]), dtype=tf.float32), trainable=False)


		with open('cbgt_net/mnist_train50.pkl', 'rb') as fileObj:
			self.image_data_train = pickle.load(fileObj)

		with open('cbgt_net/mnist_test10.pkl', 'rb') as fileObj:
			self.image_data_test = pickle.load(fileObj)

		# CHANGE : Save pkl data as tensor
		self.image_data_train = tf.convert_to_tensor(self.image_data_train, dtype=tf.float32)
		self.image_data_test = tf.convert_to_tensor(self.image_data_test, tf.float32)

		self.reset(kwargs.get("target",None), training=training)


	@property
	def categories(self):
		return self._categories

	@property
	def num_categories(self):
		return self._num_categories

	@property
	def num_observations(self):
		return self._num_observations

	@property
	def observation_shape(self):
		return (self.batch_size, self._patch_size[0], self._patch_size[1], self._patch_size[2])

	@property
	def target_shape(self):
		return (1,)
	
	@property
	def target(self):
		return self._target

	@property
	def target_value(self):
		return self._target

	@property
	def target_index(self):
		return self._target
		
	@property
	def noise(self):
		return self._noise
	
	def __str__(self):
		"""
		String representation of the environment
		"""

		return self.__class__.__name__


	@tf.function
	def _get_random_indices(self, max_idx):
		"""
		Helper function to get a tensor of random indices.  Shape of the returned
		tensor is (1,batch_size), and value ranges are [0, max_idx)
		"""

		# Create the logits --- need to stack values of an equidistribution
		logit_arg = tf.constant(1.0, dtype=tf.float32)/tf.cast(max_idx, tf.float32)
		logit_arg = tf.reshape(logit_arg, (1,1))
		logit_arg = tf.repeat(logit_arg, max_idx, axis=1)
		logits = tf.math.log(logit_arg)

		return tf.random.categorical(logits, self._batch_size, dtype=tf.int32)


	@tf.function
	def reset(self, target=None, training=True):
		"""
		Reset the environment.  Uses the target if it is given, otherwise, picks
		a target at random.

		If the target provided is a member of the categories, sets the target to
		that.  Otherwise, if it's an integer, uses that as the index to the
		category.

		Arguments
		---------
		target : category or int, default=None
			Target value or index to reset to.  If not provided, then picks a 
			random target value.
		"""

#		self._target = np.random.randint(0, self.num_categories, (self._batch_size,))
		target = tf.transpose(self._get_random_indices(self._num_categories))

		# Set up the non-target categories
#		self._non_target_categories = np.stack([np.delete(np.arange(self.num_categories),t) for t in self._target])
		stacked_categories = tf.repeat(tf.expand_dims(tf.range(self._num_categories), 0), self._batch_size, axis=0)
		stacked_target = tf.repeat(target, self._num_categories, axis=1)

		non_target_categories = tf.reshape(tf.gather_nd(stacked_categories, tf.where(stacked_categories!=stacked_target)), (self._batch_size,-1))

		self._target = self._target.assign(tf.squeeze(target))
		self._non_target_categories.assign(non_target_categories)

		# Set the number of observations to zero
		self._num_observations.assign(0)

		# self.target_value
		images_per_class = self._images_per_class_train if training else self._images_per_class_test
		target_val = tf.reshape(tf.convert_to_tensor(target, dtype=tf.dtypes.int32), (-1,1,1,1))
		target_val = tf.tile(target_val, [1, self._max_steps_per_episode, self._patch_size[0], self._patch_size[1]])

		img_index = tf.reshape(tf.random.uniform([self._batch_size], minval=0, maxval=images_per_class, dtype=tf.dtypes.int32), (-1,1,1,1))
		img_index = tf.tile(img_index, [1, self._max_steps_per_episode, self._patch_size[0], self._patch_size[1]])

		x_patch = tf.repeat(tf.random.uniform([self._batch_size, self._max_steps_per_episode, 1], minval=0, maxval=self._image_shape[0]-self._patch_size[0]+1, dtype=tf.dtypes.int32), repeats=self._patch_size[0], axis=-1)
		y_patch = tf.repeat(tf.random.uniform([self._batch_size, self._max_steps_per_episode, 1], minval=0, maxval=self._image_shape[1]-self._patch_size[1]+1, dtype=tf.dtypes.int32), repeats=self._patch_size[1], axis=-1)

		offset_x = tf.range(self._patch_size[0], delta=1, dtype=tf.dtypes.int32, name='range')
		offset_x_repeat = tf.repeat(tf.expand_dims(tf.expand_dims(offset_x, axis=0), axis=0), repeats=self._max_steps_per_episode, axis=1)
		offset_x_repeat = tf.repeat(offset_x_repeat, repeats=self._batch_size, axis=0)
		x_patch_indices = tf.repeat(tf.expand_dims(x_patch + offset_x_repeat, axis=3), repeats=self._patch_size[1], axis=3)

		offset_y = tf.range(self._patch_size[1], delta=1, dtype=tf.dtypes.int32, name='range')
		offset_y_repeat = tf.repeat(tf.expand_dims(tf.expand_dims(offset_y, axis=0), axis=0), repeats=self._max_steps_per_episode, axis=1)
		offset_y_repeat = tf.repeat(offset_y_repeat, repeats=self._batch_size, axis=0)
		y_patch_indices = tf.repeat(tf.expand_dims(y_patch + offset_y_repeat, axis=2), repeats=self._patch_size[0], axis=2)

		indices = tf.stack([target_val, img_index, x_patch_indices, y_patch_indices], axis=-1)
		# print("indices - ", indices.shape)
		# tf.print(indices[:, :,0])
		# tf.print(target)

		# self._observed_patches = self._observed_patches.assign(tf.gather_nd(self.image_data_train, indices))
		if training:
			# self._observation = tf.gather_nd(self.image_data_train, indices)
			self._observed_patches.assign(tf.gather_nd(self.image_data_train, indices))
			# tf.gather_nd(self.image_data_train, indices)
		# 	# [:, :, :, :, 0]
		else:
			# self._observation = tf.gather_nd(self.image_data_test, indices)
			self._observed_patches.assign(tf.gather_nd(self.image_data_test, indices))

		# for i in range(self.batch_size):
		



	@tf.function
	def observe(self, training=True, observation_mode=None, time_step=None):
		"""
		Generate an observation.

		Returns
		-------
		Batch of target values with probability (1-noise), or an incorrect target
		value otherwise.
		"""

		# Increment the observation count
		self._num_observations.assign(self._num_observations + 1)
		# tf.print("Time step =", time_step)
		# plt.imshow(self._observation[0, time_step, :, :, :], cmap='gray')
		# plt.savefig('.rough_figs/' + str(self.target[0])+ '.png')
		# tf.print(self.target[0])
		# tf.print(self._observed_patches[0, time_step, :, :, :])
		return self._observed_patches[:, time_step, :, :, :]



	def properties(self):
		"""
		Returns a dictionary representation of an instance of the environment,
		containing the following fields:

		* name : the name of the environment
		* class : a fully-qualified class name of the environment
		* categories : a list of categorical values that can be observed
		* num_categories : the number of categories in the environment
		* observation_mode : the representation of observations
		* noise : the noise level of the environment
		* target_value : the current target's value
		* target_index : the current index of the target

		Returns
		-------
		dict
		    Dictionary containing the representation of the environment state.
		"""

		# Required fields
		name = str(self)
		class_ = ".".join([self.__class__.__module__, self.__class__.__name__])

		# Construct the properties dictionary and return
		return { "name": name,
		         "class": class_,
		         "categories": self.categories,
		         "num_categories": self.num_categories,
		         "observation_mode": self.observation_mode.name,
		         "noise": self.noise,
		         "target_value": self.target_value,
		         "target_index": self.target_index
		       }
