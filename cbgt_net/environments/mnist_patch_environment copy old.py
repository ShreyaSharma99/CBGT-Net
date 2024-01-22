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

import enum

import numpy as np
import cv2
import random

import tensorflow as tf

import pickle
import matplotlib.pyplot as plt

from .environment import Environment


class MNIST_LargeData_CategoricalEnvironment_Vectorized_Patch_Old(Environment):
	"""
	MNIST_LargeData_CategoricalEnvironment_Vectorized_Patch environment that produces evidence for an mnist image as 
	small image patches

	Attributes
	----------
	categories : list
		List of unique categories that can be observed.  Category type is
		arbitrary, but entries are enforced to be complete.
	num_categories : int
		Number of categories that can be observed
	num_observations : int
		Number of observations that were made
	target_value : category
		Current target category.
	target_index : int
		Index of the current target category
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
		# observation_mode = config.get("observation_mode", "ONE_HOT").upper()
		# observation_mode = cls.ObservationMode[observation_mode]

		# Bulid the environment with the configuration
		return cls(config["categories"], 
			       target=config.get("target", None),
			       noise=config.get("noise", 0.1),
				   batch_size=config.get("batch_size", 32),
				   patch_size=config.get("patch_size", [8,8,3]),
				   max_steps_per_episode=config.get("max_steps_per_episode", None),
				   images_per_class_train=config.get("images_per_class_train", None),
				   images_per_class_test=config.get("images_per_class_test", None),
				   image_shape=config.get("image_shape", None),
		)
			    #    observation_mode=observation_mode)


	# class ObservationMode(enum.IntEnum):
	# 	"""
	# 	ObservationMode is used to indicate how observations should be encoded.
	# 	The available observation modes are defined as:

	# 	* CATEGORY: provides the category name as an observation
	# 	* INDEX: provides the index of the category as an observation
	# 	* ONE_HOT: provides a one-hot encoding of the category as an observation
	# 	"""

	# 	CATEGORY = enum.auto()
	# 	INDEX = enum.auto()
	# 	ONE_HOT = enum.auto()


	def __init__(self, categories, **kwargs):
		"""
		Arguments
		---------
		categories

		Keyword Arguments
		-----------------
		noise : float, default=0.2
			Noise level of the environment (change of producing incorrect 
			observation for target)
		target : int or category, default=None
		    Initial category index or value to use as the target value
		observation_mode : ObservationMode, default=ObservationMode.ONE_HOT
		    Initial observation mode that should be used for observations
		"""

		Environment.__init__(self, **kwargs)

		# Store or create the categories -- if an integer is passed, then 
		# interpret this as the number of categories desired.
		if type(categories) is int:
			self._categories = list(range(categories))
		else:
			# First try to cast the categories as a set and list, to ensure that 
			# each element is unique
			try:
				self._categories = list(set(categories))
			except Exception as e:
				self.logger.error("%s:  Unable to cast provided categories as a list: %s", self, categories)
				raise TypeError("Collection or integer expected")

		self._noise = kwargs.get("noise", 0.2)
		if self._noise < 0.0 or self._noise > 1.0:
			self.logger.warning("%s:  Invalid noise level: %0.2f.  Noise must be in range [0,1].  Noise value is capped.", self, self._noise)
			self._noise = min(max(self._noise, 0.0), 1.0)
		
		self._image_shape = kwargs.get("image_shape", None)
		self._images_per_class_train = kwargs.get("images_per_class_train", None)
		self._images_per_class_test = kwargs.get("images_per_class_test", None)
		self._max_steps_per_episode = kwargs.get("max_steps_per_episode", None)	

		# print("max_steps_per_episode", kwargs.get("max_steps_per_episode", None)	)

		# The environment will keep track of how many observations were made
		self._num_observations = 0
		# data = "/home/shreya/cbgt_net-develop/cbgt_net/shapes_data/shape_"
		with open('cbgt_net/mnist_train50.pkl', 'rb') as fileObj:
			self.image_data_train = pickle.load(fileObj)

		with open('cbgt_net/mnist_test10.pkl', 'rb') as fileObj:
			self.image_data_test = pickle.load(fileObj)

		self._patch_size = kwargs.get("patch_size", [8,8,3])

		with open('cbgt_net/mnist_train50_edge_patches_' + str(self._patch_size[0]) + '.pkl', 'rb') as fileObj:
			edge_data_train = pickle.load(fileObj)
		
		with open('cbgt_net/mnist_test10_edge_patches_' + str(self._patch_size[0]) + '.pkl', 'rb') as fileObj:
			edge_data_test = pickle.load(fileObj)

		self.edge_patches_train, self.list_len_train = edge_data_train["patch_edge"], edge_data_train["list_len"]
		self.edge_patches_test, self.list_len_test = edge_data_test["patch_edge"], edge_data_test["list_len"]
		
		# print("self.image_data_train - ", self.image_data_train.shape)

		# self.image_data_train = tf.convert_to_tensor(self.image_data_train)
		# self.image_data_test = tf.convert_to_tensor(self.image_data_test)
		# # print("Cat - ", self._categories)
		# for i in self._categories:
		# 	img_arr = cv2.imread(data + str(i) + ".png")
		# 	self.image_data.append((img_arr-1.)/254.)

		# self.image_data = np.asarray(self.image_data)	

		# Reset the environment to set up the target value
		self._batch_size = kwargs.get("batch_size", 1)
		self.reset(kwargs.get("target",None), self._batch_size, self._patch_size)


	@property
	def categories(self):
		return self._categories

	@property
	def num_categories(self):
		return len(self.categories)

	@property
	def num_observations(self):
		return self._num_observations
	
	@property
	def target_value(self):
		return self._target_value

	@property
	def target_index(self):
		return self._target_index

	@property
	def target(self):
		return tf.cast(tf.convert_to_tensor(self._target_index[:, None]),  tf.int32)
		
	@property
	def noise(self):
		return self._noise

	@property
	def patch_size(self):
		return self._patch_size

	# @property
	# def observation_mode(self):
	# 	return self._observation_mode

	# @observation_mode.setter
	# def observation_mode(self, mode):

	# 	# Check to see if the mode is valid
	# 	if not mode in self.ObservationMode:
	# 		self.logger.waring("%s:  Trying to set `observation_mode` to invalid value: %s", self, mode)
	# 		return

	# 	self.observation_mode = mode
	

	def __str__(self):
		"""
		String representation of the environment
		"""

		return self.__class__.__name__



	def reset(self, target=None, batch_size=None, patch_size=None, training=True):
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

		# No target provided -- select one at random
		if target is None:
			self._target_index = np.random.randint(0, len(self._categories), (self._batch_size,))
			# np.random.randint(0, self.num_categories)
			self._target_value = self._target_index

		# Integer provided that is not in the categories -- assume used as index
		elif type(target) is int and not target in self._categories:
			# Check to see if the given target can be used as an index
			try:
				self._target_index = target
				self._target_value = self.categories[self.target_index]
			except IndexError as e:
				self.logger.error("%s:  Target index assumed, out of bounds: %d", self, target)
				raise e 
		
		# Last Case -- assume the target provided is in the list of categories
		else:
			# Check to see if the target is in the categories
			if not target in self._categories:
				self.logger.error("%s:  Provided target is not in categories: %s", self, str(target))
				raise ValueError("Target not in categories: %s" % str(target))

			self._target_value = target
			self._target_index = self.categories.index(self.target_value)
		
		# print("self._target_index " ,self._target_index)
		# print("Batch size - ", self._batch_size)
		# Set up the non-target categories
		self._non_target_categories = [self.categories[:self._target_index[i]] + self.categories[self._target_index[i]+1:] for i in range(self._batch_size)]
		self._non_target_categories = np.array(self._non_target_categories)
		
		if batch_size is not None:
			self._batch_size = batch_size

		if patch_size is not None:
			self._patch_size = patch_size


		# Set the number of observations to zero
		self._num_observations = 0


	# def observe(self, training=True, observation_mode=None):
	# 	"""
	# 	Generate an observation.  The format of the returned observation will
	# 	be determined by the provided `observation_mode` argument, or using the
	# 	value of the `observation_mode` attribute if not provided.

	# 	Arguments
	# 	---------
	# 	observation_mode : ObservationMode, default=None
	# 	    The mode of the observation

	# 	Returns
	# 	-------
	# 	The target value with probability (1-noise), or an incorrect target 
	# 	otherwise.  The mode of observation is based on the observation_mode,
	# 	either passed as an argument, or the mode used by the instance.
	# 	"""	

	# 	patch_sz = self._patch_size
	# 	self._num_observations += 1

	# 	# if training:
	# 	if True:
	# 		num_data_per_class = 50
	# 		data_index = tf.random.uniform((self._batch_size,), minval=0, maxval=num_data_per_class, dtype=tf.int32)
	# 		print("data_index - ", data_index.shape)
	# 		observation = tf.identity(self.target_value)

	# 		# distractor_indices = tf.where(tf.random.uniform((self._batch_size,)) < self.noise)
	# 		# print("distractor_indices = ", distractor_indices.shape)
	# 		# noise_indx = tf.random.categorical(tf.math.log(tf.ones((self.num_categories-1,))), distractor_indices.shape[0])
	# 		# print("noise_indx = ", noise_indx.shape)
	# 		# noise_values = tf.gather(self._non_target_categories, noise_indx, axis=1)
	# 		# observation = tf.tensor_scatter_nd_update(observation, tf.expand_dims(distractor_indices, axis=1), noise_values)

	# 		# query_list = tf.ones((self._batch_size,), dtype=tf.int32)
	# 		# patch_idx = tf.random.uniform((self._batch_size,), minval=0, maxval=self.list_len_train.shape[1], dtype=tf.int32)

	# 		observe_img = tf.zeros([self._batch_size, patch_sz[0], patch_sz[1], self.image_data_train.shape[-1]])
	# 		for i in range(self._batch_size):
	# 			patch_idx = int(tf.random.uniform((1,), minval=0, maxval=self.list_len_train[observation[i], data_index[i]], dtype=tf.int32).numpy())
	# 			print("patch_idx", patch_idx)
	# 			x = self.edge_patches_train[observation[i]][data_index[i]][patch_idx, 0]
	# 			y = self.edge_patches_train[observation[i]][data_index[i]][patch_idx, 1]
	# 			print(int(observation[i].numpy()), int(data_index[i].numpy()), x , y )
	# 			observe_img[i] = self.image_data_train[int(observation[i].numpy()), int(data_index[i].numpy()), x : x+patch_sz[0], y : y+patch_sz[1], :]
	# 			# observe_img.append(tf.gather_nd(self.image_data_train, [[observation[i], data_index[i], x+j, y+k, c] for j in range(patch_sz[0]) for k in range(patch_sz[1]) for c in range(self.image_data_train.shape[-1])]))
				
	# 		# observe_img = tf.reshape(tf.stack(observe_img, axis=0), (self._batch_size, patch_sz[0], patch_sz[1], self.image_data_train.shape[-1]))
	# 		# observation[i], data_index[i], x : x+patch_sz[0], y : y+patch_sz[1], :
	# 		# 
	# 	print("observe_img.shape " , observe_img.shape)
	# 	return observe_img


	# def observe(self, training=True, observation_mode=None):
	# 	"""
	# 	Generate an observation.  The format of the returned observation will
	# 	be determined by the provided `observation_mode` argument, or using the
	# 	value of the `observation_mode` attribute if not provided.

	# 	Arguments
	# 	---------
	# 	observation_mode : ObservationMode, default=None
	# 	    The mode of the observation

	# 	Returns
	# 	-------
	# 	The target value with probability (1-noise), or an incorrect target 
	# 	otherwise.  The mode of observation is based on the observation_mode,
	# 	either passed as an argument, or the mode used by the instance.
	# 	"""	

	# 	patch_sz = self._patch_size
	# 	self._num_observations += 1

	# 	if training:
	# 	# if True:
	# 		num_data_per_class = 50
	# 		data_index = tf.random.uniform(shape=[self._batch_size], minval=0, maxval=num_data_per_class, dtype=tf.int32)
	# 		observation = self.target_value
			
	# 		if self.noise > 0:
	# 			raise Exception("Change the environment code to handle noise! Currently noise can only be 0.")

	# 		# # For adding noise!
	# 		# distractor_indices = np.where(np.random.random((self._batch_size,)) < self.noise)[0]
	# 		# noise_indx = random.choices(np.arange(self.num_categories-1), k=distractor_indices.shape[0])
	# 		# noise_values = self._non_target_categories[list(distractor_indices), noise_indx]
	# 		# observation[distractor_indices] = noise_values

	# 		# query_list = np.ones(self._batch_size, dtype=int)
	# 		# query_list[:] = self.list_len_train[observation, data_index]

	# 		indices = tf.stack([observation, data_index], axis=-1)
	# 		patch_idx_max = tf.gather_nd(self.list_len_train, indices)
	# 		patch_idx = tf.cast(tf.random.uniform(shape=[self._batch_size], minval=0, maxval=1) * patch_idx_max, dtype=tf.int32)

	# 		indices_obs = tf.stack([observation, data_index], axis=-1)

	# 		patch_idx = np.random.randint(self.list_len_train[observation, data_index])
			
	# 		observe_img = []
	# 		for i in range(self._batch_size):
	# 			x = self.edge_patches_train[observation[i]][data_index[i]][patch_idx[i], 0]
	# 			y = self.edge_patches_train[observation[i]][data_index[i]][patch_idx[i], 1]
	# 			observe_img.append(self.image_data_train[observation[i], data_index[i], x : x+patch_sz[0], y : y+patch_sz[1], :])
			
	# 		observe_img = np.array(observe_img)

	# 	else:  #evaluation
	# 		# print("In here for evaluation!!")
	# 		num_data_per_class = 10
	# 		data_index = np.random.randint(0, num_data_per_class, (self._batch_size,))
	# 		observation = self.target_value
	# 		# np.copy(self.target_value)
			
	# 		if self.noise > 0:
	# 			raise Exception("Change the environment code to handle noise! Currently noise can only be 0.")

	# 		# # For adding noise!
	# 		# distractor_indices = np.where(np.random.random((self._batch_size,)) < self.noise)[0]
	# 		# noise_indx = random.choices(np.arange(self.num_categories-1), k=distractor_indices.shape[0])
	# 		# noise_values = self._non_target_categories[list(distractor_indices), noise_indx]
	# 		# observation[distractor_indices] = noise_values

	# 		patch_idx = np.random.randint(self.list_len_test[observation,data_index])
	# 		observe_img = []
	# 		for i in range(self._batch_size):
	# 			x = self.edge_patches_test[observation[i]][data_index[i]][patch_idx[i], 0]
	# 			y = self.edge_patches_test[observation[i]][data_index[i]][patch_idx[i], 1]
	# 			observe_img.append(self.image_data_test[observation[i], data_index[i], x : x+patch_sz[0], y : y+patch_sz[1], :])
			
	# 		observe_img = np.array(observe_img)


		# # print(observe_img.shape)
		# # return np.array([observe_img])

		# # plt.imshow(observe_img[0, ...], cmap='gray')
		# # print("saving random img")
		# # plt.savefig('/home/shreya/cbgt_net-develop/cbgt_net/img_dump/random.png')
		# # plt.show()		

		# return tf.convert_to_tensor(observe_img)



	def observe(self, training=True, observation_mode=None, time_step=None):
		"""
		Generate an observation.  The format of the returned observation will
		be determined by the provided `observation_mode` argument, or using the
		value of the `observation_mode` attribute if not provided.

		Arguments
		---------
		observation_mode : ObservationMode, default=None
		    The mode of the observation

		Returns
		-------
		The target value with probability (1-noise), or an incorrect target 
		otherwise.  The mode of observation is based on the observation_mode,
		either passed as an argument, or the mode used by the instance.
		"""	

		patch_sz = self._patch_size
		self._num_observations += 1
		observation = self.target_value

		if training:
		# if True:
			# num_data_per_class = 50
			data_index = np.random.randint(0, self._images_per_class_train, (self._batch_size,))
			
			if self.noise > 0:
				raise Exception("Change the environment code to handle noise! Currently noise can only be 0.")

			# # For adding noise!
			# distractor_indices = np.where(np.random.random((self._batch_size,)) < self.noise)[0]
			# noise_indx = random.choices(np.arange(self.num_categories-1), k=distractor_indices.shape[0])
			# noise_values = self._non_target_categories[list(distractor_indices), noise_indx]
			# observation[distractor_indices] = noise_values

			# query_list = np.ones(self._batch_size, dtype=int)
			# query_list[:] = self.list_len_train[observation, data_index]

			patch_idx = np.random.randint(self.list_len_train[observation, data_index])
			observe_img = []
			for i in range(self._batch_size):
				x = self.edge_patches_train[observation[i]][data_index[i]][patch_idx[i], 0]
				y = self.edge_patches_train[observation[i]][data_index[i]][patch_idx[i], 1]
				observe_img.append(self.image_data_train[observation[i], data_index[i], x : x+patch_sz[0], y : y+patch_sz[1], :])
			
			observe_img = np.array(observe_img)

		else:  #evaluation
			# print("In here for evaluation!!")
			# num_data_per_class = 10
			data_index = np.random.randint(0, self._images_per_class_test, (self._batch_size,))
			
			if self.noise > 0:
				raise Exception("Change the environment code to handle noise! Currently noise can only be 0.")

			# # For adding noise!
			# distractor_indices = np.where(np.random.random((self._batch_size,)) < self.noise)[0]
			# noise_indx = random.choices(np.arange(self.num_categories-1), k=distractor_indices.shape[0])
			# noise_values = self._non_target_categories[list(distractor_indices), noise_indx]
			# observation[distractor_indices] = noise_values

			patch_idx = np.random.randint(self.list_len_test[observation,data_index])
			observe_img = []
			for i in range(self._batch_size):
				x = self.edge_patches_test[observation[i]][data_index[i]][patch_idx[i], 0]
				y = self.edge_patches_test[observation[i]][data_index[i]][patch_idx[i], 1]
				observe_img.append(self.image_data_test[observation[i], data_index[i], x : x+patch_sz[0], y : y+patch_sz[1], :])
			
			observe_img = np.array(observe_img)


		# print(observe_img.shape)
		# return np.array([observe_img])

		# plt.imshow(observe_img[0, ...], cmap='gray')
		# print("saving random img")
		# plt.savefig('/home/shreya/cbgt_net-develop/cbgt_net/img_dump/random.png')
		# plt.show()		

		return tf.convert_to_tensor(observe_img)

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
		        #  "observation_mode": self.observation_mode.name,
		         "noise": self.noise,
		         "target_value": self.target_value,
		         "target_index": self.target_index
		       }
