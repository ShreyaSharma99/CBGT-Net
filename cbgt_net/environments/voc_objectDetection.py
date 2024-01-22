# -*- coding: utf-8 -*-
"""
.. module:: voc_objdetect
   :platform: Linux, Windows, OSX
   :synopsis: Environment that produces image patche as observations

.. moduleauthor:: AARTLab <danahugh@andrew.cmu.edu>

The simple categorical environment produces one of a pre-defined set of
categories when queried for an observation.  Instances of the environment have
a _noise_ attribute:  when an observation is requested, the environment will
produce a random, non-target category with probability _noise_, and the target
category otherwise.
"""

import enum

import numpy as np

from .environment import Environment

import tensorflow as tf

import tensorflow_datasets.public_api as tfds

import random

## import skimage

import copy 

## from skimage.transform import resize

## import matplotlib.pyplot as plt


class VOC_ObjDetect(Environment):
	"""
	Simple environment that produces evidence for a target value, corrupted by 
	some noise.  

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
		Builder method for creating a VOC_ObjDetect instance
		from a configuration dictionary.  The provided config file should have
		the following keys:

		* data_index : int, default = 0
			Image datapoint in the pascal voc dataset
		* patch_size : [int, int], default = [21, 21]
			size of the image patches that we need to split the image into to give one by one as input
		* noise : probability of sampling the distracting evidence over the patches

		Arguments
		---------
		config : dict
		    Dictionary representation of the environment.
		"""

		# The `categories` key is the only requrement in the config file.
		# Make sure that it exists, and raise an error if it doesn't
		# if not "data_index" in config:
		# 	raise AttributeError("Missing data index in config dictionary.")

		if not "categories" in config:
			raise AttributeError("Missing number of categories in config dictionary.")

		# Extract the observation mode, which should be a string, and convert
		# to the enumerated value.

		# Bulid the environment with the configuration
		return cls(config["categories"],
				   data_index=config.get("data_index", None),
			       patch_size=config.get("patch_size", [21, 21]),
			       epsilon=config.get("noise", 0.2))


	class ObservationMode(enum.IntEnum):
		"""
		ObservationMode is used to indicate how observations should be encoded.
		The available observation modes are defined as:

		* CATEGORY: provides the category name as an observation
		* INDEX: provides the index of the category as an observation
		* ONE_HOT: provides a one-hot encoding of the category as an observation
		"""

		CATEGORY = enum.auto()
		INDEX = enum.auto()
		ONE_HOT = enum.auto()


	def __init__(self, categories, **kwargs):
		"""
		Arguments
		---------
		categories

		Keyword Arguments
		-----------------
		noise : float, default=0.7
			Noise level of the environment (change of producing incorrect 
			observation for target)
		data_index : int, default=0
		    Image datapoint in the pascal voc dataset
		patch_size : ObservationMode, default=ObservationMode.ONE_HOT
		    Size of the image patches that we need to split the image into to give one by one as input
		"""

		Environment.__init__(self, **kwargs)

		# Store or create the categories -- if an integer is passed, then 
		# interpret this as the number of categories desired.
		# if type(data_index) is int:
		# 	self._data_index = data_index
		# else:
		# 	self.logger.error("%s:  Unable to get any data point: %s", self, data_index)
		# 	raise TypeError("Index for the dataset expected")

		self._categories = categories
		if not type(self._categories) is int:
			self.logger.warning("%s:  Invalid categories : %0.2f.  Ctegories must be integre values.", self, categories)
			self._categories = 20

		self._noise = kwargs.get("noise", 0.0)
		if self._noise < 0.0 or self._noise > 1.0:
			self.logger.warning("%s:  Invalid noise level: %0.2f.  Noise must be in range [0,1].  Noise value is capped.", self, self._noise)
			self._noise = min(max(self._noise, 0.0), 1.0)

		self._patch_size = kwargs.get("patch_size", [21, 21])
		if type(self._patch_size) is not list or len(self._patch_size) != 2:
			self.logger.error("%s: patch_size should be of a list of the form [a, b] %s", self, self._patch_size)
			raise TypeError("Incorrect form of patch_size")

		if np.any(np.asarray(self._patch_size) > 500) or np.any(np.asarray(self._patch_size) < 0):
			self.logger.warning("%s:  Invalid patch size : %0.2f.  Must be of the form [a, b] such that a and b are in range [0,500].", self, self._patch_size)
			self._patch_size = [21, 21]

		# number of datapoints in PASCAL VOC dataset
		voc_data = tfds.as_numpy(tfds.load(
						'voc',
						split='train',
						batch_size=-1,
						))

		self._num_datapoints = voc_data["image"].shape[0]
		# self._num_datapoints = len(voc_limited_ind)

		# The environment will keep track of how many observations were made
		self._num_observations = 0

		limited_ind = []
		# allowed_classes = None
		allowed_classes = {6:0, 8:1, 14:2}
		class_wts = [17, 13, 9, 12, 7, 20, 3, 14, 3, 14, 18, 9, 13, 14, 1, 8, 14, 12, 17, 14]

		for i in range(voc_data["image"].shape[0]):
			box_list_copy = copy.deepcopy(voc_data["objects"]["bbox"][i])
			bb_count = np.argmin(np.sum(box_list_copy, axis=1))
			for j in range(bb_count):
				label_value = voc_data["objects"]["label"][i][j]
				if label_value in allowed_classes.keys():
					limited_ind += ([(i, j)] * class_wts[label_value])

		self._limited_ind = limited_ind
		self._voc_data = voc_data
		# _data_index, _noise, _patch_size, _image, _filename, _labels, _labels_no_difficult, _objects, _num_datapoints, image_dim
		# Reset the environment to set up the target value
		# print("Noise - ", self._noise)
		self.reset(kwargs.get("data_index",None))

	@property
	def voc_data(self):
		return self._voc_data

	@property
	def limited_ind(self):
		return self._limited_ind

	@property
	def data_index(self):
		return self._data_index

	@property
	def image_bbox_pair(self):
		return self._image_bbox_pair

	@property
	def categories(self):
		return self._categories

	@property
	def num_datapoints(self):
		return self._num_datapoints

	@property
	def num_observations(self):
		return self._num_observations
	
	@property
	def patch_size(self):
		return self._patch_size

	@property
	def image(self):
		return self._image

	@property
	def filename(self):
		return self._filename

	@property
	def labels(self):
		return self._labels
	
	@property
	def labels_no_difficult(self):
		return self._labels_no_difficult

	@property
	def image_dim(self):
		return self._image_dim

	@property
	def objects(self):
		return self._objects
	
	@property
	def noise(self):
		return self._noise

	@property
	def target_index(self):
		return self._target_index

	@property
	def target_value(self):
		return self._target_value

	@property
	def target(self):
		return np.array([self._target_index])
		
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


	def reset(self, data_index=None):
		"""
		Reset the environment.  Uses the data_index if it is given, otherwise, picks
		a data point at random.

		Arguments
		---------
		data_index : int, default=None
			Target value or index to reset to.  If not provided, then picks a 
			random data point.
		"""

		# Data point sampled for this episode
		# if voc_data is None:
		# voc_data = tfds.as_numpy(tfds.load(
		# 			'voc',
		# 			split='train',
		# 			batch_size=-1,
		# 			))

		voc_data = copy.deepcopy(self.voc_data)

		# limited_ind = []
		# allowed_classes = None
		# # allowed_classes = {6:0, 14:1}
		# class_wts = [17, 13, 9, 12, 7, 20, 3, 14, 3, 14, 18, 9, 13, 14, 1, 8, 14, 12, 17, 14]
		# # image_bbox_pair = []
		# class_count = [0, 0]

		# for i in range(self.voc_data["image"].shape[0]):
		# 	bb_count = np.argmin(np.sum(self.voc_data["objects"]["bbox"][i], axis=1))
		# 	for j in range(bb_count):
		# 		label_value = self.voc_data["objects"]["label"][i][j]
		# 		limited_ind += ([(i, j)] * class_wts[label_value])
		# 	# bbox_ind = self.largest_box(voc_data["objects"]["bbox"][i])
		# 	# label_val = voc_data["objects"]["label"][i][bbox_ind]
		# 	# if label_val in allowed_classes.keys()and class_count[allowed_classes[label_val]]<=250:
		# 	# 	limited_ind.append(i)
		# 	# 	class_count[allowed_classes[label_val]] += 1

		# print("self.voc_data[image] shape ", self.voc_data["image"].shape)
		# print("limited_ind len ", len(self.limited_ind))
		self._num_datapoints = len(self.limited_ind)

		# No data_index provided -- select one at random
		if data_index is None:
			self._data_index = np.random.randint(0, self.num_datapoints)

		# Integer provided that is not in the range of dataset size -- assume used as index
		elif type(data_index) is int and data_index < self.num_datapoints :
			# Check to see if the given data_index can be used as an index
			try:
				self._data_index = data_index
			except IndexError as e:
				self.logger.error("%s:  Data index assumed, out of bounds: %d", self, data_index)
				raise e 
		
		# Last Case -- assume the target provided is in the list of categories
		else:
			self.logger.error("%s:  Data index assumed because provided data_index out of bounds or not an int: %d", self, data_index)
			self._data_index = np.random.randint(0, self.num_datapoints)

		if self.limited_ind is not None:
			self._image_bbox_pair = self.limited_ind[self._data_index]
		else:
			print("ERROR!! - limited_ind is None")

		# dict_keys(['image', 'image/filename', 'labels', 'labels_no_difficult', 'objects'])
		# objects = dict_keys(['bbox', 'is_difficult', 'is_truncated', 'label', 'pose'])
		self._image = voc_data["image"][self._image_bbox_pair[0]]
		self._filename = voc_data["image/filename"][self._image_bbox_pair[0]]
		self._labels = voc_data["labels"][self._image_bbox_pair[0]]
		self._labels_no_difficult = voc_data["labels_no_difficult"][self._image_bbox_pair[0]]
		self._objects = voc_data["objects"]

		# Just considering the largest box in the image
		# bbox_ind = self.largest_box(voc_data["objects"]["bbox"][self._image_bbox_pair[0]])
		bbox_ind = self._image_bbox_pair[1]
		# print("Data ind = ", self.image_bbox_pair)
		# print("voc_data[object][bbox] shape ", voc_data["objects"]["bbox"].shape)
		self._objects["bbox"] = voc_data["objects"]["bbox"][self._image_bbox_pair[0]][bbox_ind]
		self._objects["is_difficult"] = voc_data["objects"]["is_difficult"][self._image_bbox_pair[0]][bbox_ind]
		self._objects["is_truncated"] = voc_data["objects"]["is_truncated"][self._image_bbox_pair[0]][bbox_ind]
		self._objects["label"] = voc_data["objects"]["label"][self._image_bbox_pair[0]][bbox_ind]
		self._objects["pose"] = voc_data["objects"]["pose"][self._image_bbox_pair[0]][bbox_ind]
		self._image_dim = self.image_size(self._image)

		allowed_classes = {6:0, 8:1, 14:2}
		if allowed_classes is not None:
			self._target_index = allowed_classes[self.objects["label"]]
		else:
			self._target_index = self.objects["label"]
		# self._target_value = self._target_index

		# Set the number of observations to zero
		self._num_observations = 0
		
		'''
		Step 1 : Given the image and the patch_size, sample a patch from the ouding box or 
				 outside it based on the noise probability
		Step 2 : Encode the sampled patch of image using an FCN/CNN layer and then return
		'''

		'''  METHOD 1 ---- Sampling from a grid of patches based on the overlap with the largest bounding box!
		image_h, image_w = self.image_size(self._image)
		[H, W, C] = self._image.shape

		# Assumption - patch_size is a factor of image size (500)
		patch_points_row = np.arange(0, H, self._patch_size[0])
		patch_points_col = np.arange(0, W, self._patch_size[1])
	
		bbox_coord = self._objects["bbox"]
		y1, x1, y2, x2 = np.round(bbox_coord[0]*image_h), np.round(bbox_coord[1]*image_w), np.round(bbox_coord[2]*image_h), np.round(bbox_coord[3]*image_w)
		# print("Bbox coord = ", y1, x1, y2, x2)
		in_bbox = []
		out_bbox = []
		for i in patch_points_row:
			i_right = i + self._patch_size[0]
			for j in patch_points_col:
				j_right = j + self._patch_size[1]
				areaI = self.area_intersection([y1, x1], [y2, x2], [i, j], [i_right, j_right])
				if areaI > 0.3 * min(self._patch_size[0] * self._patch_size[1], (y2-y1)*(x2-x1)) :
					in_bbox.append([i, j])
				else:
					out_bbox.append([i, j])
		self._in_bbox = in_bbox
		self._out_bbox = out_bbox
		'''
		# image_h, image_w = self.image_size(self._image)
		# [H, W, C] = self._image.shape
		# bbox_coord = self._objects["bbox"]



	def area_intersection(self, b_left, b_right, p_left, p_right):
		x_dist = (min(b_right[0], p_right[0]) - max(b_left[0], p_left[0]))
		y_dist = (min(b_right[1], p_right[1]) - max(b_left[1], p_left[1]))
		areaI = 0
		# condition for overlap of rectangles
		if x_dist > 0 and y_dist > 0: 
			areaI = x_dist * y_dist
		return areaI

	def image_size(self, img):
		row, column = 0, 0
		for i in range(img.shape[0]):
			if not np.all(img[i, :, :] == 0):
				row = i
		
		for j in range(img.shape[1]):
			if not np.all(img[:, j, :] == 0):
				column = j
		return [row+1, column+1]

	# bbox_list - [41,4] size vector of boxe coordinates
	def largest_box(self, bbox_list):
		return np.argmax([(bb[2] -  bb[0]) * (bb[3] -  bb[1]) for bb in bbox_list])

	def observe(self, observation_mode=None):
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

		self._num_observations += 1

		# Step 1 : Given the image and the patch_size, sample a patch from the ouding box or 
		# 		   outside it based on the noise probability
		# Step 2 : Encode the sampled patch of image using an FCN/CNN layer and then return

		image_h, image_w = self.image_dim[0], self.image_dim[1]
		# [H, W, C] = self._image.shape

		# # Assumption - patch_size is a factor of image size (500)
		# patch_points_row = np.arange(0, H, self._patch_size[0])
		# patch_points_col = np.arange(0, W, self._patch_size[1])
	
		bbox_coord = self._objects["bbox"]
		y1, x1, y2, x2 = np.round(bbox_coord[0]*image_h), np.round(bbox_coord[1]*image_w), np.round(bbox_coord[2]*image_h), np.round(bbox_coord[3]*image_w)
		
		'''  METHOD 1 ---- Sampling from a grid of patches based on the overlap with the largest bounding box!
		if len(self._in_bbox) == 0:
			print("Zero in bounding box!!! - ", y1, x1, y2, x2)
			
		r_num = random.random()
		if len(self._out_bbox) > 0 and (r_num < self._noise or len(self._in_bbox) == 0):
			r_ind = random.randrange(len(self._out_bbox))
			patch_index = self._out_bbox[r_ind]
		else:
			r_ind = random.randrange(len(self._in_bbox))
			patch_index = self._in_bbox[r_ind]

		'''
		pad_x, pad_y = int(self._patch_size[1]/2), int(self._patch_size[0]/2)
		# print("Noise - ", self._noise)
		image_padded = np.pad(self._image, [(pad_y, pad_y), (pad_x, pad_x), (0, 0)], mode='constant')

		# inbox_x = [max(0. + np.ceil(self._patch_size[1]/2.), x1), min(image_w-np.ceil(self._patch_size[1]/2.), x2)]
		# inbox_y = [max(0. + np.ceil(self._patch_size[0]/2.), y1), min(image_h-np.ceil(self._patch_size[0]/2.), y2)]

		# all_x = [0. + np.ceil(self._patch_size[1]/2.), image_w-np.ceil(self._patch_size[1]/2.)]
		# all_y = [0. + np.ceil(self._patch_size[1]/2.), image_w-np.ceil(self._patch_size[1]/2.)]

		out_x = np.hstack((np.arange(0, x1), np.arange(x2, image_w)))
		out_y = np.hstack((np.arange(0, y1), np.arange(y2, image_h)))
		# out_y = np.hstack((np.arange(all_y[0], inbox_y[0]), np.arange(inbox_y[1], all_y[1])))

		r_num = random.random()
		no_in_points = (x2 - x1 <1) or (y2- y1 <1)
		no_out_points = out_x.shape[0]==0 or out_y.shape[0]==0

		if no_in_points and no_out_points:
			print("No in or out points for the given patch and box size", self.patch_size, " box coord = " ,y1, x1, y2, x2)

		if not(no_out_points) and ((r_num < self._noise) or no_in_points):
			r_x = random.randrange(out_x.shape[0])
			r_y = random.randrange(out_y.shape[0])
			patch_index = [int(out_x[r_x]), int(out_y[r_y])]
		else:
			r_x = random.randrange(x1, x2)
			r_y = random.randrange(y1, y2)
			patch_index = [int(r_x), int(r_y)]

		img_patch = np.array(image_padded[patch_index[1] : patch_index[1] + self._patch_size[1], patch_index[0] : patch_index[0] + self._patch_size[0], :])
		# img_patch = np.array(self._image[patch_index[0] : patch_index[0] + self._patch_size[0], patch_index[1] : patch_index[1] + self._patch_size[1], :])
		# print("patch size - ", patch_index)
		# print("self._image_bbox_pair - ", self._image_bbox_pair)
		# plt.imshow(img_patch)
		# plt.show()
		img_patch = img_patch*1.0
		# resize image patch to patch_size*patch_size of it is smaller
		# if (patch_index[0] + self._patch_size[0] > self._image.shape[0]) or (patch_index[1] + self._patch_size[1] > self._image.shape[1]):
		# 	# print("Patch index - ", patch_index)
		# 	# print("Shape of image patch ", img_patch.shape)
		# 	img_patch = resize(img_patch, (self._patch_size[0], self._patch_size[1]))
		return np.expand_dims(img_patch, axis=0)


	def properties(self):
		"""
		Returns a dictionary representation of an instance of the environment,
		containing the following fields:

		# _categories, _data_index, _noise, _patch_size, _image, _filename, _labels, _labels_no_difficult, _objects, _num_datapoints, _image_dim
		
		* name : the name of the environment
		* class : a fully-qualified class name of the environment
		* categories : number of classes to classify into
		* data_index : index of the data point from the VOC dataset
		* num_datapoints : total size of VOC train dataset
		* noise : the noise level of the environment
		* patch_size : size of patch to be sampled at each instance in an episode
		* image : the current image 
		* filename : name of the image file
		* lables : numpy array of list of lables for objects in this image (n,0), where n is number of objects identified in this image
		* labels_no_difficult : 
		* objects : dictionary with dict_keys(['bbox', 'is_difficult', 'is_truncated', 'label', 'pose'])

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
				 "data_index" : self.data_index,
		         "num_datapoints" : self.num_datapoints,
				 "noise" : self.noise,
				 "patch_size" : self.patch_size,
				 "image" : self.image,
				 "filename" : self.filename,
				 "labels" : self.labels,
				 "labels_no_difficult" : self.labels_no_difficult,
				 "objects" :  self.objects,
				 "target_value": self.target_value,
		         "target_index": self.target_index,
				 "image_dim" : self.image_dim,
				 "image_bbox_pair" : self.image_bbox_pair,
				 "voc_data" : self.voc_data,
				 "limited_ind" : self.limited_ind
		       }
