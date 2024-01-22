# -*- coding: utf-8 -*-
"""
.. module:: experiment.py
   :platform: Linux, Windows, OSX
   :synopsis: Definition of a class to encapsulate experiments

.. moduleauthor:: AARTLab <danahugh@andrew.cmu.edu>

This module defines an experiment class, which manages the configuration,
running, reporting and lifecycle of an experiment.
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import yappi
import pathlib
import uuid
import argparse
import warnings
import tensorflow as tf
# import wandb

from . import CBGT_Net
from .utils import DependencyInjection as DI
from .utils import Loggable, Observable, ConfigLoader
### from .observers import ConsoleProgressMonitor

from .training.reward import CompositeReward
from .training.losses import CompositeLoss
import time


class Experiment(Loggable, Observable):
	"""
	The Experiment class is designed to manage the configuration, running,
	reporting, and lifecycle of an atomic experiment.  The purpose of this
	class is to allow experiments to be represented as a dictionary, which will
	be used to construct the components of the experiment, and to generate
	report results when the experiment is run.

	Usage
	-----
	Assuming an experiment is defined as a JSON file, typical use of this class
	is as follows::

	>>> config = json.load("experiment.json")
	>>> with Experiment(config) as experiment:
	>>>     experiment.run()
	"""

	def __init__(self, config):
		"""
		Arguments
		---------
		config : dict, string, or pathlib.Path
		    Dictionary containing experiment configuration, or a path to a JSON
		    file containing the same.
		"""

		# TODO:  As part of the experiment config, have a field to limit GPU
		#        usage.

		# Store the configuration file for use as needed, and generate an
		# experiment id if needed.  Will generate a UUID if no experiment ID
		# is provided
		self._config = self.__load_config(config)

		self._id = self._config.get("experiment_id", None)
		if self._id is None:
			self._id = str(uuid.uuid4())

		# TODO:  Validate that the config is a dictionary and has all the
		#        required fields

		# Set up a logger and observer functionality
		Loggable.__init__(self, **self._config)
		Observable.__init__(self, **self._config)

		# Placeholders for the needed components
		self._environment = None
		self._model = None
		self._trainer = None
		self.checkpoint_file = None

		# Attribute indicating if the experiment components have been
		# constructed.  The experiment is lazily constructed to allow for
		# adding, e.g., observers to various components or the experimenter.
		self._constructed = False


	@property
	def id(self):
		return self._id

	@property
	def config(self):
		return self._config

	@property
	def environment(self):
		return self._environment

	@property
	def model(self):
		return self._model

	@property
	def trainer(self):
		return self._trainer


	def __load_config(self, config):
		"""
		Loads the config, if needed, from a provided path
		"""

		# If `config` is already in a dictionary format, then no need 
		# for loading
		if isinstance(config, dict):
			return config

		# `config` was not a dictionary, so assume that the provided argument 
		# is a path.  If it's a string, convert it to a pathlib.Path
		if type(config) is str:
			config = pathlib.Path(config)

		# Check to see if `self._config` refers to a path.  If not, raise an error
		if not isinstance(config, pathlib.Path):
			raise TypeError("Provided argument is not string, dictionary, or pathlib.Path: %s" % str(config))

		# Check to see if the provided path exists, and if it's a file
		if not config.exists():
			raise FileNotFoundError(str(config))
		if not config.is_file():
			raise IsADirectoryError("Provided path is not a file: %s" % str(config))

		# Try to load the contents of the file as a JSON string
		try:
			config_loader = ConfigLoader()
			return config_loader.load(config)

		# The contents of the JSON file were not parsable for some reason
		except Exception as e:
			raise e
	

	def __construct_environment(self, config):
		"""
		Construct the environment based on the contents of the provided config.

		Arguments
		---------
		config : dict
		   Dictionary containing configuration of the environment
		"""
		_, EnvironmentClass = DI.get_module_and_class(config["class"])
		return EnvironmentClass.build(config["properties"])


	def __construct_model(self, config):
		"""
		Construct the environment based on the contents of the provided config.

		Arguments
		---------
		config : dict
		   Dictionary containing configuration of the model
		"""

		num_categories = config["num_categories"]
		batch_size = config["batch_size"]

		# Create the needed submodules, if provided.  Otherwise, use the
		# default values
		evidence_module = None
		accumulator_module = None
		threshold_module = None

		if config["evidence_module"] is not None:
			evidence_module = DI.create_instance(config["evidence_module"]["class"],
				                                 *(num_categories,),
				                                 **{ 'batch_size': batch_size, 
				                                      **config["evidence_module"]["properties"]})

		if config["accumulator_module"] is not None:
			accumulator_module = DI.create_instance(config["accumulator_module"]["class"],
				                                    *(num_categories,),
				                                    **{ 'batch_size': batch_size, 
				                                        **config["accumulator_module"]["properties"]})

		if config["threshold_module"] is not None:
			threshold_module = DI.create_instance(config["threshold_module"]["class"],
				                                  *(),
				                                  **{ 'batch_size': batch_size, 
				                                      **config["threshold_module"]["properties"]})

		return CBGT_Net(num_categories,
			            evidence_module,
			            accumulator_module,
			            threshold_module,
			            batch_size=batch_size,
			            **config.get("properties",{}))


	def __construct_reward(self, configs):
		"""
		Construct a reward function for the trainer based on the provided config.
		A list of configs needs to be provided, if one reward configuration is in
		the list, then it will be constructed and returned.  If multiple are
		provided, a CompositeReward will be constructed with each configuration
		being used to construct a subreward.

		Arguments
		---------
		configs : list
			List of dictionaries containing configuration of reward functions
		"""

		# Create the reward function(s), storing the reward and its corresponding
		# weight and name (if provided).  The default name needs to be unique, so
		# it will be given a name "number<N>", where _N_ is the order of the 
		# reward in the sequence.
		rewards = []
		for config in configs:
			reward = DI.create_instance(config["class"],
				                         *(),
				                         **config.get("properties", {}))
			weight = config.get("weight", 1.0)
			name = config.get("name", f"number{len(rewards)}")
			rewards.append((reward, weight, name))

		# If there's only one reward in the config, then simply use that reward.
		# Otherwise, create a CompositeReward with each of the rewards.  If no 
		# reward was provided, raise an error
		if len(rewards) == 0:
			self.logger.error("No reward provided for the trainer")
			# TODO: Raise an exception here
			return None
		elif len(rewards) == 1:
			reward = rewards[0][0]
		else:
			reward = CompositeReward()
			for reward_tuple in rewards:
				_reward, _weight, _name = reward_tuple
				reward.add_reward(_name, _reward, _weight)

		return reward


	def __construct_loss(self, configs):
		"""
		Construct a loss function for the trainer based on the provided configs.
		A list of configs needs to be provided.  If a single loss configuration
		is in the list, then it well be constructed and returned.  Otherwise, the
		multiple lossess will be aggregated into a CompositeLoss instance.

		Arguments
		---------
		configs : list
			List of dictionaries containing configuration of the loss functions
		"""

		# Create loss function(s) and store their weight and name, if provided.
		# Since names need to be unique, default to "number<N>", where _N_ is
		# unique (though not necessarily sequential).
		losses = []
		for config in configs:
			loss = DI.create_instance(config["class"], 
				                       *(), 
				                       **config.get("properties", {}))
			weight = config.get("weight", 1.0)
			name = config.get("name", f"number{len(losses)}")
			losses.append((loss, weight, name))

		# If there's only one loss in the config, then simply return that.
		# If none are provided, raise an exception
		# Otherwise, create an appropriate CompositeLoss
		if len(losses) == 0:
			self.logger.error("No loss provided for the trainer")
			# TODO: Raise an exception here
			return None
		elif len(losses) == 1:
			loss = losses[0][0]
		else:
			loss = CompositeLoss()
			for loss_tuple in losses:
				_loss, _weight, _name = loss_tuple
				loss.add_loss(_name, _loss, _weight)

		return loss


	def __construct_trainer(self, model, environment, config):
		"""
		Construct the trainer based on the contents of the provided config

		Arguments
		---------
		model : CBGT_Net
		    Model to train
		environment : Environment
		    Environment to train upon
		config : dict
		    Dictionary containing configuration of the trainer
		"""

		# Create an optimizer, if one was provided
		if config.get("optimizer", None) is not None:
			optimizer = DI.create_instance(config["optimizer"]["class"],
				                           *(),
				                           **config["optimizer"].get("properties", {}))
		else:
			optimizer = None

		reward = self.__construct_reward(config.get("rewards", []))

		loss = self.__construct_loss(config.get("losses", []))

		# Create the trainer and return
		trainer =  DI.create_instance(config["class"],
												*(model, environment, reward, loss, optimizer),
			                     		**config.get("properties", {}))

		# Construct the measures, and add as either post training or post
		# evaluation measures, or both, depending on the "stage" property.  If no
		# "stage" property is provided, then assume that it is to be applied
		# after both training and evaluation.
		for measure_config in config.get("measures", []):
			measure = DI.create_instance(measure_config["class"],
				                          *(),
				                          **measure_config.get("properties", {}))
			stage = measure_config.get("stage", ["training", "evaluation"])

			if "training" in stage:
				trainer.add_post_training_measure(measure)
			if "evaluation" in stage:
				trainer.add_post_evaluation_measure(measure)

		return trainer



	def __construct_observers(self, observer_configs):
		"""
		Construct observers listed in the provided config, and register them
		with this experiment.

		Arguments
		---------
		observer_configs : list
		    List of configuration dictionaries for each observer to register
		"""

		for config in observer_configs:
			# Create the observer instance
			observer = DI.create_instance(config["class"],
				                          *(self,),
				                          **config.get("properties", {}))

			# Register the observer
			self.register_observer(observer, priority=config.get("priority", 1))


	def __construct(self):
		"""
		Construct the experiment, based on the contents of the config
		parameter.
		"""

		# If a random seed was provided, seed the RNGs
		if "random_seed" in self._config:
			tf.keras.utils.set_random_seed(self._config["random_seed"])

		# Create the required components
		self._environment = self.__construct_environment(self._config.get("environment", {}))
	
		self._model = self.__construct_model({'batch_size': self._environment.batch_size, 
			                                   **self._config.get("model", {})})
		self._trainer = self.__construct_trainer(self._model, 
			                                     self._environment, 
			                                     self._config.get("trainer", {}))

		# # # # ############
		checkpoint = tf.train.Checkpoint(model=self._model, optimizer=self.trainer.optimizer, step=tf.Variable(1))
		if not self.checkpoint_file is None:
			checkpoint_path = '/home/shreya/cbgt_net/logs_checkpoint/' + self.checkpoint_file

			manager = tf.train.CheckpointManager(checkpoint, checkpoint_path, max_to_keep=10)  # Replace max_to_keep with the desired number of checkpoints to keep
			# Restore the latest checkpoint
			checkpoint.restore(manager.latest_checkpoint)

			# Check if the checkpoint is successfully restored
			if manager.latest_checkpoint:
				print("Model restored from:", manager.latest_checkpoint)
			else:
				print("No checkpoint found.")

		# Create any observers
		self.__construct_observers(self._config.get("observers", []))

		# Indicate that the experiment has been constructed
		self._constructed = True


	def run(self):
		"""
		Run the experiment
		"""
		
		# raise Exception("This is a custom exception.")
	
		# If the experiment has not been constructed, construct the experiment
		if not self._constructed:
			self.__construct()


		# Keep track of how many episodes have been run, and when the next
		# evaluation should be
		current_episode = 1
		next_evaluation = self._config.get("evaluation_rate", None)
		early_stop = False

		# Run for the requisite number of steps
		while current_episode < self._config["num_episodes"] + 1 and not early_stop:
			
			episode = self._trainer.train()

			current_episode += episode.batch_size


			for observer in self._observers:
				observer.on_training_step(episode)


			# See if an evaluation needs to be run, and if so, then run one and
			# report it to any observers
			if next_evaluation is not None and current_episode >= next_evaluation:

				# Run an evaluation and update when to do the next one
				evaluation_episode = self.trainer.evaluate(self._config.get("num_evaluation_episodes", 1))
				next_evaluation += self._config["evaluation_rate"]

				for observer in self._observers:
					observer.on_evaluation(evaluation_episode)

					# check if early stopping condition is met in the tensorborad monitor
					if hasattr(observer, 'early_stop'):

						# print("I DID ENTERE!!!! ------------------------")
						if observer.early_stop:
							print(f'Early stopping after {current_episode} episodes!')
							early_stop = True


	def __enter__(self):
		"""
		Return the instance of the Experiment as a context
		"""

		# Construct the environment if it has not yet been done
		if not self._constructed:
			self.__construct()

		# Indicate to each observer that the experiment is starting
		for observer in self._observers:
			observer.on_experiment_start()

		return self


	def __exit__(self, exception_type, exception_value, exception_traceback):
		"""
		Completion of using the Experiment instance as a context.

		Arguments
		---------
		exception_type : type or None
		    Type of exception thrown while in context, or None if not thrownhreyu9990024s
		    Value of an exception thrown, or None if not thrown
		exception_traceback
		    Traceback of a thrown exception, or None if not thrown
		"""

		# Will want to pass the exception to observers, if one exists
		if exception_type is not None:
			exception = (exception_type, exception_value, exception_traceback)
		else:
			exception = None

		# Indicate to the observers that the experiment is completed, with the
		# exception if one was raised
		for observer in self._observers:
			observer.on_experiment_complete(exception)

		# If a KeyboardInterrupt was generated, suppress it so that it isn't 
		# propagated to the console
		return isinstance(exception_value, KeyboardInterrupt)

import json

if __name__ == "__main__":
	"""
	Runs an experiment from the command-line
	"""

	# Parse command line arguments
	parser = argparse.ArgumentParser(description="Run an experiment from command line using a config file.")
	# parser.add_argument('config_path', help="Path to the experiment configuration file.")
	parser.add_argument('--env', type=str, default='cifar', help='Specify the environment (cifar, mnist)')
	parser.add_argument('--patch_sz', type=int, default=16, help='Specify the environment (5, 8, 10, 12, 16, 20)')
	parser.add_argument('--threshold', type=int, default=4, help='Specify the environment (1, 2, 3, 4, 5)')
	parser.add_argument('--working_dir', '-wd', help="Path to working directory to run the experiment.")
	parser.add_argument('--append_id', '-id', action="store_true", help="Append the experiment ID to the working directory for the experiment.")
	args = parser.parse_args()

	exp = args.env
	patch_sz = args.patch_sz
	threshold = args.threshold

	print("Running CBGT-Net training for = ", exp)

	if exp == "mnist":
		org_patch_size = [patch_sz]
		patch_size = [28]
		fixed_thresh = [threshold]
		max_steps_per_episode = [int(threshold*10 + 1)]

		noise = [0.0]
		threshold_wt = [0.001]
		entropy_wt = [0]
		batch_size = [512]
		checkpoint_files = [None]

		activation = "softmax"
		
		config_path = "examples/mnist_patch.json"
		config_f = open(config_path)
		config_data = json.load(config_f)

		# wandb.login()

		for i in range(len(org_patch_size)):
			
			# exp = "Trial"
			exp = "mnist_ " + str(org_patch_size[i]) + "patch_" + str(fixed_thresh[i]) + "fixedThreshold"
			config_data["model"]["evidence_module"]["properties"]["input_shape"]= [patch_size[0], patch_size[0], 3]
			config_data["model"]["evidence_module"]["properties"]["output_activation"]= activation
			
			config_data["model"]["accumulator_module"]["properties"]["batch_size"]= batch_size[i]
			
			config_data["environment"]["properties"]["patch_size"] = [org_patch_size[i], org_patch_size[i], 3]
			# config_data["environment"]["properties"]["patch_size"] = [patch_size[0], patch_size[0], 1]

			config_data["environment"]["properties"]["noise"] = noise[0]
			config_data["environment"]["properties"]["batch_size"] = batch_size[i]
			config_data["environment"]["properties"]["max_steps_per_episode"] = max_steps_per_episode[i]
			config_data["environment"]["properties"]["images_per_class_train"] = 5421
			config_data["environment"]["properties"]["images_per_class_test"] = 892
			
			config_data["model"]["threshold_module"]["properties"]["decision_threshold"] = fixed_thresh[i]
			config_data["observers"][0]["properties"]["tb_logs_path"] = "logs/" + exp
			config_data["observers"][1]["properties"]["log_path"] = "logs/" + exp
			config_data["observers"][2]["properties"]["checkpoint_path"] = "logs/checkpoints/" + exp
			config_data["trainer"]["properties"]["max_steps_per_episode"] = max_steps_per_episode[i]
			config_data["trainer"]["losses"][1]["weight"] = threshold_wt[0]
			config_data["trainer"]["losses"][-1]["weight"] = entropy_wt[0]

			experiment = Experiment(config_data)
			experiment.checkpoint_file = checkpoint_files[i]
			
			# Create the working dir.  If the `append_id` argument is set, append the
			# experiment ID to the working dir.  If needed, create the working dir
			working_dir = args.working_dir if args.working_dir is not None else '.'
			working_dir = pathlib.Path(working_dir)

			if args.append_id:
				working_dir = working_dir / experiment.id

			if not working_dir.exists():
				working_dir.mkdir(parents=True, exist_ok=True)

			os.chdir(working_dir)

			# Run the experiment
			with experiment:
				experiment.run()

	elif exp == "cifar":
		
		org_patch_size = [patch_sz]
		patch_size = [32]
		fixed_thresh = [threshold]
		max_steps_per_episode = [int(threshold*10 + 1)]

		noise = [0.0]
		threshold_wt = [0.1]
		entropy_wt = [0]
		batch_size = 256 if threshold < 3  else 128
		if patch_sz==20 and threshold==2: batch_size = 128
		batch_size = [batch_size]

		checkpoint_files = [None]

		activation = "softmax"

		config_path = "examples/cifar_patch.json"
		config_f = open(config_path)
		config_data = json.load(config_f)

		# wandb.login()

		for i in range(len(org_patch_size)):
			
			exp = "cifar_ " + str(org_patch_size[i]) + "patch_" + str(fixed_thresh[i]) + "fixedThreshold"
			config_data["model"]["evidence_module"]["properties"]["input_shape"]= [patch_size[0], patch_size[0], 3]
			config_data["model"]["evidence_module"]["properties"]["output_activation"]= activation
			config_data["model"]["accumulator_module"]["properties"]["batch_size"]= batch_size[i]
			config_data["environment"]["properties"]["patch_size"] = [org_patch_size[i], org_patch_size[i], 3]

			config_data["environment"]["properties"]["noise"] = noise[0]
			config_data["environment"]["properties"]["batch_size"] = batch_size[i]
			config_data["environment"]["properties"]["max_steps_per_episode"] = max_steps_per_episode[i]
			config_data["environment"]["properties"]["images_per_class_train"] = 5000
			config_data["environment"]["properties"]["images_per_class_test"] = 1000
			
			config_data["model"]["threshold_module"]["properties"]["decision_threshold"] = fixed_thresh[i]
			config_data["observers"][0]["properties"]["tb_logs_path"] = "logs/tensorboard/" + exp
			config_data["observers"][1]["properties"]["log_path"] = "logs/tensorboard/" + exp
			config_data["observers"][2]["properties"]["checkpoint_path"] = "logs/checkpoints/" + exp
			config_data["trainer"]["properties"]["max_steps_per_episode"] = max_steps_per_episode[i]
			config_data["trainer"]["losses"][1]["weight"] = threshold_wt[0]
			config_data["trainer"]["losses"][-1]["weight"] = entropy_wt[0]

			# Create the experiment
			experiment = Experiment(config_data)
			experiment.checkpoint_file = checkpoint_files[i]
			
			# Create the working dir.  If the `append_id` argument is set, append the
			# experiment ID to the working dir.  If needed, create the working dir
			working_dir = args.working_dir if args.working_dir is not None else '.'
			working_dir = pathlib.Path(working_dir)

			if args.append_id:
				working_dir = working_dir / experiment.id

			if not working_dir.exists():
				working_dir.mkdir(parents=True, exist_ok=True)

			os.chdir(working_dir)

			# Run the experiment
			with experiment:
				experiment.run()