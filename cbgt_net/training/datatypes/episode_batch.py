# -*- coding: utf-8 -*-
"""
.. module:: training.datatypes.episode_batch
   :platform: Linux, Windows, OSX
   :synopsis: EpisodeBatch definition

.. moduleauthor:: AARTLab <danahugh@andrew.cmu.edu>

Definition of a simple datatype for handling a batch of generated episode data
during trianing.
"""

import tensorflow as tf
from typing import Mapping, Optional

from .reward_batch import RewardBatch
from .loss_batch import LossBatch


class EpisodeBatch(tf.experimental.ExtensionType):
	"""
	An EpisodeBatch is a data structure class for collecting data generated
	by an episode during training.

	Unless otherwise noted, the shape of all Tensor attributes will start with
	(episode_length, batch_size).  Remaining shape is dependent on the
	environment or model output.

	Methods
	-------
	set_rewards(rewards)
		Returns a copy of the EpisodeBatch with the return attribute set
	set_losses(losses)
		Returns a copy of the EpisodeBatch with the loss attribute set
	add_attribute(name, value)
		Returns a copy of the EpisodeBatch with an added named attribute
	add_attributes(attributes)
		Returns a copy of the EpisodeBatch with multiple named attributes


	Attributes
	----------
	observations : tf.Tensor
		Observations generated by the environment
	targets : tf.Tensor
		Target values generated by the environment
	evidence : tf.Tensor
		Evidence at each time step produced by the model
	accumulators : tf.Tensor
		Accumulated evidence at each time step in the model
	decision_probabilities : tf.Tensor
		Probability of each decision generated by the model
	did_decide_probabilities : tf.Tensor
		Probability that the model signals a decision to be made
	did_decide : tf.Tensor
		Indicator of whether the model made a decision
	decisions : tf.Tensor
		Decision made by the model
	decision_masks : tf.Tensor
		Indicator of whether the model had previously made a decision	
	rewards : RewardBatch, optional, default=None
		Reward generated by the trainer
	losses : LossBatch, optional, default=None
		Loss generated by the trainer
	attributes : Mapping[str, tf.Tensor], default={}
		Additional named attributes
	"""

	observations : tf.Tensor
	targets : tf.Tensor
	evidence : tf.Tensor
	accumulators : tf.Tensor
	decision_probabilities : tf.Tensor
	did_decide_probabilities : tf.Tensor
	did_decide : tf.Tensor
	decisions : tf.Tensor
	decision_masks : tf.Tensor
	rewards : Optional[RewardBatch] = None
	losses : Optional[LossBatch] = None
	attributes : Mapping[str, tf.Tensor] = {}

	@property
	def batch_size(self):
		return self.observations.shape[1]

	@property
	def num_timesteps(self):
		return self.observations.shape[0]


	def set_rewards(self, rewards):
		"""
		Returns a copy of this EpisodeBatch, with the reward attribute set to 
		the provided RewardBatch.

		Arguments
		---------
		reward : RewardBatch
			RewardBatch to add to this EpisodeBatch
		"""

		return EpisodeBatch( observations = self.observations,
			                 targets = self.targets,
			                 evidence = self.evidence,
			                 accumulators = self.accumulators,
			                 decision_probabilities = self.decision_probabilities,
			                 did_decide_probabilities = self.did_decide_probabilities,
			                 did_decide = self.did_decide,
			                 decisions = self.decisions,
			                 decision_masks = self.decision_masks,
			                 rewards = rewards,
			                 losses = self.losses,
			                 attributes = self.attributes )


	def set_losses(self, losses):
		"""
		Returns a copy of this EpisodeBatch, with the loss attribute set to the
		provided LossBatch.

		Arguments
		---------
		losses : LossBatch
			LossBatch to add to this EpisodeBatch
		"""

		return EpisodeBatch( observations = self.observations,
			                 targets = self.targets,
			                 evidence = self.evidence,
			                 accumulators = self.accumulators,
			                 decision_probabilities = self.decision_probabilities,
			                 did_decide_probabilities = self.did_decide_probabilities,
			                 did_decide = self.did_decide,
			                 decisions = self.decisions,
			                 decision_masks = self.decision_masks,
			                 rewards = self.rewards,
			                 losses = losses,
			                 attributes = self.attributes )


	def add_attribute(self, name, value):
		"""
		Return a copy fo this EpisodeBatch with a named attribute added.  If
		multiple attributes are to be added, `add_attributes` should be used
		instead, as it is more efficient.

		Arguments
		---------
		name : str
		    Name of the attribute to add
		value : tf.Tensor
		    Value of the attribute to add
		"""

		return EpisodeBatch( observations = self.observations,
			                 targets = self.targets,
			                 evidence = self.evidence,
			                 accumulators = self.accumulators,
			                 decision_probabilities = self.decision_probabilities,
			                 did_decide_probabilities = self.did_decide_probabilities,
			                 did_decide = self.did_decide,
			                 decisions = self.decisions,
			                 decision_masks = self.decision_masks,
			                 rewards = self.rewards,
			                 losses = self.losses,
			                 attributes = { name: value, **self.attributes } )


	def add_attributes(self, attributes):
		"""
		Return a copy of this EpisodeBatch with named attributes added.  

		Arguments
		---------
		attributes : Mapping[str, tf.Tensor]
		    Attributes to add
		"""

		return EpisodeBatch( observations = self.observations,
			                 targets = self.targets,
			                 evidence = self.evidence,
			                 accumulators = self.accumulators,
			                 decision_probabilities = self.decision_probabilities,
			                 did_decide_probabilities = self.did_decide_probabilities,
			                 did_decide = self.did_decide,
			                 decisions = self.decisions,
			                 decision_masks = self.decision_masks,
			                 rewards = self.rewards,
			                 losses = self.losses,
			                 attributes = { **attributes, **self.attributes } )


