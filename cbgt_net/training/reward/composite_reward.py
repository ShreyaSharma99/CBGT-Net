# -*- coding: utf-8 -*-
"""
.. module:: composite_reward.py
   :platform: Linux, Windows, OSX
   :synopsis: Class for generating a single reward from individual rewards

.. moduleauthor:: AARTLab <danahugh@andrew.cmu.edu>

A reward class for generating a reward from a weighted combination of
individual rewards.
"""

import tensorflow as tf

from ..datatypes import RewardBatch


class CompositeReward:
   """
   The CompositeReward class defines a class for aggregating a set of weighte
   rewards to a single reward.


   Methods
   -------
   add_reward
      Add a named and optionally weighted subreward to the composite reward
   reweight
      Change the weight of one of the named subrewards


   Usage
   -----
   When created, the CompositeReward will not contain any rewards.  These need
   to be individually added through the `add_reward` method.  Each added reward
   needs to have a _unique_ name, and optionally a weight::

       >>> composite_reward = CompositeReward()
       >>> reward1 = RewardType1()
       >>> reward2 = RewardType2()
       >>> composite_reward.add_reward("some_reward", reward1)
       >>> composite_reward.add_reward("other_reward", reward2, 0.7)

   If the composite_reward is called without any added rewards, it will raise
   an exception.

   The weight of individual subrewards can be reset using the `reweight`
   method::

       >>> composite_reward.reweight("some_reward", 1.2)
   """

   def __init__(self, **kwargs):
      """
      """

      # Rewards will be mapped from a string reward name to reward instances
      # and weights.  Reward weights will map named rewards to tf.Variable
      # to allow for calling via tf.function
      self._rewards = {}
      self._weights = {}


   def __repr__(self):
      """
      Representation of the CompositeReward
      """

      representation = [f'{self.__class__.__name__}:']

      # Add the representation of the subrewards
      for name in self._rewards:
         reward = self._rewards[name]
         weight = self._weights[name].numpy()
         representation.append(f'  {name} (weight={weight}): {repr(reward)}')

      # Join the individual representations into a single string and return
      return "\n".join(representation)


   @tf.function
   def reset(self):
      """
      """

      pass


   def add_reward(self, name, reward, weight=1.0):
      """
      Add a reward to the composite reward.

      Arguments
      ---------
      name : string
         String name of the reward
      reward : Reward
         Reward instance
      weight : double, default=1.0
         Weight for the reward
      """

      # Check if the reward is in the rewards dictionary, and raise a warning
      # that it is being replaced if so
      if name in self._rewards:
         pass

      # Create a named tuple of the reward and add to the rewards
      self._rewards[name] = reward
      self._weights[name] = tf.Variable(weight, dtype=tf.float32)


   def reweight(self, name, weight):
      """
      Set the weight of an existing subreward.

      Arguments
      ---------
      name : string
         Name of the subreward to change weight
      weight : double
         New weight of the subreward
      """

      # Check if the subreward exists in the rewards dictionary, and issue a 
      # warning if not
      if not name in self._reward_weights:
         # TODO: Issue warning
         return

      self._weights[name].assign(weight)


   @tf.function
   def __call__(self, episode):
      """
      Calculate and return the reward.  For each subreward, the subreward will 
      be added to the RewardBatch subrewards attribute with its name as a key.

      Arguments
      ---------
      episode : EpisodeBatch
         Batch of episodes to calculate reward for
      """

      total_rewards = 0

      subrewards = {}

      # Calculate all the reward values, updating the total reward value
      # in parallel
      for name in self._rewards:
         subreward = self._rewards[name](episode)
         subrewards[name] = subreward

         total_rewards = total_rewards + self._weights[name] * subreward.rewards

      # Construct a RewardBatch with the final total reward and all subrewards
      return RewardBatch(rewards = total_rewards, subrewards=subrewards)
