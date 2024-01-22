# -*- coding: utf-8 -*-
"""
.. module:: composite_loss.py
   :platform: Linux, Windows, OSX
   :synopsis: Class for generating a single loss from individual losses

.. moduleauthor:: AARTLab <danahugh@andrew.cmu.edu>

A loss class for generating a loss from a weighted combination of
individual losses.
"""

import tensorflow as tf

from ..datatypes import LossBatch


class CompositeLoss:
   """
   The CompositeLoss class defines a class for aggregating a set of weighted
   losses to a single loss.

   Methods
   -------
   add_loss
      Add a named and optionally weighted subloss to the composite loss
   reweight
      Change the weight of one of the named sublosses


   Usage
   -----
   When created, the CompositeLoss will not contain any losses.  These need
   to be individually added through the `add_loss` method.  Each added loss
   needs to have a _unique_ name, and optionally a weight::

       >>> composite_loss = CompositeLoss()
       >>> loss1 = LossType1()
       >>> loss2 = LossType2()
       >>> composite_loss.add_loss("some_loss", loss1)
       >>> composite_loss.add_loss("other_loss", loss2, 0.7)

   The weight of individual sublosses can be reset using the `reweight`
   method::

       >>> composite_loss.reweight("some_loss", 1.2)
   """


   def __init__(self, **kwargs):
      """
      """

      # Losses will be mapped from a string loss name to loss instances 
      # and weights.  Loss weights will map named losses to tf.Variable
      # to allow for calling via tf.function
      self._losses = {}
      self._weights = {}


   def __repr__(self):
      """
      Representation of the CompositeLoss
      """

      representation = [f'{self.__class__.__name__}:']

      # Add the representation of the subrewards
      for name in self._losses:
         loss = self._losses[name]
         weight = self._weights[name].numpy()
         representation.append(f'  {name} (weight={weight}): {repr(loss)}')

      # Join the individual representations into a single string and return
      return "\n".join(representation)


   def add_loss(self, name, loss, weight=1.0):
      """
      Add a loss to the composite loss.

      Arguments
      ---------
      name : string
         String name of the loss
      loss : Loss
         Loss instance
      weight : double, default=1.0
         Weight for the loss
      """

      # Check if the loss is in the lossess dictionary, and raise a warning
      # that it is being replaced if so
      if name in self._losses:
         pass

      # Create a named tuple of the loss and add to the losses
      self._losses[name] = loss
      self._weights[name] = tf.Variable(weight, dtype=tf.float32)


   def reweight(self, name, weight):
      """
      Set the weight of an existing subloss.

      Arguments
      ---------
      name : string
         Name of the subloss to change weight
      weight : double
         New weight of the subloss
      """

      # Check if the subloss exists in the losses dictionary, and issue a 
      # warning if not
      if not name in self._losses:
         # TODO: Issue warning
         return

      # The SubLoss is a namedtuple, and is not mutable, so we need to
      # create a new one.
      self._weights[name].assign(weight)


   @tf.function
   def __call__(self, episode, model=None):
      """
      Calculate and return the losses.  For each subloss, the subloss will be
      added to the LossBatch sublosses attribute with its name as a key.

      Arguments
      ---------
      episode : EpisodeBatch
         Batch of episodes to calculate losses for
      model : CBGT_Net, default=None
         Model being trained
      """

      total_losses = 0

      sublosses = {}

      # Calculate all the loss values, updating the total loss value in
      # parallel
      for name in self._losses:
         subloss = self._losses[name](episode, model)
         sublosses[name] = subloss

         # if name == "tanh":
         #    # tf.print("Epoch --------------------------------------------------------------- ", episode.attributes["epoch"])
         #    # wt_upadted = self._weights[name]/tf.math.sqrt(tf.cast(episode.attributes["epoch"], tf.float32)) + 0.01
         #    # wt_upadted = 0.01*tf.math.log(tf.cast(episode.attributes["epoch"], tf.float32)) + 0.001
         #    wt_upadted = self._weights[name]*tf.math.exp(-tf.math.pow(tf.cast(episode.attributes["epoch"], tf.float32), 0.05)) + 0.001
         #    total_losses = total_losses + wt_upadted * subloss.losses
         #    tf.print("wt_upadted = ", wt_upadted)
         # else:
         total_losses = total_losses + self._weights[name] * subloss.losses

      # harmonic_mean = 2 / (1 / (self._weights["reinforce"]*sublosses["reinforce"].losses + 1e-16) + 1 / (self._weights["threshold"]*sublosses["threshold"].losses + + 1e-16))
      # total_losses = total_losses + harmonic_mean

      # Construct a LossBatch with the final total loss and all sublosses
      return LossBatch(losses=total_losses, sublosses=sublosses)