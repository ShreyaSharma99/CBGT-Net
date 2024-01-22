# -*- coding: utf-8 -*-
"""
.. module:: accuracy_measure.py
   :platform: Linux, Windows, OSX
   :synopsis: Class for calculating accuracy measures of a provided trajectory

.. moduleauthor:: AARTLab <danahugh@andrew.cmu.edu>


"""

import tensorflow as tf
import numpy as np


class AccuracyMeasure:
   """
   The AccuracyMeasure class is used to calculate accuracy measures of a 
   provided EpisodeBatch.
   """

   def __init__(self):
      """
      """

      pass


   def __call__(self, episode, model=None, append=True):
      """
      Calculate the accuracy measures of the provided batch, and optionally
      append to the episode.

      Arguments
      ---------
      episode : EpisodeBatch
         Episode to calculate accuracy of
      model : CBGT_Net, default=None
         CBGT_Net model, unused and for API compatibility
      append : boolean, default=True
         Indicator of whether results should be added to the provided
         episode_batch

      Returns
      -------
      dict
         Dictionary mapping measure names to calculated values
      """

      # Determine the indices where initial decisions were made, and get the
      # predictions and target values at these points
      num_timesteps, batch_size = episode.decision_masks.shape[:2]

      decision_idx = num_timesteps - tf.reduce_sum(tf.cast(episode.decision_masks, tf.int32), axis=0) - 1
      batch_idx = tf.reshape(tf.range(batch_size), (-1,1))
      idx = tf.concat([decision_idx, batch_idx], 1)

      predictions = tf.gather_nd(episode.decisions, idx)
      targets = tf.gather_nd(episode.targets, idx)

      # Calculate the number of correct predictions
      correct = tf.reduce_sum(tf.cast(predictions==targets, tf.int32))
      accuracy = tf.cast(correct, tf.float32) / batch_size

      # max_time_steps, batch_size, num_classes
      # print("episode.evidence = ", episode.evidence.shape)
      # print("evidence true = ", tf.cast(episode.targets[:, :, 0],tf.int32).shape)
      # print("tf.cast(1 - tf.cast(episode.decision_masks[:, :, 0], tf.int32), bool) =", tf.cast(1 - tf.cast(episode.decision_masks[:, :, 0], tf.int32), bool)[:, 0])
      evidence_pred = tf.cast(tf.math.argmax(episode.evidence, -1), tf.int32) 
      # ev_cor =evidence_pred==tf.cast(episode.targets[:, :, 0]
      
      evidence_correct = tf.cast(tf.math.logical_and(tf.cast(evidence_pred==tf.cast(episode.targets[:, :, 0],tf.int32), bool) , tf.cast(1 - tf.cast(episode.decision_masks[:, :, 0], tf.int32), bool)), tf.int32) 
      # print("decision_idx[:, 0] ", decision_idx[0, 0])
      evidence_acc =  tf.reduce_mean(tf.reduce_sum(evidence_correct,axis=0)/(decision_idx[:, 0]+1))
      # print("decision_idx ", decision_idx[0, 0])
      # tf.print("evidence_correct - ",  episode.decision_masks[:, 0, 0])

      # evidence_acc = evidence_correct / evidence_pred.shape[0]
      # print("evidence_acc: ",evidence_acc.shape)
      # episode.targets[:, :, 0].shape, episode.decision_masks[..., 0].shape, decision_idx.shape)

      # accumulator =  tf.transpose(episode.accumulators, perm=[1, 0, 2]) # max_length, batch_sz, 10
      # target = tf.transpose(tf.squeeze(episode.targets), perm=[1, 0])

      # # print("Accumulator shape = ", accumulator.shape)
      # # print("target shape = ", target.shape, target[:50])

      # a, b, c = accumulator.shape
      # row_indices, col_indices = tf.meshgrid(tf.range(a), tf.range(b), indexing='ij')

      # # print("row_indices, col_indices = ", row_indices.shape, col_indices.shape)
      # # Create indices for gathering
      # indices = tf.concat([tf.reshape(row_indices, [-1])[..., tf.newaxis], tf.reshape(col_indices, [-1])[..., tf.newaxis], tf.reshape(target, [-1])[..., tf.newaxis]], axis=-1)
      # # print("indices shape = ", indices.shape)
      # accumulator_target_val =  tf.gather_nd(accumulator, indices)

      # # print("accumulator_target_val shape = ", accumulator_target_val.shape)
      # accumulator_target_val = tf.reshape(accumulator_target_val, [a, b])

      if append:
         episode = episode.add_attributes({"correct_predictions": correct,
                                           "accuracy": accuracy,
                                           "decision_times": decision_idx,
                                           "evidence_encoder_acc" : evidence_acc})
                                          #  "accumulator_target_val" : accumulator_target_val})

      return episode

      




