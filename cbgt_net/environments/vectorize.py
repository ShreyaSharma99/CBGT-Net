# -*- coding: utf-8 -*-
"""
.. module:: vectorize
   :platform: Linux, Windows, OSX
   :synopsis: Simple environment wrapper for vectorizing oservations

.. moduleauthor:: AARTLab <danahugh@andrew.cmu.edu>
"""

import numpy as np

from .environment import Environment
from ..utils import DependencyInjection as DI


class Vectorize(Environment):
   """
   This class acts as a wrapper for vectorizing environments.
   """

   @classmethod
   def _build(cls, config):
      """
      Builder method for creating a Vectorized set of environments from a
      configuration dictionary.  The provided config file should have the 
      following keys:

      * num_environments : int
          Number of instances of the environment to construct
      * class : string
          Fully qualified class name for the environments to construct
      * properties : dictionary
          Set of properties to use to build the environments
      """

      # Get the class to use to build the environment
      _, EnvironmentClass = DI.get_module_and_class(config["class"])

      return cls(config["num_environments"],
                 EnvironmentClass,
                 config.get("properties", {}))


   def __init__(self, num_environments, EnvironmentClass, parameters, **kwargs):
      """
      Create a set of environments using the `build` method, passing the
      configuration to each independent environment built.

      Arguments
      ---------
      num_environments : int
          Number of environments to construct
      EnvironmentClass : Environment type
          Type of environment to construct
      parameters : dictionary
          Configuration parameters for the environment
      """

      Environment.__init__(self, **kwargs)

      self._num_environments = num_environments

      # Create each environment
      self._environments = [EnvironmentClass.build(parameters) for _ in range(self._num_environments)]


   @property
   def target(self):
      return np.vstack([env.target for env in self._environments])


   def properties(self):
      """
      """

      return [env.properties() for env in self._environments]


   def reset(self):
      """
      """

      for env in self._environments:
         env.reset()


   def observe(self):
      """
      """

      return np.vstack([env.observe() for env in self._environments])