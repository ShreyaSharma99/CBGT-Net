# -*- coding: utf-8 -*-
"""
.. module:: loggable.py
   :platform: Linux, Windows, OSX
   :synopsis: Definition of a mixin to simplify and create common attributes
              for logging capabilities.

.. moduleauthor:: AARTLab <danahugh@andrew.cmu.edu>

This module defines a mixin class for providing a common implementation on 
per-class logging.
"""

import logging

class Loggable:
    """
    The Loggable mixin provides a simple superclass to provide per-class 
    logging capabilities using a common interface.

    Attributes
    ----------
    _logger : logging.Logger
        Instance of a logger

    USage
    -----

    """

    def __init__(self, **kwargs):
        """
        Keyword Arguments
        -----------------
        log_level : int, default=logging.WARNING
            Log level to use for the inheriting class
        """

        # Create a logger for the environment
        self._logger = logging.getLogger(str(self))
        self._logger.setLevel(kwargs.get("log_level", logging.WARNING))

    @property
    def logger(self):
        return self._logger
    