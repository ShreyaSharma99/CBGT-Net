# -*- coding: utf-8 -*-
"""
.. module:: observable.py
   :platform: Linux, Windows, OSX
   :synopsis: Definition of a mixin to provide classes the functionality to
              register observers.

.. moduleauthor:: AARTLab <danahugh@andrew.cmu.edu>

This module defines a mixin class for handling registration and deregistration
of observers.
"""

import logging


class Observable:
    """
    The Observable mixin enables a class to include attributes and methods to
    register and deregister observers of the subject class.  

    When initialized, the Observable class will introspect the inheriting class
    for a logger; it will search for attributes named `_logger` and `logger`
    (in that order), and will store the corresponding logger _if_ it is an
    instance of `logging.Logger`.  Alternatively, a Logger instance can be 
    passed to the Observable using the `logger` keyword.

    Attributes
    ----------
    _observers : set
        Set of observers registered to receive updates from the subject.

    USage
    -----
    A subject class should inherit the Observable mixin, and call its __init__
    method when initializing, for instance::

        class Subject(Observable):

            def __init__(self, ...):

                super(Observable, self).__init__()

    Note that, since the `__init__` method introspects the object for a logger,
    calling the `__init__` method of the Observable should occur only after the
    logger attribute (`self.logger` or `self._logger`) has been created.

    The Observer mixin provides two public methods, `register_observer` and 
    `deregister_observer`.  Observers should call the `register_observer` 
    method to receive updates from the subject, and `deregister_observer` to no
    longer receive updates.  Both methods accept an object to receive updates
    from the subject.  For instance, during the __init__ method of an observer,
    the observer can call the registration method::

        class Observer:

            def __init__(self, subject, ...):

                subject.register_observer(self)

            def subject_callback(self, ...):

                # Called by the subject

    The Observer mixin also provides a private attribute, `_observers`, which
    the subject can use to notify observers.  Note that there is no assumption
    or constraint on the methods being called, but it is assumed that the
    subject implements the method.

        class Subject(Observable):

            def something(self):

                ...

                # Inform the observers of some update
                for observer in self._observers:
                    observer.subject_callback(...)
    """

    def __init__(self, **kwargs):
        """
        Keyword Arguments
        -----------------
        logger : logging.Logger, default=None
            Instance of a Logger to report warnings to
        """

        # Store the passed logger, if provided.  Otherwise, introspect the
        # parent class for an attribute named `logger` or `_logger`.
        self.__logger = kwargs.get("logger", None)

        if self.__logger is None:

            # Check to see if `_logger` exists, and is an instance of
            # logging.Logger
            if hasattr(self, "_logger") and type(self._logger) is logging.Logger:
                self.__logger = self._logger
            elif hasattr(self, "logger") and type(self.logger) is logging.Logger:
                self.__logger = self.logger

        # Create the _observers set
        self._observers = set()


    def register_observer(self, observer, priority=1):
        """
        Register an observer to receive updates from this subject.

        Arguments
        ---------
        observer : object
            Object to receive updates from the subject
        priority : int
            Priority of the observer.  Observers with lower priority will be
            iterated prior to those with higher priority.
        """

        # Check to see if the observer is already in the set of observers.
        # There is no need to try to add it, if so, and would be worth issuing
        # a warning.
        if observer in self._observers:
            if self.__logger is not None:
                self.__logger.warning("%s:  Attempting to register an observer that has already been registered: %s", self, observer)
            return

        self._observers.add(observer)


    def deregister_observer(self, observer):
        """
        Remove the observer from the set of objects to receive updates from 
        this subject.

        Arguments
        ---------
        observer : object
            Object to no loger receive updates from the subject
        """

        # Check to see if the observer is in the set of observers.  If it isn't
        # a warning should be issued.
        if not observer is self._observers:
            if self.__logger is not None:
                self.__logger.warning("%s:  Attempting to deregister an observer that is not currently registered: %s", self, observer)
            return

        self._observers.remove(observer)

