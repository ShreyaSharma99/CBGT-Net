# -*- coding: utf-8 -*-
"""
.. module:: experiment_profiler.py
   :platform: Linux, Windows, OSX
   :synopsis: Observer used to generate a profile of experiment

.. moduleauthor:: AARTLab <danahugh@andrew.cmu.edu>

A simple observer to generate a profile of the experiment

Requires
--------
yappi
"""

import yappi


class ExperimentProfiler:
	"""
	An ExperimentProfiler is a class that starts a profiler at the start of an
	experiment, and dumps the timing profile at the end of the experiment.
	"""

	def __init__(self, experiment, **kwargs):
		"""
		Arguments
		---------
		experiment : Experiment
			Experiment being profiled
		"""

		self._experiment = experiment


	def on_experiment_start(self):
		"""
		Callback when the experiment is started
		"""

		yappi.set_clock_type("cpu")
		yappi.start()


	def on_training_step(self, episode):
		"""
		Callback after a training step has been performed.

		Arguments
		---------
		episode : dict
		    Dictionary containing episode details
		"""

		pass


	def on_evaluation(self, episode):
		"""
		Callback after an evaluation step has been performed.  This method does
		nothing, and is simply for conforming to observer API.

		Arguments
		---------
		episode : dict
			Dictionary containing episode details
		"""

		pass


	def on_experiment_complete(self, exception=None):
		"""
		Callback when the experiment is completed.
		"""

		yappi.get_func_stats().print_all()
		yappi.set_thread_stats().print_all()



