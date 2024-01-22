# -*- coding: utf-8 -*-
"""
.. module:: config_loader
   :platform: Linux, Windows, OSX
   :synopsis: Functionality to load configurations from possibly multiple
              files.

.. moduleauthor:: Dana Hughes <danahugh@andrew.cmu.edu>

This module defines a simple class for loading agent configurations from one
or more files.  Configuration files are expected to be written in JSON, and
may be contained in multiple files.  
"""

import json
import os
from collections import defaultdict


class CircularDependencyException(Exception):
    """
    Exception when a circular dependency is detected.

    Attributes
    ----------
    node : string
        Name of the node that has a circular dependency
    message : string
        Exception message
    """

    def __init__(self, node):
        """
        Arguments
        ---------
        node : string
            Name of the node that was identified as having a circular
            dependency
        """

        self.node = node
        self.message = "Circular Dependency detected at node %s" % self.node

        super().__init__(self.message)



    def __str__(self):

        return "Circular Dependency @ %s" % self.node





class ConfigLoader:
    """
    Defines a class to load a single configuration dictionary across multiple
    files.  
    """

    def __init__(self, working_directory="."):
        """
        Arguments
        ---------
        working_directory : string, default='.'
            Base directory for loading relative paths from
        """

        self._working_directory = working_directory

        # Keep a mapping of URI to loaded JSON, as well as a map from URIs to
        # dependendies (as child URIs)
        self._configs = dict()
        self._dependencies = defaultdict(set)


    def _is_uri(self, value):
        """
        Check to see if the provided value is a URI.  URIs will be a string 
        that (at the moment) starts with "file:///"

        value : object
            Possible URI
        """

        # If the value isn't a string, then it's not a URI
        if type(value) is not str:
            return False

        # If the value starts with "file:///", then it's a URI.  If we decide
        # to get fancy in the future, we can check other schemes
        if value.startswith("file:///"):
            return True

        return False


    def _load(self, json_path):
        """
        Load the contents of a json file, maintaining the file URIs.

        Arguments
        ---------
        json_path : string
            Path to the JSON file
        """

        # Check if the path is in URI format, and remove the 'file:///' if so
        if self._is_uri(json_path):
            json_path = json_path[8:]

        json_path = os.path.join(self._working_directory, json_path)

        # Check to see if the json file exists and is a file, and raise an 
        # error if it doesn't
        if not os.path.isfile(json_path):
            raise RuntimeError("Configuration file does not exist: %s" % json_path)

        with open(json_path) as json_file:
            try:
                json_data = json.load(json_file)
            except Exception as e:
                print("Error parsing %s:", json_path)
                print(str(e))
                json_data = None


        return json_data


    def _traverse(self, item, config_uri='.'):
        """
        Traverse through a configuration item, and recursively load any 
        configuration referenced by a URI.  Only dictionaries, lists, or URI
        strings are considered traversable.

        Arguments
        ---------
        item : object
            Item to (potentially) traverse
        config_uri : string
            URI of the config file that contains the item
        """

        # Check if the item is a URI
        if self._is_uri(item):

            # If the URI isn't loaded yet, then load it and traverse
            if not item in self._configs:
                self._configs[item] = self._load(item)
                self._traverse(self._configs[item], item)

            # Indicate the dependency
            self._dependencies[config_uri].add(item)

        # Check if the item is a dictionary, and traverse each value in the
        # dictionary if so
        elif isinstance(item, dict):
            for dict_item in item.values():
                self._traverse(dict_item, config_uri)

        # Check if the item is a list or tuple, and traverse each item in the
        # list if so
        elif isinstance(item, list) or isinstance(item, tuple):
            for list_item in item:
                self._traverse(list_item, config_uri)

        # Otherwise, do nothing
        else:
            return


    def _replace_with_content(self, config_item, URI):
        """
        Replace the content of references in the given URI with the content of
        those references

        Arguments
        ---------
        config : dictionary, list, or value
            Content of a configuration in whom URI references should be
            replaced with the content of the URI
        URI : string
            URI to dereference
        """

        # Is the config_item the URI?
        if config_item == URI:
            return self._configs[URI]

        # Is the config_item a list or tuple?
        if isinstance(config_item, list):
            return [self._replace_with_content(x, URI) for x in config_item]

        if isinstance(config_item, tuple):
            return tuple([self._replace_with_content(x, URI) for x in config_item])

        # Is the config_item a dictionary?
        if isinstance(config_item, dict):
            return { key: self._replace_with_content(value, URI) for key,value in config_item.items() }

        # Normal, everyday item
        return config_item


    def _topological_sort(self, sorted_URIs = [], URI = None, URI_marks = None):
        """
        Create a topological sort of the config URIs to determine the order of 
        replacing URIs with config content.

        Arguments
        ---------
        sorted_URIs : list
            Topologically sorted list of the dependency graph
        URI : string
            URI currently being sordted
        URI_marks : dictionary
            Dictionary of URI markings indicating if the URI is unmarked,
            temporary, or permanent

        Returns
        -------
        list
            List of topologically sorted URIs, with leaf nodes at the head of
            the list

        Raises
        ------
        RuntimeError if a circular dependency is detected

        Reference
        ---------
        https://en.wikipedia.org/wiki/Topological_sorting
        """

        # Set all URIs to "UNMARKED" if not provided
        if URI_marks is None:
            URI_marks = { URI: "UNMARKED" for URI in self._configs.keys() }

        # Select a URI if one is not given
        if URI is None:
            URI = list(self._configs.keys())[0]

        # If the URI is marked as PERMANENT, it has already been sorted, so no
        # additional work is needed.  If it is marked as TEMPORARY, then it has
        # been visited during a current sort operation, and so indicates a 
        # circular dependency
        if URI_marks[URI] == "PERMANENT":
            return sorted_URIs
        if URI_marks[URI] == "TEMPORARY":
            raise CircularDependencyException(URI)

        # Mark the URI as "TEMPORARY", to indicate that it is part of a 
        # dependency sort that hasn't been resolved yet.
        URI_marks[URI] = "TEMPORARY"

        # Sort the children URIs of the current URI
        for child_URI in self._dependencies[URI]:
            sorted_URIs = self._topological_sort(sorted_URIs, child_URI, URI_marks)

        # Indicate that this URI has been sorted, and put into the sorted list
        URI_marks[URI] = "PERMANENT"
        sorted_URIs.append(URI)

        return sorted_URIs


    def load(self, config_path):
        """
        Load the configuration, using the file in `config_path` as a base.

        Arguments
        ---------
        config_path : string
            Path to the base configuration
        """

        # Load the base configuration
        self._configs['.'] = self._load(config_path)

        # Traverse the configuration tree and load any child config files
        self._traverse(self._configs['.'])

        # Create a topological sort of the URIs, and replace URI references with
        # the contents of the loaded configs
        try:
            sorted_URIs = self._topological_sort()
        except CircularDependencyException as e:
            print(str(e))
            return None


        # Now simply need to iterate over _all_ the configs, replacing URI
        # references with content in the order of the sorted URI list
        for URI in sorted_URIs:
            self._configs = { key: self._replace_with_content(value, URI) for key, value in self._configs.items() }

        return self._configs['.']

