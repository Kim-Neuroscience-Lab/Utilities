.. ROI Area Analysis documentation master file, created by
   sphinx-quickstart on Wed Mar 12 14:39:26 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Kim Lab Utilities Documentation
=====================================

This documentation covers the utilities and tools used in the Kim Neuroscience Lab for data analysis and processing.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   modules/analysis
   modules/core
   modules/utils
   modules/data
   modules/io
   usage

Overview
--------

The Utilities repository contains a collection of shared computational tools and utilities used across various projects in the Kim Neuroscience Lab. This module provides reusable components, analysis tools, and helper functions that support the lab's research workflows.

Package Structure
---------------

The package is organized into several main components:

analysis/
    Tools for data analysis, including:
    
    * ROI area analysis
    * Other analysis utilities

core/
    Core functionality and models:
    
    * Base classes and interfaces
    * Data models (Animal, Region, Segment)
    * Common exceptions and error handling
    * Core services

data/
    Data handling and management utilities

io/
    Input/output operations and file handling

utils/
    General utility functions:
    
    * Hardware detection and optimization
    * Logging configuration
    * Performance monitoring
    * Constants and configuration

Available Tools
-------------

ROI Area Analysis
~~~~~~~~~~~~~~~~
Tool for computing areas of regions of interest (ROIs) from pickle files, with support for:

* Multiple computation methods (fast, GPU, sparse)
* GPU acceleration (CUDA and MPS)
* Batch processing
* Memory optimization

More tools will be added as they are developed and documented.

Quick Links
----------

* :doc:`installation`
* :doc:`usage`
* :ref:`API Documentation <modindex>`
* :ref:`search`

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

