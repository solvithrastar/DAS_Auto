[![DOI](https://zenodo.org/badge/436188798.svg)](https://zenodo.org/badge/latestdoi/436188798)

# DAS_Auto

This is a repository for a method which detects earthquakes in datasets
recorded with fiber-optic seismology (also known as Distributed Acoustic Sensing, DAS).

The method does this with image processing and computer vision techniques,
exploiting the main benefit of the DAS measurement technique, which is its high
spatial sampling.

Check out how it works in the Showcase notebook using Binder:
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/solvithrastar/DAS_Auto/main?labpath=example_notebook.ipynb)

If you want to use the code, there is an `environment.yml` file there,
so you can create a conda environment by running:

`conda env create -f environment.yml`

and that should be enough to be able to use the code.

The repository contains both a collection of image processing functions as well as two example pipelines which can be used to detect earthquakes in DAS measurements.
