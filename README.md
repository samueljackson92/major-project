MIA: Mammogram Image Analysis
=============================

MIA is a python library for analysing mammographic image data. The library includes both an image analysis API and a command line interface. This project was produced as part of my 4th year coursework for the major project at Aberystwyth University.

## Overview
The official documentation surrounding the major project (final report latex files etc.) reside in the `documents` folder. The code dcoumentation generated from [sphinx](http://sphinx-doc.org) resides in the `doc` folder. The source code and tests for the code base lives in the `src` folder. A full listing of the source coder hierarchy is shown below:

```
`-- src
    |-- mia
    |   |-- __init__.py
    |   |-- analysis.py
    |   |-- command.py
    |   |-- convolve_tools
    |   |   `-- convolve_tools.c
    |   |-- coranking.py
    |   |-- features
    |   |   |-- __init__.py
    |   |   |-- _adjacency_graph.py
    |   |   |-- _nonmaximum_suppression.py
    |   |   |-- _orientated_bins.py
    |   |   |-- blobs.py
    |   |   |-- intensity.py
    |   |   |-- linear_structure.py
    |   |   `-- texture.py
    |   |-- io_tools.py
    |   |-- plotting.py
    |   |-- reduction
    |   |   |-- __init__.py
    |   |   |-- multi_processed_reduction.py
    |   |   |-- reducers.py
    |   |   `-- reduction.py
    |   `-- utils.py
    |-- mia.egg-info
    |-- scripts
    |   |-- make_masks.py
    |   |-- make_thumbs.py
    |   `-- swiss_roll.py
    |-- setup.py
    `-- tests
        |-- __init__.py
        |-- regression_tests
        |   |-- __init__.py
        |   |-- blob_detection_regression_test.py
        |   |-- intensity_regression_test.py
        |   |-- reducers_regression_test.py
        |   |-- texture_features_regression_test.py
        |   `-- utils_regression_test.py
        |-- test_data
        |   |-- mias
        |   |   `-- masks
        |   |-- reference_results
        |   `-- texture_patches
        |-- test_utils.py
        `-- unit_tests
            |-- __init__.py
            |-- adjacency_graph_test.py
            |-- analysis_test.py
            |-- blob_detection_test.py
            |-- command_test.py
            |-- convolve_tools_test.py
            |-- coranking_test.py
            |-- intensity_features_test.py
            |-- io_tools_test.py
            |-- nonmaximum_suppression_test.py
            |-- orientated_bins_test.py
            |-- plotting_test.py
            |-- texture_features_test.py
            `-- utils_test.py
```

## Installation
MIA can be install using the command line utility pip. If your Python install does not already include pip you can install it by following the instructions in this [link](https://pip.pypa.io/en/stable/installing.html).

To install MIA `cd` into the folder called src which contains a file called setup.py. Then run the following command:

```
pip install .
```

## Command Line Interface
The command line interface can be used to run the feature detection algorithms implemented as part of this project over a folder containing an image dataset. This can be performed by using the reduction command which has the following format:

```
mia reduction \[type of feature\] \[name of folder containing images\] \[name of folder containing masks\] \[output file\]
```

For example, to detect blobs from a data you might use the following command:

```
mia reduction blobs ./data ./data/masks blobs_output.csv
```

This command will iterate over all of the images in the data folder and use the corresponding masks for each images in the data/masks folder. The output of blob detection would then be saved to the file output.csv. When detecting intensity and texture features from a pacth defined by an ROI (a blobs/lines detected in a previous run) an additional file must supplied:

```
mia reduction intensity_from_patch ./data ./data/masks blobs_output.csv output.csv
```

## Running the tests
The unit/regression tests for the module can be run by using [nose](http://nose.readthedocs.org/en/latest/). The unit tests are located in `src/tests/unit_tests` and the regression tests are located in `src/tests/egression_tests`. You can run the all of the tests by using the following command:

```
nosetests --cover-package=mia
```

Running thre regression tests can take a long time (typically this was just run on the build server). To only run the unit tests the following command may be used:

```
nosetests src/tests/unit_tests
```
