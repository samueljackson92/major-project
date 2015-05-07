MIA: Mammogram Data Analysis
============================

MIA is a python library for analysing mammographic image data. The library includes both an image analysis API and a command line interface. This project was produced as part of my 4th year coursework for the major project as Aberystwyth University.

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

## 
