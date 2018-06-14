# Wearable sensors for personalized monitoring of Parkinson’s disease: is more data always better?
Michael J. Fox Foundation for Parkinson’s Research Clinician Input Study (CIS-PD) Wireless Adhesive Sensor Sub-Study

## Contents

- [Overview](#overview)
- [Repo Contents](#repo-contents)
- [System Requirements](#system-requirements)
- [Installation Guide](#installation-guide)
- [Demo](#demo)
- [Results](#results)
- [License](./LICENSE)
- [Issues](https://github.com/Luke3D/CIS-PD_npg/issues)
- [Citation](#citation)

# Overview

Individuals living with Parkinson’s disease (PD) often experience variable and unpredictable motor symptoms such as tremor, rigidity, and postural instability, which can affect activities in daily life. Machine learning algorithms that use data streams captured from soft wearable sensors have the potential to automatically detect PD symptoms and inform clinicians about the progression of disease with minimal burden and discomfort to the patient. However, these algorithms must be trained with annotated data from clinical experts who can recognize symptoms, and collecting such data is costly. Understanding how many sensors and how much labeled data are required is key to successfully deploying these models outside of the clinic. Here we measured movement data using 6 flexible wearable sensors in 20 individuals with PD over the course of multiple clinical assessments. The patients were monitored on one day and assessed again two weeks later. Participants performed a range of daily activities, such as walking or typing, while a clinician rated the severity of their symptoms (bradykinesia and tremor). We then trained random forest classifiers to detect whether a segment of movement showed signs of bradykinesia or tremor based on data from tasks performed by a single individual (personal models) or multiple individuals (population models). Our results show that a single wearable sensor on the back of the hand is sufficient for symptom detection, and that using personal data improves accuracy over population models. Training on repeated assessments on one day does not improve detection on subsequent days. Our results suggest that PD patients could significantly benefit from continuous detection of motor symptoms in small, personalized datasets collected longitudinally.

# Repo Contents

- [code](./code): `Python 3.5` package code as `Jupyter Notebook` files.
- [docs](./docs): Accompanying documentation on code and usage structure.
- [tests](./tests): Simulated test data for running the `Jupyter Notebook` files. Please see the [Demo](#demo) section for information about availability of the dataset used in this study.

# System Requirements

## Hardware Requirements

The code in the `CIS-PD_npg` repo can be run on a standard computer with enough RAM to support processing of the complete dataset as defined by the user. For absolute minimum performance, a computer with 4 GB of RAM. For optimal performance, the following specifications are recommended:

RAM: 16+ GB

CPU: 4+ Cores, 2.8+ GHz/core

Additionally, [CNNModels.ipynb](./code/CNNModels.ipynb) uses the `keras` package in training the convolutional neural networks. Although `keras` can be run with a CPU, for optimal performance we recommend running [CNNModels.ipynb](./code/CNNModels.ipynb) on a computer with a dedicated GPU. The code has been tested on the following GPUs:

NVIDIA GeForce GTX TITAN X

NVIDIA GeForce GTX 1050

The runtimes were generated using a computer with 32 GB RAM, 4 cores @ 2.8 GHz (i7-7700HQ), an NVIDIA GeForce GTX 1050 GPU, and an internet speed of 20 Mbps.

## Software Requirements

### OS Requirements

The package development was tested on *Linux* and *Windows* operating systems, and has been tested on the following versions:

Linux:

Mac OSX: Sierra 10.12.6

Windows: 7 and 10

All of the packages used to run these `Jupyter Notebook` files were installed using either `pip` or `Anaconda` and should be compatible across all platforms. We ran the code on `Anaconda` using `Python 3.6.2` or higher.

Four main packages used in the code were tested and developed using the versions listed below:

- `pandas`: 0.20.3+
- `keras`: 2.0.2
- `tensorflow`: 1.8.0
- `theanos`: 0.9.0


# Installation Guide
Users should first install the latest version of `Anaconda` using the following [link](https://www.anaconda.com/download/) and downloading the installer or appropriate version for `Python 3.6 version`. This will automatically install a majority of the default `Python` packages needed, along with `Jupyter Notebook`.

## Package Installation
There are a number of additional packages and software required to fully run all of the code in the `CIS-PD_npg` repository.

### Features Calculation
For features calculation on the raw data, the `nolds` package is required and can be installed using the following command through pip:
```
pip install nolds
```
and will complete in no longer than 1 minute on a recommended machine.

Additionally, the `nolds` .whl file has been provided and is located [here](./docs/nolds-0.4.1-py2.py3-none-any.whl) and can be installed using the following command:
```
sudo pip install --upgrade nolds-0.4.1-py2.py3-none-any.whl
```

### CNN Calculation
#### TensorFlow
In order to run [CNNModels.ipynb](./code/CNNModels.ipynb), an installation of `keras` and a backend of either `tensorflow` or `theano` is required.

`tensorflow` can be installed using the following command on a dedicated GPU machine through pip
```
pip3 install --upgrade tensorflow-gpu
```
or for a CPU-only machine:
```
pip3 install --upgrade tensorflow
```
Additional installation instructions for `tensorflow` can be found [here](https://www.tensorflow.org/install/).

#### Theano
`theano` can be installed using conda with the following command:
```
conda install theano pygpu
```
Additional installation instructions for `theano` can be found [here](http://deeplearning.net/software/theano/install.html).

#### Keras
Finally, once a `tensorflow` or `theano` backend has been chosen and installed, the last package required to run all of the code is the installation of `keras`. This can be done through `pip` using the following command:
```
pip install keras
```
Alternatively, one can also use `conda` to install `keras`.

Additional installation instructions for `keras` can be found [here](https://keras.io/#installation).

#### GPU Drivers
Of note for the NVIDIA graphics cards that the code was tested on is the requirement of installing the appropriate CUDA drivers, if the graphics card is CUDA-capable. For the NVIDIA GeForce GTX 1050 used for a majority of the code, `CUDA 9.2` was installed and tested when running the CNN operation code.

More information can be found on the [CUDA Toolkit Documentation](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html#installing-cuda-development-tools).

## Installation Issues
If you encounter any issues with installing any of these required packages, or still encounter issues running the code after successfully installing them, please raise an [Issue](https://github.com/Luke3D/CIS-PD_npg/issues).

# Demo
The dataset used to support the findings of this publication are available from the Michael J. Fox Foundation but restrictions apply to the availability of these data, which were used under license for this study. The Michael J. Fox Foundation plans to release the dataset used in this publication alongside a significant, additional portion of related PD data from a separate smartwatch as part of a community analysis in the larger CIS-PD study timeline. Data are however available from the authors upon reasonable request and with permission from the Michael J. Fox Foundation.

# Results
To do

# Citation
To do
