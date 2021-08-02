# PlagDef

[![PyPI version](https://badge.fury.io/py/plagdef.svg)](https://badge.fury.io/py/plagdef)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/plagdef)
![GitHub](https://img.shields.io/github/license/devWhyqueue/plagdef)
[![Test](https://github.com/devWhyqueue/plagdef/actions/workflows/cd.yml/badge.svg)](https://github.com/devWhyqueue/plagdef/actions/workflows/test.yml)
[![Coverage](https://sonarcloud.io/api/project_badges/measure?project=devWhyqueue_plagdef&metric=coverage)](https://sonarcloud.io/dashboard?id=devWhyqueue_plagdef)

A tool which makes life hard for students who try to make theirs simple.

# Installation

Get it from PyPI:

```
$ pip install plagdef
````

Or build it yourself:

```
$ git clone git://github.com/devWhyqueue/plagdef
$ poetry build
$ pip install dist/plagdef-{VERSION}.tar.gz
````

# Requirements

## EasyOCR

For acceptable OCR performance on PDF files you need CUDA.\
To install CUDA for your operating system take a look at:\
https://docs.nvidia.com/cuda/cuda-quick-start-guide/index.html

Then install torch and torchvision for your CUDA version:
https://pytorch.org

## Libmagic

**After** (important!) you installed PlagDef, install the libmagic library.\
PlagDef uses it to detect character encodings.\
Further instructions can be found here:\
https://github.com/ahupp/python-magic#installation

# Usage

Run the GUI:

```
$ plagdef-gui
````

Or if you prefer a CLI:

```
$ plagdef -h
````

# Development

Clone the repo and install dependencies:

```
$ git clone git://github.com/devWhyqueue/plagdef
$ poetry install
````

# Publish to PyPI

In your virtual environment build and upload PlagDef:

```
$ poetry publish --build
````
