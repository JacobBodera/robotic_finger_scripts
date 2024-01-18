# Neural Networks

This repository is a collection of all of the python scripts that I created while working on silicon robotics.

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Acknowledgements](#acknowledgements)

## Introduction

This project revolved around creating an open-loop control system for silicon robitics controlled through pneumatic systems. My role was to create the control through embedded systems while capturing all of the 

## Features

There are a few main scripts that help perform various tasks, such as: controlling pneumatic systems through controllers to produce robotic movements, capturing images using an integrated camera system, plotting real-time sensor data, and using computer vision systems to automate tracking deflection curves.\

## Getting Started

### Prerequisites

- Python >= 3.9
- Numpy
- Matplotlib
- Pandas
- Pyfirmata
- OpenCV
- Pickle
- Imutils
- Scipy
- PIL

### Installation

Clone the repository:

`git clone https://github.com/JacobBodera/robotic_finger_scripts.git`

`cd robotic_finger_scripts`

Physical Components:

There are two main systems that work in collaboration with software:

- the camera system
- the robotic control system

These systems will not be described in this project, however, the dot-tracking system can be tested by capturing a series of red dots and placing the file in the `test_images` directory and running `dot_deflection_tracking.py`. This will produce an `output_images` directory with the tracked image.

## Acknowledgements

Alasdair MacLean - Contributor to DIC scripts
