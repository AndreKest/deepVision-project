# deepVision-project
Computer Vision project to detect objects with YolovX and Yolov8

Was build for a project at my university.

Task:
Train the two Yolo networks (YolovX and Yolov8) with the Udacity Self Driving Car dataset. Work out the differences between the two architectures and compare the accuracy of the two networks.

------------------------------------------------------------------------------------------
### Folder structure
.<br>
├── data                    &emsp;# Dataset (images and annotation)<br>
├── src                     &emsp;# Main data ipynb and py files<br>
├── doc                     &emsp;# Documentation of the project<br>
├── requirements.txt        &emsp;# Requirements for installation<br>
└── readme.md<br>


------------------------------------------------------------------------------------------
### Version:

Python: 3.11.3<br>
NumPy
Matplotlib
PyTorch
Torchvision


------------------------------------------------------------------------------------------
### Install packages
pip install -r requirements.txt




------------------------------------------------------------------------------------------
### Dataset
Source: https://www.kaggle.com/datasets/sshikamaru/udacity-self-driving-car-dataset

The dataset contains 97,942 labels across 11 classes and 15,000 images. There are 1,720 null examples (images with no labels).

All images are 512x512 (size ~1.2 GB).

Annotations have been hand-checked for accuracy by Roboflow.


| Class                     | Count | Index |
|---------------------------|:-----:|:-----:|
| cat                       | 64399 |  0    |
| pedestrian                | 10806 |  1    |
| trafficLight-Red          |  6870 |  2    |
| trafficLight-Green        |  5465 |  3    |
| truck                     |  3623 |  4    |
| trafficLight              |  2568 |  5    |
| biker                     |  1846 |  6    |
| trafficLight-RedLeft      |  1751 |  7    |
| trafficLight-GreenLeft    |  310  |  8    |
| trafficLight-Yellow       |  272  |  9    |
| trafficLight-YellowLeft   |  14   |  10   |


------------------------------------------------------------------------------------------
### Model
## YoloX
Source:
- https://github.com/Megvii-BaseDetection/YOLOX
- https://arxiv.org/pdf/2107.08430.pdf



------------------------------------------------------------------------------------------
# Yolov8
Source:
- https://github.com/ultralytics/ultralytics
- https://docs.ultralytics.com/models/yolov8/
- https://docs.ultralytics.com



------------------------------------------------------------------------------------------
