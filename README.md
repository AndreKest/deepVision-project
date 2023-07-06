# deepVision-project
Computer Vision project to detect objects with YOLOX and YOLOv8

Was build for a project at my university.

Task:
Train the two Yolo networks (YolovX and Yolov8) with the Udacity Self Driving Car dataset. Work out the differences between the two architectures and compare the accuracy of the two networks.

------------------------------------------------------------------------------------------
### Folder structure
deepvision-project/  
├── data                    &emsp;# Dataset (images and annotation)  
&emsp; ├── dataFiltered     &emsp;# Filtered data (changed classes)  
&emsp; ├── dataRaw          &emsp;# Raw data  
&emsp; ├── dataYolov8       &emsp;# Data in format for Yolov8  
&emsp; └── dataYoloX        &emsp;# Data in format for YoloX  
├── src                     &emsp;# Main data ipynb and py files  
&emsp; ├── dataset          &emsp;# Files for dataset (convertion, preprocessing)  
&emsp; ├── yolov8Model      &emsp;# Yolov8 stuff  
&emsp; └── yoloxModel       &emsp;# YoloX stuff  
├── doc                     &emsp;# Documentation of the project  
├── utils                   &emsp;# YoloX GitHub Repository data  
├── .gitignore              &emsp;# Gitignore file for Repository  
├── requirements.txt        &emsp;# Requirements for installation  
└── readme.md<br>


------------------------------------------------------------------------------------------
### Version:

Python: 3.8.16  
NumPy: 1.21.5  
Matplotlib: 3.5.1  
PyTorch: 1.10.2+cu111  
Torchvision: 0.11.3+cu111  
Torchaudio: 0.10.2+cu111  
Ultralytics: 8.0.124  
Tensorboard: 2.11.2  

--> More information about packages and versions are in **requirements.txt**  
Install: pip install -r requirements.txt


------------------------------------------------------------------------------------------
### Dataset
Source: 
- https://www.kaggle.com/datasets/sshikamaru/udacity-self-driving-car-dataset
- https://public.roboflow.com/object-detection/self-driving-car

The dataset contains 97,942 labels across 11 classes and 15,000 images. There are 1,720 null examples (images with no labels).

All images are 512x512 (size ~1.2 GB).

Annotations have been hand-checked for accuracy by Roboflow.


| Class                     | Count | Index |
|---------------------------|:-----:|:-----:|
| car                       | 64399 |  0    |
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
<br>

Changed to
| Class                     | Count | Index |
|---------------------------|:-----:|:-----:|
| car                       | 64399 |  0    |
| pedestrian                | 10806 |  1    |
| trafficLight              | 17250 |  2    |
| truck                     |  3623 |  3    |
| biker                     |  1846 |  4    |


------------------------------------------------------------------------------------------
### Model
## YOLOX
Source:
- https://github.com/Megvii-BaseDetection/YOLOX
- https://arxiv.org/pdf/2107.08430.pdf
- Training: https://towardsdatascience.com/object-detection-neural-network-building-a-yolox-model-on-a-custom-dataset-77d29d85ae7f


### COCO Dataset Format
{
    “image”: [{
        “id”: int,
        “width”: int,
        “height”: int,
        “file_name: str,
        “license”: int,
        “flickr_url”: str,
        “coco_url”: str,
        “date_captured”: datetime
    },
    ... ]
    “annotations”: [{
        “id”: int,
        “image_id: int”,
        “category_id”: int
        “area”: float,
        “bbox”: [x,y,width,height],
        “iscrowd”: 0 or 1
    },
    ... ]
}


Train:
!python3 train.py -f yolox_m.py -d 1 -b 8 --fp16 -o -c weights/yolox_m.pth


Evaluation:
!python3 eval.py -n yolox-m -c YOLOX_outputs/yolox_m_100_epoch_no_Oversample/weights/latest_ckpt.pth -b 8 -d 1 --conf 0.001 -f yolox_m.py





------------------------------------------------------------------------------------------
## YOLOv8
Source:
- https://github.com/ultralytics/ultralytics
- https://docs.ultralytics.com/models/yolov8/
- https://docs.ultralytics.com/modes/




### YOLO Dataset Format


------------------------------------------------------------------------------------------
