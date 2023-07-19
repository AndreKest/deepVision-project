# deepVision-project
Computer Vision project to detect objects with YOLOX and YOLOv8

Was build for a project at my university.

Task:
Train the two Yolo networks (YolovX and Yolov8) with the Udacity Self Driving Car dataset. Work out the differences between the two architectures and compare the accuracy of the two networks.


------------------------------------------------------------------------------------------
### Weights
To download the pretrained and the trained weights you need to make a few steps!
- git clone <repo>
- git lfs pull 

You need to install git lfs to downlaod the files!


------------------------------------------------------------------------------------------
### Folder structure
#### Overview
deepvision-project/  
├── data                    &emsp;# Dataset (images and annotation)  
&emsp; ├── dataFiltered     &emsp;# Filtered data (merged classes, train/val/test split)  
&emsp; ├── dataRaw          &emsp;# Raw data [1]  
&emsp; ├── dataYolov8       &emsp;# Data in format for Yolov8 (*datasetConvertion.ipynb*)  
&emsp; └── dataYoloX        &emsp;# Data in format for YoloX  (*datasetConvertion.ipynb*)  
├── src                     &emsp;# Main data ipynb and py files  
&emsp; ├── dataset          &emsp;# Files for dataset (convertion, preprocessing)  
&emsp; ├── yolov8Model      &emsp;# Yolov8 stuff  
&emsp; └── yoloxModel       &emsp;# YoloX stuff  
├── doc                     &emsp;# Documentation of the project (LaTex)  
├── utils                   &emsp;# YoloX GitHub Repository data  
├── .gitignore              &emsp;# Gitignore file for Repository  
├── requirements.txt        &emsp;# Requirements for installation  
└── readme.md<br>


#### Files: Dataset
* datasetConvertion.ipynb: Script for converting the data into the formats required by the networks.
* datasetPreprocessing.ipynb: Script for dataset preprocessing.

#### Files: YOLOX
* datasets: This directory contains the dataset in YOLOX format (copy of ./dataset/dataYoloX).
* testImages: Four test files from the test dataset.
* weights: Initialization weights from COCO training.
* yolox: The yolox network from the YOLOX GitHub repository [2].
* YOLOX_ouputs: Output after training (weights, logs, visualizations, etc.).
* yoloxEvalOutput: Output in JSON format for evaluation.
* demo.py: Script to run the network on input images (from the YOLOX GitHub repository [2]).
* eval.py: Script to evaluate the network (from the YOLOX GitHub repository [2]).
* train.py: Script to train the network (from the YOLOX GitHub repository [2]).
* trainYoloX.ipynb: Collection of commands for training/evaluating/running the network.
* yolox_m.py: Network configuration.
* yoloxMetric.ipynb: Calculate evaluation metrics.

#### Files: YOLOv8
* datasets: This directory contains the dataset in YOLOv8 format (copy of ./dataset/dataYolov8).
* runs: Output after training (weights, etc.).
* testImages: Four test files from the test dataset.
* weights: Initialization weights from COCO training.
* yolov8Metrics: Files for network evaluation.
* config.yaml: Network configuration.
* trainYolov8.ipynb: Network training.
* yolov8Metric.ipynb: Calculate evaluation metrics.


#### Source
[1]: Der unbearbeitete Datensatz: https://www.kaggle.com/datasets/sshikamaru/udacity-self-driving-car-dataset  
[2]: GitHub Repository von YOLOX: https://github.com/Megvii-BaseDetection/YOLOX

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

Maybe you don't need the PyTorch and CUDA stuff in the requirements -> check this first!


------------------------------------------------------------------------------------------
### Dataset
Source: 
- https://www.kaggle.com/datasets/sshikamaru/udacity-self-driving-car-dataset

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
- GitHub Repository: https://github.com/Megvii-BaseDetection/YOLOX
- Paper: https://arxiv.org/pdf/2107.08430.pdf
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

**To train the model go to src/yoloxModel/trainYoloX.ipynb**

Train:
!python3 train.py -f yolox_m.py -d 1 -b 8 --fp16 -o -c weights/yolox_m.pth

Evaluation:
!python3 eval.py -n yolox-m -c YOLOX_outputs/yolox_m_100_epoch_no_Oversample/weights/latest_ckpt.pth -b 8 -d 1 --conf 0.001 -f yolox_m.py

Prediction:
!python3 demo.py image -f yolox_m.py -c YOLOX_outputs/yolox_m_200_epoch/weights/best_ckpt.pth --path testImages/testImage_1.jpg --conf 0.25 --nms 0.45 --tsize=512 --save_result --device gpu





------------------------------------------------------------------------------------------
## YOLOv8
Source:
- https://github.com/ultralytics/ultralytics
- https://docs.ultralytics.com/models/yolov8/
- https://docs.ultralytics.com/modes/

### YOLO Dataset Format
- Folder for test, train and val  
- In every folder is a folder with images and labels
- images contains images jpg-files
- labels contains labels txt-files



------------------------------------------------------------------------------------------
