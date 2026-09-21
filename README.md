# Bonefracture Detection with YOLOv8 (using Google Colab)

<img width="1244" height="732" alt="BFD-SamplePicture" src="https://github.com/user-attachments/assets/425415e1-0a80-4679-9e79-c0114d5804ef" />


## Motivation
This repository contains the code and structure to enable everyone ( with a Google account ) to train a Yolov8 models on publicly available datasets, 
giving them a first look into the world of Computer Vision.

The notebooks (.ipynb) provided show 
- how to download the dataset from kaggle
- train their yolov8 model
- evaluate the results
- saving the best model directly to their google drive
- while pushing and pulling to their respective github repository

## Overview
<img width="2205" height="1241" alt="OverviewGraphic-1" src="https://github.com/user-attachments/assets/f3c78b03-633c-4775-bdd9-35ba07edbdf8" />


## Dataset
Before training the original dataset from kaggle was examined with different python functions for integrity. Said functions can be found in [src/bonefracture/data_processing.py](https://github.com/baotle/BoneFractureDetection_Computervision_Project/blob/main/src/bonefracture/data_processing.py)):

- **check_image_integrity** ( makes sure all files in the dataset can be opened)
- **check_label_integrity** ( makes sure all files in the dataset adhere to label conventions for training)
- **check_split_leakage** ( makes sure there are no near duplicates across splits )
- **count_class_distribution** ( sanity check for class distribution before and after augmenting )

Specific to the "Bone Fracture Detection Dataset:

Upon further investigation the original "humerus" and "humerus fracture" class were both merged into one "humerus fracture" class
- because they both contain fractured humeri
- to standardize naming conventions accross the classes

To reduce computation time on a simpler model, the 'coords' of the outlines of the fractures were reduced to simple 
rectangular bounding boxes via the 'convert_split_tobbox' function

Before training the images were further augmented to simulate slightly misaligned, shifted, blurry X-Ray pictures. 
With in-built ultralytics functionality the following parameters were applied


| Parameter | value | Parameter | value |
|  --------- | ------ | ---------- | ------- |
| weight_decay | 0.001  |hsv_v | 0.2 |
| hsv_h | 0|hsv_s|0.0|  scale | 0.2 |
| degrees|5 |translate|0.1| 
| shear|0.0| flipud|0.0| 
|fliplr|0.5| mosaic|0.0|
|scale|0.2|

## Results

The following results were achieved by training a YOLOv8 model for 30 epochs and taking its best performing model.


| Class              | Images | Instances | Precision | Recall | mAP50 | mAP50-95 |
|--------------------|--------|-----------|-----------|--------|-------|----------|
| **All**            | 169    | 96        | 0.186     | 0.142  | 0.145 | 0.0622   |
| Elbow positive     | 13     | 17        | **0.000**     | **0.000**  | 0.033 | 0.008    |
| Fingers positive   | 22     | 27        | 0.234     | 0.074  | 0.062 | 0.018    |
| Forearm fracture   | 13     | 14        | 0.143     | 0.429  | 0.284 | 0.143    |
| Humerus fracture   | 14     | 15        | 0.220     | 0.067  | 0.168 | 0.080    |
| Shoulder fracture  | 15     | 17        | 0.230     | 0.118  | 0.151 | 0.038    |
| Wrist positive     | 6      | 6         | 0.290     | 0.167  | 0.174 | 0.085    |

Worth highlighting are the precision (0.000) and recall (0.000) of the elbow positive class.
This result was also recovered by another YOLO model, specifically YOLOv11 training on the same dataset but with the humerus / humerus fracture distinction in tact.
This indicates a systemic problem with the dataset that cannot be remedied by adjusting the model parameters or the training regiment.

## Challenges 




## References

- Jocher, G., Chaurasia, A., & Qiu, J. (2023). Ultralytics YOLOv8 (Version 8.0.0) [Computer software]. https://github.com/ultralytics/ultralytics
- Dataset: Bone Fracture Detection, Roboflow Universe (veda), https://universe.roboflow.com/veda/bone-fracture-detection-daoon
- How to write a Good README : https://www.freecodecamp.org/news/how-to-write-a-good-readme-file/  
- Blankname78 kaggle notebook:  https://www.kaggle.com/code/blankname78/yolov11
- [Kaggle](https://www.kaggle.com/static/images/logos/kaggle-logo-transparent-300.png)  
*Kaggle logo used for identification purposes; trademark of Kaggle Inc. (a Google company).*
