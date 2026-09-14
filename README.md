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

## Dataset

Specific to the "Bone Fracture Detection Dataset"[1]:
This repository also contains code on how to merge two classes in this case ( humerus, humerus fracture)
- Class imbalance
- humerus / humerus fracture merge decision
- picture
- data leakage check

## Results
- mAP50 / mAP50-95, per class table
- confusion matrix / PR curve images embedded

## Challenges 

## Limitations & Future Work
- Ablation ideas(fliplr, mosaic)
- Segmentation as a strecth goal
  


## References

- Jocher, G., Chaurasia, A., & Qiu, J. (2023). Ultralytics YOLOv8 (Version 8.0.0) [Computer software]. https://github.com/ultralytics/ultralytics
- Dataset: Bone Fracture Detection, Roboflow Universe (veda), https://universe.roboflow.com/veda/bone-fracture-detection-daoon
- How to write a Good README : https://www.freecodecamp.org/news/how-to-write-a-good-readme-file/  
