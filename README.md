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

Specific to the "Bone Fracture Detection Dataset:
This repository also contains code on how to merge two classes in this case ( humerus, humerus fracture)
- Class imbalance
- humerus / humerus fracture merge decision
- picture
- data leakage check

## Results
| Class              | Images | Instances | Precision | Recall | mAP50 | mAP50-95 |
|--------------------|--------|-----------|-----------|--------|-------|----------|
| **All**            | 169    | 96        | 0.186     | 0.142  | 0.145 | 0.0622   |
| Elbow positive     | 13     | 17        | 0.000     | 0.000  | 0.033 | 0.008    |
| Fingers positive   | 22     | 27        | 0.234     | 0.074  | 0.062 | 0.018    |
| Forearm fracture   | 13     | 14        | 0.143     | 0.429  | 0.284 | 0.143    |
| Humerus fracture   | 14     | 15        | 0.220     | 0.067  | 0.168 | 0.080    |
| Shoulder fracture  | 15     | 17        | 0.230     | 0.118  | 0.151 | 0.038    |
| Wrist positive     | 6      | 6         | 0.290     | 0.167  | 0.174 | 0.085    |

worth highlighting are the precision (0.000) and recall (0.000) of the elbow positive class.
This result was also recovered by another YOLO model, specifically YOLOv11 training on the same dataset but with the humerus / humerus fracture distinction in tact.

- confusion matrix / PR curve images embedded

## Challenges 

## Limitations & Future Work
  worth highlighting are the precision (0.000) and recall (0.000) of the elbow positive class.
  This result was also recovered by another YOLO model, specifically YOLOv11 training on the same dataset but with the humerus / humerus fracture distinction in tact.

- Ablation ideas(fliplr, mosaic)
- Segmentation as a strecth goal
  


## References

- Jocher, G., Chaurasia, A., & Qiu, J. (2023). Ultralytics YOLOv8 (Version 8.0.0) [Computer software]. https://github.com/ultralytics/ultralytics
- Dataset: Bone Fracture Detection, Roboflow Universe (veda), https://universe.roboflow.com/veda/bone-fracture-detection-daoon
- How to write a Good README : https://www.freecodecamp.org/news/how-to-write-a-good-readme-file/  
- Blankname78 kaggle notebook:  https://www.kaggle.com/code/blankname78/yolov11
