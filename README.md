# Bonefracture Detection with YOLOv8 (using Google Colab)

<img width="1244" height="732" alt="BFD-SamplePicture" src="https://github.com/user-attachments/assets/425415e1-0a80-4679-9e79-c0114d5804ef" />

***Figure 1. Sample Results of YOLOv8 trained on the bonefracture dataset***

**Tech Stack:** Python · PyTorch · Ultralytics YOLOv8 · Google Colab · Kaggle · Git

## Getting Started
1. Open `notebooks/nb00-setup_and_exploration.ipynb` in Google Colab
2. Mount your Drive and set up a GitHub personal access token (see notebook for details)
3. Run `notebooks/nb01-data-exploration.ipynb` to download and validate the dataset
4. Run `notebooks/nb02_training.ipynb` to train and evaluate your own model

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

***Figure 1. Overview of the dataset processing pipeline***

## Dataset
Before training the original dataset from kaggle was examined with different python functions for integrity. Said functions can be found in [src/bonefracture/data_processing.py](https://github.com/baotle/BoneFractureDetection_Computervision_Project/blob/main/src/bonefracture/data_processing.py)):

- **check_image_integrity** ( makes sure all files in the dataset can be opened)
- **check_label_integrity** ( makes sure all files in the dataset adhere to label conventions for training)
- **check_split_leakage** ( makes sure there are no near duplicates across splits )
- **count_class_distribution** ( sanity check for class distribution before and after augmenting )

Specific to the "Bone Fracture Detection Dataset:

Upon further investigation the original "humerus" and "humerus fracture" class were both merged into one "humerus fracture" class
- because they both contain fractured humeri
- to standardize naming conventions across the classes

To reduce computation time on a simpler model, the 'coords' of the outlines of the fractures were reduced to simple 
rectangular bounding boxes via the 'convert_split_to_bbox' function

Before training the images were further augmented to account for slightly misaligned, shifted or out of focus framing. 
With in-built ultralytics functionality the following parameters were applied


| Parameter | value | Reasoning
|  --------- | ------ | ---------- | 
|hsv_v | 0.2 | Mild brightness jitter, exposure variation |
| hsv_h | 0| Off-grayscale X-rays, hue has no meaning |
|hsv_s|0.0| Off- same reasoning |
|  scale | 0.2 | Mild zoom, patient distance variation |
| degrees|5 | Small rotation only, X-rays are near-upright|
|translate|0.1|  Mild framing variation
| shear|0.0| Off, unlikely for X-ray|
| flipud|0.0| Off, unlikely for X-ray|
|fliplr|0.5| On, no left/ right disctinction in classes |
| mosaic|0.0| Off, doesn't map to single-subject images |

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

## Discussion
Surprisingly, the performance does not track training-set size in the way expected. fingers positive (531 train examples) has the worst test mAP, while humerus fracture (314 examples) performs best. 
This is  likely due to some fracture types (fingers, wrist, elbow) are visually subtler/smaller in the X-ray than others (humerus, forearm), making them intrinsically harder to detect regardless of example count. 
At the same time the test set has very few instances per class (20-48), so individual metrics carry real variance/noise.

Worth highlighting are the precision (0.000) and recall (0.000) of the elbow positive class.
This result was also recovered by another YOLO model, specifically YOLOv11 training on the same dataset but with the humerus / humerus fracture distinction in tact.
This suggests a systemic issue with the dataset's annotation consistency, rather than a model-capacity limitation.

## Challenges 

During a session it is likely that the label format have been updated between an earlier format and a later one. This lead to incompatibility issues with especially the **polygon_to_bbox** function and ultimately led to the construction of the idempotent **rebuild_dataset_verfied** function that from then on handled downloading the dataset and verifying its compatibility. For future projects it's highly advised to keep track of the different kaggle dataset versions.

## Limitations & Future Work

**Class-specific performance gaps.** As discussed above, the "positive"-named
classes (elbow, fingers, wrist) consistently underperform the "fracture"-named
classes (forearm, humerus, shoulder), which is a held across two
independently trained models. This points toward inconsistent annotation
quality across the original label set rather than a fixable training issue,
but it hasn't been rigorously confirmed. A closer manual audit of a sample
of "positive"-labeled images against their bounding boxes would help
determine whether this is truly an annotation problem or something else
entirely.

**Overfitting.** Validation loss began rising while training loss continued
to fall, and mAP plateaued well before training completed, which is a sign the model
began memorizing the training set rather than generalizing further.
Deliberately disabling several default augmentations (mosaic, hue/saturation
shift) for domain-realism reasons likely reduced the model's built-in
regularization, on an already modest ~3,600-image dataset. Worth exploring:
stronger weight decay, more aggressive early stopping, or reintroducing a
carefully chosen subset of augmentations even at some cost to realism.

**Untested augmentation choices.** The augmentation parameters above were
chosen by domain reasoning (e.g. disabling color augmentation since X-rays
are grayscale) rather than empirical testing. Two choices in particular are
flagged for a future ablation study:
- `fliplr` (horizontal flip) — enabled on the assumption that none of the
  current classes distinguish left/right anatomy. Worth confirming this
  empirically rather than by assumption alone.
- `mosaic` — disabled on the reasoning that stitching four images together
  doesn't suit single-subject medical X-rays. Also untested directly.

**Small test set, high metric variance.** Per-class test instance counts
range from just 6 to 48 images. At this scale, a handful of hard or easy
examples can swing a class's reported mAP substantially. The results should be
read as indicative, not precise, especially for the smallest classes.

**Bounding boxes over segmentation.** The source dataset provides polygon
(segmentation-style) annotations, which were deliberately converted to
simple bounding boxes to keep training scope and compute manageable within
project constraints. Training a YOLOv8 segmentation model directly on the
original polygons. This trades off a chunk of speed for the model outputting
actual fracture *shape* rather than just an approximate box. This is a natural
next step if more time/compute becomes available.

**Dataset versioning.** `kagglehub` always pulls the latest version of a
dataset, which changed mid-project (the label format shifted from a fixed
5-value structure to variable-length polygons between two downloads in the
same session). Future iterations should pin a specific dataset version for
reproducibility, and re-run the integrity checks in
[`data_processing.py`](https://github.com/baotle/BoneFractureDetection_Computervision_Project/blob/main/src/bonefracture/data_processing.py)
after any redownload rather than assuming the format is stable.

## References

- Jocher, G., Chaurasia, A., & Qiu, J. (2023). Ultralytics YOLOv8 (Version 8.0.0) [Computer software]. https://github.com/ultralytics/ultralytics
- Dataset: Bone Fracture Detection, Roboflow Universe (veda), https://universe.roboflow.com/veda/bone-fracture-detection-daoon
- How to write a Good README : https://www.freecodecamp.org/news/how-to-write-a-good-readme-file/  
- Blankname78 kaggle notebook:  https://www.kaggle.com/code/blankname78/yolov11
- [Kaggle](https://www.kaggle.com/static/images/logos/kaggle-logo-transparent-300.png)  
*Kaggle logo used for identification purposes; trademark of Kaggle Inc. (a Google company).*
