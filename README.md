# CV-Project

The aim of the project is to compare the different performance in two different License Plate detection and recognition pipeplines in order to recognize Chinese License Plates.
The different pipelines are
1. YOLOv5 + PDLPR, proposed by \mettere la citazione del paper;
2. Fater-R-CNN + STAR-Net.

The dataset used is the Chinese City Parking Dataset, in 2019 version, which can be downloaded following the instruction at the [CCPD](https://github.com/detectRecog/CCPD) repository. 
The dataset splits are given by the train.txt, val.txt and test.txt. These will provide the directory of the chosen files for each split.
Because the CCPD labels are contained in the image name, these informations are extracted in the corresponding csv files: train.csv, test.csv and val.csv.

# CCPD dataset:
The dataset is divided in the following folders:
```bash
CCPD2019
|--ccpd_base
|--ccpd_blur
|--ccpd_challenge
|--ccpd_db
|--ccpd_fn
|--ccpd_rotate
|--ccpd_tilt
|--ccpd_weather
```
The validation and train dataset are composed of images in ccpd_base, while the test set is build up by all the other folders.

# About the repository:
The notebook [YOLOv5_detection.ipynb]() is dedicated to the YOLOv5s network development and training.
The notebook \mettere_nome is about the PDLPR network, taking as example the proposal on \metttere_link_paper from the network structure point of view.
The notebook \mettere_nome deals with Faster-R-CNN network development.
The notebook [STAR-Net_recognition.ipynb]() is about STAR-Net whose structure is taken from [deep-text-recognition-benchmark](https://github.com/clovaai/deep-text-recognition-benchmark).

- csv_splits folder: contains the csv files of each ccpd test folder.
- deep-text-recognition-benchmarck: copy of the repository [deep-text-recognition-benchmark](https://github.com/clovaai/deep-text-recognition-benchmark). There are provided also the modified versions of the files in the repository.
- yolov5: is the copy of [YOLOv5](https://github.com/ultralytics/yolov5) repository, and inside this folder there is the modified version of val.py.
