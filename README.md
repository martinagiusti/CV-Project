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
    ├── ccpd_base
    │   ├── 00205459770115-90_85-352&516_448&547-444&547_368&549_364&517_440&515-0_0_22_10_26_29_24-128-7.jpg
    │   ├── 00221264367816-91_91-283&519_381&553-375&551_280&552_285&514_380&513-0_0_7_26_17_33_29-95-9.jpg
    │   ├── 00223060344828-90_89-441&517_538&546-530&552_447&548_447&512_530&516-0_0_13_16_33_30_33-148-14.jpg
    │   ├── 00224137931034-90_87-351&564_451&606-440&599_362&600_359&572_437&571-0_0_3_21_30_28_24-88-5.jpg
    │   ├── 00227490421455-90_88-341&443_436&482-437&479_338&486_335&452_434&445-0_0_9_33_33_29_28-128-12.jpg
    │   └── ...
    ├── ccpd_blur
    │   ├── 0019-1_1-340&500_404&526-404&524_340&526_340&502_404&500-0_0_11_26_25_28_17-66-3.jpg
    │   ├── 0022-0_1-360&474_433&500-432&500_360&499_361&474_433&475-0_0_20_24_26_27_10-143-5.jpg
    │   ├── 0022-0_2-289&482_362&508-361&508_289&508_290&482_362&482-0_0_3_11_31_25_33-94-7.jpg
    │   ├── 0022-0_4-337&385_411&410-411&410_339&410_337&385_409&385-0_0_13_24_9_31_30-74-6.jpg
    │   ├── 0023-0_0-290&386_367&412-367&412_290&411_290&386_367&387-0_0_23_26_5_31_24-69-2.jpg
    │   └── ...
    ├── ccpd_challenge
    │   ├── 0018-0_7-339&483_411&505-411&505_342&505_339&483_408&483-0_0_16_9_27_27_27-82-8.jpg
    │   ├── 0018-3_1-284&522_343&548-341&548_284&545_286&522_343&525-0_0_18_31_29_32_8-74-14.jpg
    │   ├── 0019-1_1-340&500_404&526-404&524_340&526_340&502_404&500-0_0_11_26_25_28_17-66-3.jpg
    │   ├── 0020-0_3-334&519_404&544-402&544_334&543_336&519_404&520-0_0_8_33_17_24_32-55-14.jpg
    │   ├── 0021-0_0-312&484_385&509-385&509_312&509_312&484_385&484-0_0_13_25_4_24_32-66-19.jpg
    │   └── ...
    ├── ccpd_db
    │   ├── 0029-1_0-295&497_376&527-375&527_295&525_296&497_376&499-0_0_18_33_23_32_32-197-25.jpg
    │   ├── 0030-0_0-482&504_553&540-553&539_482&540_482&505_553&504-10_6_25_15_32_29_31-43-19.jpg
    │   ├── 0034-0_2-341&512_432&544-431&543_341&544_342&513_432&512-0_0_30_33_32_16_30-199-5.jpg
    │   ├── 0034-0_3-298&497_388&529-386&529_298&529_300&497_388&497-0_0_6_25_27_24_23-39-6.jpg
    │   ├── 0036-0_9-560&630_652&663-647&662_560&663_565&631_652&630-0_0_29_32_20_26_27-38-4.jpg
    │   └── ...
    ├── ccpd_fn
    │   ├── 0018-0_7-339&483_411&505-411&505_342&505_339&483_408&483-0_0_16_9_27_27_27-82-8.jpg
    │   ├── 0018-3_1-284&522_343&548-341&548_284&545_286&522_343&525-0_0_18_31_29_32_8-74-14.jpg
    │   ├── 0019-1_1-340&500_404&526-404&524_340&526_340&502_404&500-0_0_11_26_25_28_17-66-3.jpg
    │   ├── 0020-0_3-334&519_404&544-402&544_334&543_336&519_404&520-0_0_8_33_17_24_32-55-14.jpg
    │   ├── 0021-0_0-312&484_385&509-385&509_312&509_312&484_385&484-0_0_13_25_4_24_32-66-19.jpg
    │   └── ...
    ├── ccpd_np
    │   ├── 1005.jpg
    │   ├── 1019.jpg
    │   ├── 1027.jpg
    │   ├── 1029.jpg
    │   ├── 1038.jpg
    │   └── ...
    ├── ccpd_rotate
    │   ├── 0038-16_16-346&482_408&534-408&534_346&516_346&482_408&500-0_0_4_11_31_27_31-63-30.jpg
    │   ├── 0050-14_17-232&510_305&568-305&568_234&550_232&510_303&528-0_0_20_3_27_27_24-87-4.jpg
    │   ├── 0056-15_17-299&542_370&608-368&589_299&608_301&561_370&542-0_0_31_8_27_27_33-39-27.jpg
    │   ├── 0056-16_14-363&486_439&548-438&548_363&526_364&486_439&508-0_0_4_0_25_30_29-137-11.jpg
    │   ├── 0056-18_11-304&528_375&595-369&595_304&573_310&528_375&550-0_0_21_28_26_27_15-91-26.jpg
    │   └── ...
    ├── ccpd_tilt
    │   ├── 0038-16_16-346&482_408&534-408&534_346&516_346&482_408&500-0_0_4_11_31_27_31-63-30.jpg
    │   ├── 0042-14_14-379&452_444&507-444&507_379&490_379&452_444&469-0_0_23_32_32_30_26-65-15.jpg
    │   ├── 0045-15_19-365&473_431&531-431&531_368&514_365&473_428&490-0_11_10_33_33_30_31-69-25.jpg
    │   ├── 0050-14_17-232&510_305&568-305&568_234&550_232&510_303&528-0_0_20_3_27_27_24-87-4.jpg
    │   ├── 0054-16_16-309&494_381&557-381&557_309&536_309&494_381&515-0_0_2_0_24_33_32-104-11.jpg
    │   └── ...
    ├── ccpd_weather
    │   ├── 0042-5_0-294&496_374&540-374&533_298&540_294&503_370&496-0_0_3_25_27_24_20-92-5.jpg
    │   ├── 0044-0_6-304&542_411&577-407&577_304&577_308&542_411&542-0_0_18_17_26_30_31-63-8.jpg
    │   ├── 0045-0_1-339&547_444&583-443&583_339&582_340&547_444&548-0_11_23_28_32_32_26-54-16.jpg
    │   ├── 0045-5_4-177&511_268&553-262&553_177&545_183&511_268&519-0_0_19_3_24_26_32-71-15.jpg
    │   ├── 0046-0_1-402&426_497&467-497&467_403&467_402&426_496&426-0_15_10_26_26_26_13-129-17.jpg
    │   └── ...
    ├── splits
    │   ├── ccpd_blur.txt
    │   ├── ccpd_challenge.txt
    │   ├── ccpd_db.txt
    │   ├── ccpd_fn.txt
    │   ├── ccpd_rotate.txt
    │   ├── ccpd_tilt.txt
    │   ├── test.txt
    │   ├── train.txt
    │   └── val.txt
    ├── LICENSE
    └── README.md
```
The validation and train dataset are composed of images in ccpd_base, while the test set is build up by all the other folders.

# About the repository:
The notebook [YOLOv5_detection.ipynb](https://github.com/martinagiusti/CV-Project/blob/main/YOLOv5_detection.ipynb) is dedicated to the YOLOv5s network development and training.
The notebook \mettere_nome is about the PDLPR network, taking as example the proposal on \metttere_link_paper from the network structure point of view.
The notebook \mettere_nome deals with Faster-R-CNN network development.
The notebook [STAR-Net_recognition.ipynb](https://github.com/martinagiusti/CV-Project/blob/main/STAR-Net_recognition.ipynb) is about STAR-Net whose structure is taken from [deep-text-recognition-benchmark](https://github.com/clovaai/deep-text-recognition-benchmark).

- csv_splits folder: contains the csv files of each ccpd test folder.
- deep-text-recognition-benchmarck: copy of the repository [deep-text-recognition-benchmark](https://github.com/clovaai/deep-text-recognition-benchmark). There are provided also the modified versions of the files in the repository.
- yolov5: is the copy of [YOLOv5](https://github.com/ultralytics/yolov5) repository, and inside this folder there is the modified version of val.py.

---

## Notebooks

### 1. **Dataset_CCPD.ipynb**
- **Operations**: 
  - Image size, format, color channels, and pixel range inspection.
  - Organizes informations into CSVs based on folders (e.g., `train.csv`, `ccpd_blur.csv`).
  - Visualizes images with bounding boxes and annotations.

### 2. **Faster RCNN_detection.ipynb**
- **Model**: Faster R-CNN with ResNet50 FPN backbone for license plate detection.
- **Features**: 
  - Custom dataset handling, training, evaluation, test.
  - Model checkpoints saved for further use.

### 3. **STAR-Net**
- **Model**: TPS-ResNet-BiLSTM-Attn modules.
- **Features**:
  - Custom dataset handling, training, evaluation, test.
  - Download the [deep-text-recognition-benchmark](https://github.com/clovaai/deep-text-recognition-benchmark), use the modified scripts attached in [deep-text-recognition-benchmark folder](https://github.com/martinagiusti/CV-Project/tree/main/deep_text_recognition_benchmark).
  - Model checkpoints saved for further use in [checkpoints](https://github.com/martinagiusti/CV-Project/tree/main/checkpoints).

### 4. **YOLO_Detection.ipynb**
- **Model**: Yolo v5s for license plate detection.
- **Features**:
  - Custom dataset formatted for YOLO.
  - Model checkpoints saved for further use.
  - **Configuration Files**:
    - **`data.yaml`**: This file contains dataset-specific configurations, such as paths to the training and validation images, class names, and other necessary details. It should be placed in the `yolov5/` directory.
    - **`hyp_ccpd.yaml`**: This file includes custom hyperparameters specific to training on the CCPD dataset. It should be placed in the `data/hyps` directory or as specified in the YOLOv5 repository instructions.


### 5. **PDLPR_recognition.ipynb**
- **Architecture**: The PDLPR model follows the structure described in the paper [A Real‑Time License Plate Detection and Recognition Model in
Unconstrained Scenarios](https://www.mdpi.com/1424-8220/24/9/2791). It includes:
  - **Improved Global Feature Extractor (IGFE)**.
  - **Encoder**.
  - **Parallel Decoder**.

- **Features**:
  - Custom dataset handling, training, evaluation, test.
  - Model checkpoints saved for further use.

---

## How to Use
1. Download the dataset from [CCPD GitHub](https://github.com/detectRecog/CCPD).
2. Ensure dataset is correctly placed in the same directory of the notebooks.
3. Run notebooks to train and evaluate models.
4. Use pre-trained weights for testing.

---

## Model Weights and training history
- Pre-trained weights are available for both detection and recognition tasks.
  - **Faster RCNN (Detection)**: `best_model_iou.pth`,`checkpoint_full.pth`.
  - **Yolo v5s (Detection)**: `best.pt`, `last.pt`.
  - **PDLPR (Recognition)**: `checkpoint_epoch75.pth`, `latest_checkpoint.pth`, `training_history.json`.




