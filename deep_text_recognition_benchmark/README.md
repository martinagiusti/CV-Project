The repository [deep-text-recognition-benchmark]() is taken as a reference, but the main files where modified and all the modified ones are reported here in this folder.
The create_lmdb_dataset.py is used to create the dataset. See the folders train_lmdb and val_lmdb to see the commands to create the dataset.
train.py is used to train the network, selecting the desired modules, in the case of STAR-Net:

```bash
python deep_text_recognition_benchmark/train.py --train_data train_lmdb --valid_data val_lmdb ^
--select_data MJ-ST --batch_ratio 0.5-0.5 --Transformation TPS --FeatureExtraction ResNet ^
--SequenceModeling BiLSTM --Prediction Attn --saved_model saved_models\TPS-ResNet-BiLSTM-Attn.pth ^
--num_iter 10420 --valInterval 2084 --data_filtering_off
```

demo.py is for try a certain model through the command:

```bash
python deep_text_recognition_benchmark/demo.py --image_folder data_test/onlyLP_test^
--saved_model saved_models/TPS-ResNet-BiLSTM-Attn-Seed1111/best_accuracy.pth --Transformation TPS ^
--FeatureExtraction ResNet --SequenceModeling BiLSTM --Prediction Attn 
```

In particular, the file test_raw.py makes possible making the test analysis without having a dataset using mdb form. For this purpose, the command for doing the inference analysis is:

```bash
python deep_text_recognition_benchmark\test_raw.py --eval_data data_test_split\ccpd_blur\data --label_file data_test_split\ccpd_blur\labels.txt ^
--saved_model saved_models\TPS-ResNet-BiLSTM-Attn-Seed1111\best_accuracy.pth --Transformation TPS ^
--FeatureExtraction ResNet --SequenceModeling BiLSTM --Prediction Attn
```
The results of the inference are in folder DTRB_results.

test.py is the original function but performes the inference using mdb data.
