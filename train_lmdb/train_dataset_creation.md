To create the dataset, follow the instructions provided in [deep-text-recognition-benchmark](https://github.com/clovaai/deep-text-recognition-benchmark).

Run the following commands:

```bash
pip3 install fire
python create_lmdb_dataset.py --inputPath data_train/onlyLP_train --gtFile data_train/labels_train.txt --outputPath train_lmdb
```
then a data.mdb and lock.mdb will be created in train_lmdb.
these two files must be copied in two splitted folders in train_lmdb as follows:
```bash
train_lmdb
|
|----MJ
|     |--data.mdb
|     |--lock.mdb
|
|----ST
      |--data.mdb
      |--lock.mdb
```
As for the validation set the same proceedure is followed but the data.mdb and lock.mdb are gathered in a unique folder:
```bash
val_lmdb
|----data
       |--data.mdb
       |--lock.mdb
```
