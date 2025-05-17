# K-Half

## Requirements

Ensure you have the following dependencies installed:

```bash
pip install torch==11.3 transformers numpy scikit-learn matplotlib tqdm
```

## Usage

### Labeling the Dataset

```bash
python label.py --dataset <DATASET_NAME>
```

Example:

```bash
python label.py --dataset ICEWS14
```
### Training the Model

```bash
python train.py --dataset <DATASET_NAME> --train_file <train|test|valid> --entity_out_dim_1 <DIM> --entity_out_dim_2 <DIM> --epochs <NUM_EPOCHS> --batch <BATCH_SIZE> --threshold <THRESHOLD>
```

Example:

```bash
python train.py --dataset ICEWS14 --train_file train --entity_out_dim_1 32 --entity_out_dim_2 32 --epochs 50 --batch 5000 --threshold 0
```

### Process of Filtering Outdated Facts 

```bash
python out.py --dataset <DATASET_NAME> --train_file <train|test|valid> --threshold <VALIDITY_THRESHOLD>
```

Example:

```bash
python out.py --dataset ICEWS14 --train_file train --threshold 0.01
```

Place data-input.py into the dataset folder to be trained. For example, if you need to train the ICEWS14 dataset, put data-input.py into the ICEWS14 folder and run the following code

```bash
cd ICEWS14
python data-input.py
```
The dataset at this point is the dataset after filtering for outdated knowledge

### Reasoing over Temporal Knowledge Graphs

We have put the link to the Baseline model used in the paper below for readers to use:

CYGNet：https://github.com/CunchaoZ/CyGNet

xERTE：https://github.com/TemporalKGTeam/xERTE

REGCN：https://github.com/Lee-zix/RE-GCN

CENET：https://github.com/xyjigsaw/CENET

LogCL：https://github.com/WeiChen3690/LogCL

Replace the dataset obtained in the previous step that filters out outdated knowledge with the original dataset in the folder of the model you want to use, make sure that the filename and format are the same, and then just run the training step for that model normally.

