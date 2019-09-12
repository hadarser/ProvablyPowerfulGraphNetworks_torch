# Provably Powerful Graph Networks
This repository holds a pytorch version of the code for the paper: https://arxiv.org/abs/1905.11136.
The code used to achieve the results in the paper is using tensorflow and can be found in:
https://github.com/hadarser/ProvablyPowerfulGraphNetworks.
If you want to learn more about it, check out these two blog posts explaining invariant graph networks and their power: http://irregulardeep.org

## Data
Before running the code the data should be downloaded using the following commands:

```
cd ProvablyPowerfulGraphNetworks
python utils/get_data.py
```

This script will download all the data from Dropbox links. 



## Code

### Prerequisites

python 3.7.3

pytorch 1.2

Additional modules: numpy, pandas, matplotlib, tqdm, easydict



### Running the tests

The folder main_scripts contains scripts that run different experiments:
1. To run 10-fold cross-validation with our chosen hyper-parameters, run the main_10fold_experiment.py script. You can choose the dataset in 10fold_config.json or using the command line option. These hyper-parameters refer to version 1 from the paper. 
example:
to run 10-fold cross-validation experiment:
```
python main_scripts/main_10fold_experiment.py --config=configs/10fold_config.json --dataset_name=NCI1
```
2. To run the QM9 experiment with our hyper-parameters, run the main_qm9_experiment.py script:
```
python main_scripts/main_qm9_experiment.py --config=configs/qm9_config.json
```

In the paper, we have two models for the QM9 task: the first predicts a all the outputs quantity at once, while the other predicts a single chosen output quantity. You can switch between these models by changing the following line in the configs/qm9_config.json:
```
  "target_param": false,
```
to the chosen target (range 0-11).


Note: The script mentioned in the data section above will download a processed version of QM9 which is needed for our main code. We also share our processing code (requires pytorch), which is is based on the pytorch-geometric package.
see: https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/datasets/qm9.html#QM9:

```
cd ProvablyPowerfulGraphNetworks
python utils/get_qm9_data.py
```
