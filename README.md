# AICup_2019_Tagging-of-Thesis

## Competition description

The aim of this competition is to attempt to solve a problem that has troubled researchers for long: “How do you design a system that could automatically analyze the abstract of a thesis and summarize utilized, compared, or newly proposed algorithm used within the theses?”

The contestants will be provided theses with the topic of Computer Science sourced from arXiv. The contestants should use the provided materials to predict if a sentence in a thesis should be classified as the following categories: Background, Objectives, Methods, Results, Conclusions, or Others. Note that a sentence may have multiple classifications, e.g. a sentence may be classified as both Objective and Methods.

## Score Leaderboard
Team Name: 公鹿總冠軍 (Milwaukee Bucks Champion:trophy:)

Public Score:

0.746900 (Rank:1/469)

Private Score:

0.743458 (Rank:1/469)

## Installation

To execute our code successfully, you need to install Python3.7 and PyTorch (our deep learning framework) first. Please refer to [Python installing page](https://www.python.org/downloads/) and [Pytorch installing page](https://pytorch.org/get-started/locally/#start-locally) regarding the specific install command for your platform.

When PyTorch has been installed, Transformers can be installed using pip as follows:
```
pip install transformers
```

Other packages can be also installed using pip as follows:
```
pip install -r requirements.txt
```

## How to execute our code?

1. Split task1_trainset.csv (raw train data) to trainset.csv and validset.csv
```
cd data/
python split_train.py
```

2. Preprocess the data
```
cd src/
python make_dataset.py ../data/
```

3. Train model
```
python train.py ../models/
```

4. Predict 
```
python predict.py ../models/ --epoch 3
```
where `--epoch` specifies the save model of which epoch to use.

Note: You should comment out the 50th line in `src/dataset.py` so that you can successfully run the prediction code :)

## Experiments


### Output Embedding


Model |  Max Seq Length | Output Embedding  |  Validation F1 | Public Test F1 |
----- |:--------------: |:----------: | :-------------:| :-------------:|
`Roberta-Base`|   500  |[SEP] token |  0.7331 | 0.7345   | 
`Roberta-Base`|   500  | Mean Pooling |  0.7359 | 0.7355   | 

### Different Pretrained Model

Model |  Max Seq Length | Output Embedding  |  Validation F1 | Public Test F1 |
----- |:--------------: |:----------: | :-------------:| :-------------:|
`Roberta-Base`|   500  | Mean Pooling |  0.7462 | 0.7344   | 
`Roberta-Large`|   500  | Mean Pooling |  0.7468 | 0.7412   | 
`XLNet-Large-cased`|   600  | Mean Pooling |  0.7470 | 0.7413   | 


### Different Learning Rate


### Advanced Text Preprocess

### Ensemble

The categorical F1 score of each settings are recorded in the log file (`results/log.txt`). This information may help improve our ensemble strategy.


## How to reproduce our results?

Detailed settings about our model are recorded in the log file (`results/log.txt`). You can follow the settings to train the model yourself and get our experiment results above. To reach our highest score on the leaderboard, you need to ensemble our predictions by executing the simple weighted voting code as follows:

```
cd results/
python ensemble.py
```

## Contact information

For help or issues using our code, please contact Sung-Ping Chang (`joshspchang@gmail.com`).
