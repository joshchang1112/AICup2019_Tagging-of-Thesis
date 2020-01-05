# AICup_2019_Tagging-of-Thesis

## Competition description

The aim of this competition is to attempt to solve a problem that has troubled researchers for long: “How do you design a system that could automatically analyze the abstract of a thesis and summarize utilized, compared, or newly proposed algorithm used within the theses?”

The contestants will be provided theses with the topic of Computer Science sourced from arXiv. The contestants should use the provided materials to predict if a sentence in a thesis should be classified as the following categories: Background, Objectives, Methods, Results, Conclusions, or Others. Note that a sentence may have multiple classifications, e.g. a sentence may be classified as both Objective and Methods.

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

1. Split task1_trainset.csv (raw train data) to trainset.csv and validset.csv data
```
cd data/
python split_train.py
```

2. Preprocess the data
```
cd src/
python make_dataset.py ../data/
```

3. Training:
```
python train.py ../models/
```

4. Predicting
```
python predict.py ../models/ --epoch 3
```
where `--epoch` specifies the save model of which epoch to use.

## Framework


## Experiments


### Output Embedding


Model |  Max Seq Length | Output Embedding  |  Validation F1 | Public Test F1 |
----- |:--------------: |:----------: | :-------------:| :-------------:|
`Roberta-Base`|   400  |[SEP] token |  0.7331 | 0.7345   | 
`Roberta-Base`|   400  | Mean Pooling |  0.7359 | 0.7355   | 

### Different Pretrained Model


### Different Learning Rate


### Advanced Text Preprocess


## Score Leaderboard
Team Name: 公鹿總冠軍 (Milwaukee Bucks Champion:trophy:)

Public Score:

0.746900 (Rank:1/469)

Private Score:

0.743458 (Rank:1/469)

## How to reproduce our results?

Detailed settings about our model is in the log file (`results/log.txt`). You can follow the settings to train the model yourself and get our experiment results above. To reach our highest score on the leaderboard, you need to ensemble our predictions by executing the simple weighted voting code as follows:

```
cd src/
python ensemble.py ../results
```

## Contact information

For help or issues using our code, please contact Sung-Ping Chang (`joshspchang@gmail.com`).
