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

We tried two method to extract the information of the output embedding from the pretrain model.

#### SEP Token
We add the sep token between the sentences and extract its position of the output embedding as the representation of the sentence. 

#### Mean Pooling
Mean pooling is applied over the output embedding of every words in the sentence to extract the information as the representation of the sentence. 

The comparison of the two settings is shown in the table below:

Model |  Max Seq Length | Output Embedding  |  Validation F1 | Public Test F1 |
----- |:--------------: |:----------: | :-------------:| :-------------:|
`Roberta-Base`|   500  |[SEP] token |  0.7331 | 0.7345   | 
`Roberta-Base`|   500  | Mean Pooling |  0.7359 | 0.7355   | 

### Different Pretrained Model

We tried several pretrained models and compared their F1 score. After many trials and tuning hyperparameters, we decided to use `Roberta-Base`, `Roberta-Large`, and `XLNet-Large-cased` as our models.

Model |  Max Seq Length | Output Embedding  |  Validation F1 | Public Test F1 |
----- |:--------------: |:----------: | :-------------:| :-------------:|
`Roberta-Base`|   500  | Mean Pooling |  0.7462 | 0.7344   | 
`Roberta-Large`|   500  | Mean Pooling |  0.7468 | 0.7412   | 
`XLNet-Large-cased`|   600  | Mean Pooling |  0.7470 | 0.7413   | 


### Different Learning Rate

Model |  Max Seq Length | Batch size  | Learning Rate |  Validation F1 | Public Test F1 |
----- |:--------------: |:----------: |:----------: | :-------------:| :-------------:|
`Roberta-Large`|   500  | 6        |  8e-6       | 0.7465 | 0.7388   | 
`Roberta-Large`|   500  | 4        |  6e-6       | 0.7468 | 0.7412   | 

### Advanced Text Preprocess

#### I. Replace Contraction

To let the model better understand the relationship between words, we unify some words with the same meaning but different writing. For example: will not/won't, can not/can't, do not/don't ...etc

#### II & III. Remove(i)(ii)(iii).../ Replace (i)(ii)... to first, second...

In the abstract, we often see authors using (i)(ii)(iii) to illustrate his research. However, these symbols may confuse the model and understand them as multiple I(me). Therefore, we tried two method to solve the problems. The first one is just remove these symbols, and the second one is replace them as first, second...

The combination of these methods help me reach my best Validation F1 score : 0.7484, and the comparison of the several settings is shown in the table below:

Model |  Max Seq Length | Adv. Text Preprocess  | Validation F1 | Public Test F1 |
----- |:--------------: |:---------------------:| :-------------:| :-------------:|
`Roberta-Large`|   500  | None |  0.7468 | 0.7412   | 
`Roberta-Large`|   500  | I |  0.7474 | 0.7416   | 
`Roberta-Large`|   500  | I + II |  0.7455 | 0.7397   | 
`Roberta-Large`|   500  | I + III |  $0.7484$ | 0.7397   |
`XLNet-Large-cased`|   600  | None |  0.7470 | 0.7413  |
`XLNet-Large-cased`|   600  | I |  0.7462 | 0.7414  |


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
