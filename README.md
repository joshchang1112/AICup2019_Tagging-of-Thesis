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
```
***** Constant Settings *****
Optimizer: Adam
Scheduler: CosineAnnealing

```


Model        | Seq Length | Batch Size    | Learning Rate  | Token / Mean pooling | Replace Contraction
------------ | ---------- | ----------    | -------------  | -------------------- | -------------------
`BERT-Base`  | 64         | 64            |                |                      |



## Contact information
