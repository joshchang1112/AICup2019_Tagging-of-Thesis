# AICup_2019_Tagging-of-Thesis

## Competition description

The aim of this competition is to attempt to solve a problem that has troubled researchers for long: “How do you design a system that could automatically analyze the abstract of a thesis and summarize utilized, compared, or newly proposed algorithm used within the theses?”

The contestants will be provided theses with the topic of Computer Science sourced from arXiv. The contestants should use the provided materials to predict if a sentence in a thesis should be classified as the following categories: Background, Objectives, Methods, Results, Conclusions, or Others. Note that a sentence may have multiple classifications, e.g. a sentence may be classified as both Objective and Methods.

## Installation

First you need to install PyTorch. Please refer to [Pytorch installing page](https://pytorch.org/get-started/locally/#start-locally) regarding the specific install command for your platform.

When PyTorch has been installed, Transformers can be installed using pip as follows:
```
pip install transformers
```

Other packages can be also installed using pip as follows:
```
pip install -r requirements.txt
```

##
