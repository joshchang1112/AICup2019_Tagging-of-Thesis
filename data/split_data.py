import pandas as pd
from sklearn.model_selection import train_test_split

# random seed: 210, train:valid = 0.8 : 0.2
dataset = pd.read_csv('task1_trainset.csv')
train, valid = train_test_split(dataset, test_size=0.2, random_state=210)

train.to_csv('trainset.csv', index=False)
valid.to_csv('validset.csv', index=False)

