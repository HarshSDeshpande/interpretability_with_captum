# %%
import numpy as np
import torch
from captum.attr import IntegratedGradients
from captum.attr import LayerConductance
from captum.attr import NeuronConductance
import  matplotlib
import matplotlib.pyplot as plt
# %matplotlib inline
from scipy import stats
import pandas as pd
NAME = False
if __name__ == "__main__" : 
    NAME = True
# %%
dataset_path = "titanic.csv"
# %%
titanic_data = pd.read_csv(dataset_path)
# %%
titanic_data = pd.concat([titanic_data,
                          pd.get_dummies(titanic_data['Sex']),
                          pd.get_dummies(titanic_data['Embarked'],prefix="embark"),
                          pd.get_dummies(titanic_data['Pclass'],prefix="class")], axis = 1)
titanic_data['Age'] = titanic_data['Age'].fillna(titanic_data['Age'].mean())
titanic_data['Fare'] = titanic_data['Fare'].fillna(titanic_data['Fare'].mean())
titanic_data = titanic_data.drop(['Name','Ticket','Cabin','Sex','Embarked','Pclass'],axis=1)
# %%
np.random.seed(123456)

labels = titanic_data['Survived'].to_numpy()
titanic_data = titanic_data.drop(['Survived'],axis=1)
feature_names = list(titanic_data.columns)
data = titanic_data.to_numpy()

train_indices = np.random.choice(len(labels), int(0.7*len(labels)),replace=False)
test_indices = list(set(range(len(labels))) - set(train_indices))
train_features = np.array(data[train_indices],dtype=float)
train_labels = labels[train_indices]
test_features = np.array(data[test_indices],dtype=float)
test_labels = labels[test_indices]
# %%
