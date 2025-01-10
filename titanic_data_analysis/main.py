# %%
import numpy as np
import torch
import torch.nn as nn
from captum.attr import IntegratedGradients
from captum.attr import LayerConductance
from captum.attr import NeuronConductance
import  matplotlib
import matplotlib.pyplot as plt
# %matplotlib inline
from scipy import stats
import pandas as pd
MAIN = False
if __name__ == "__main__" : 
    MAIN = True
# %%
if MAIN: 
    dataset_path = "titanic.csv"
# %%
if MAIN:
    titanic_data = pd.read_csv(dataset_path)
# %%
if MAIN:
    titanic_data = pd.concat([titanic_data,
                            pd.get_dummies(titanic_data['Sex']),
                            pd.get_dummies(titanic_data['Embarked'],prefix="embark"),
                            pd.get_dummies(titanic_data['Pclass'],prefix="class")], axis = 1)
    titanic_data['Age'] = titanic_data['Age'].fillna(titanic_data['Age'].mean())
    titanic_data['Fare'] = titanic_data['Fare'].fillna(titanic_data['Fare'].mean())
    titanic_data = titanic_data.drop(['Name','Ticket','Cabin','Sex','Embarked','Pclass'],axis=1)
# %%
np.random.seed(123456)

if MAIN:
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
torch.manual_seed(42)

class TitanicSimpleNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(13,13)
        self.sigmoid1 = nn.Sigmoid()
        self.linear2 = nn.Linear(13,8)
        self.sigmoid2 = nn.Sigmoid()
        self.linear3 = nn.Linear(8,2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self,x):
        lin1_out = self.linear1(x)
        sigmoid_out1 = self.sigmoid1(lin1_out)
        sigmoid_out2 = self.sigmoid2(self.linear2(sigmoid_out1))
        return self.softmax(self.linear3(sigmoid_out2))
# %%
if MAIN:
    net =TitanicSimpleNNModel()
    criterion = nn.CrossEntropyLoss()
    num_epochs = 200

    optimizer = torch.optim.Adam(net.parameters(),lr=0.1)
    input_tensor = torch.from_numpy(train_features).type(torch.FloatTensor)
    label_tensor = torch.from_numpy(train_labels)
    for epoch in range(num_epochs):
        output = net(input_tensor)
        loss = criterion(output, label_tensor)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch%20==0:
            print('Epoch {}/{} => Loss: {:.2f}'.format(epoch+1,num_epochs,loss.item()))

    torch.save(net.state_dict(),'titanic_model.pt')
# %%
