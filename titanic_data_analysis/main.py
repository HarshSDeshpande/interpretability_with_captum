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
torch.manual_seed(424242)

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
if MAIN:
    out_probs = net(input_tensor).detach().numpy()
    out_classes = np.argmax(out_probs, axis=1)
    print('Training Accuracy: {:.2f}'.format(np.mean(out_classes == train_labels)))
# %%
if MAIN:
    test_input_tensor = torch.from_numpy(test_features).type(torch.FloatTensor)
    out_probs = net(test_input_tensor).detach().numpy()
    out_classes = np.argmax(out_probs, axis=1)
    print('Test Accuracy: {:.2f}'.format(np.mean(out_classes == test_labels)))
# %%
if MAIN:
    ig = IntegratedGradients(net)
    test_input_tensor.requires_grad_()
    attr,delta = ig.attribute(test_input_tensor, target=1,return_convergence_delta = True)
    attr = attr.detach().numpy()
# %%
def visualize_importances(feature_names, importances, title = "Average Feature Importances", plot = True, axis_title="Features"):
    print(title)
    for i in range(len(feature_names)):
        print(feature_names[i],":",importances[i])
    x_pos = np.arange(len(feature_names))
    if plot:
        plt.figure(figsize=(12,6))
        plt.bar(x_pos, importances,align='center')
        plt.xticks(x_pos,feature_names,wrap=True)
        plt.xlabel(axis_title)
        plt.title(title)
if MAIN:
    visualize_importances(feature_names,np.mean(attr,axis=0))
# %%
if MAIN:
    plt.hist(attr[:,2],100)
    plt.title("Distribution of Sibsp Attribution Values")

# %%
if MAIN:
    bin_means,bin_edges, _  = stats.binned_statistic(test_features[:,2],attr[:,2],statistic='mean',bins=6)
    bin_count,_,_ = stats.binned_statistic(test_features[:,2],attr[:,2],statistic='count',bins=6)
    bin_width = (bin_edges[1]-bin_edges[0])
    bin_centers = bin_edges[1:] - bin_width/2
    plt.scatter(bin_centers, bin_means,s=bin_count)
    plt.xlabel("Average Sibsp Feature Value")
    plt.ylabel("Average Attribution")
# %%
if MAIN:
    cond = LayerConductance(net, net.sigmoid1)
    cond_vals = cond.attribute(test_input_tensor, target =1)
    cond_vals = cond_vals.detach().numpy()
    visualize_importances(range(13),np.mean(cond_vals,axis=0),title="Average Neuron Importances in Sigmoid Layer",axis_title="Neurons")
# %%
if MAIN:
    plt.hist(cond_vals[:,9],100)
    plt.title("Neuron 9 Distribution")
    plt.figure()
    plt.hist(cond_vals[:,7],100)
    plt.title("Neuron 7 Distribution")
# %%
if MAIN:
    plt.hist(cond_vals[:,0],100)
    plt.title("Neuron 0 Distribution")
    plt.figure()
    plt.hist(cond_vals[:,10],100)
    plt.title("Neuron 10 Distribution")
# %%
if MAIN:
    neuron_cond = NeuronConductance(net, net.sigmoid1)
    