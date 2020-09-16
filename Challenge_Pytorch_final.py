#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 20:09:46 2019

@author: sharma
"""
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
import pdb
import pandas as pd
from torch.utils.data import TensorDataset
# import torchvision.datasets as dsets
# import torchvision
# import torchvision.transforms as transforms
import sys
np.random.seed(1)
torch.manual_seed(1)


#Define a basic MLP with pyTorch
class MLP(nn.Module):
    def __init__(self): #pass in 2d input data for convolution. (K = number of layers)
        super(MLP, self).__init__()
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, kernel_size=3, stride=1),
            torch.nn.LeakyReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=1),
            torch.nn.Dropout(p=0.1))

        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, kernel_size=2, stride=1),
            torch.nn.LeakyReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=1),
            torch.nn.Dropout(p=0.1))

        self.fc1 = torch.nn.Linear(48128, 64)
        self.layer3 = torch.nn.Sequential(
            self.fc1,
            torch.nn.LeakyReLU())
        self.fc2 = torch.nn.Linear(64, 35)

        #torch.nn.init.xavier_uniform_(self.fc2.weight) # initialize parameters
        self.lossfunction= nn.CrossEntropyLoss(reduction='sum') # to make sure that the final probabilities sum to 1

    def _predict_proba(self, X):
        h1 =  self.layer1(X)
        h2 = self.layer2(h1)
        h2p = h2.view(h2.shape[0], -1)
        h3 = self.layer3(h2p)
        yhat = self.fc2(h3)
        return yhat

    def forward(self, X, y): #loss function (difference in prediction vs actual labels)
        yhat = self._predict_proba(X)
        loss = self.lossfunction(yhat,y.long())
        return loss

    def predict_proba(self, X): #model prediction

        yhat = self._predict_proba(X)
        return np.argmax(yhat.detach().numpy(), axis=1)

    def getPaddedData(self, arr):
      if arr.shape != (99,13):
         offset_axis0 = 99 - arr.shape[0]
         offset_axis1 = 13 - arr.shape[1]
         if offset_axis0 > 0:
           arr_ = np.pad(arr, ((0,offset_axis0), (0,offset_axis1)),'constant', constant_values=0)
           return arr_
      return arr

    def fit(self,X,y,verbose=False):
        # list of None indices
        N = X.shape[0]
        X2 = torch.zeros(X.shape[0], 1, X[0].shape[0] , X[0].shape[1])
        # create pytorch vector
        for i in range(len(X)):
            X2[i][0] = torch.from_numpy(self.getPaddedData(X[i])).float()
        y2 = torch.from_numpy(y)
        optimizer = optim.Adam(self.parameters(),lr=0.003,weight_decay=1e-4) #parameters that you want to optimize
        old_loss=np.inf
        batchsize = 128
        for epoch in range(15): # change epoch, data subset
            dataset = TensorDataset(X2, y2)#dataset = TensorDataset(X2, y2)
            loader = torch.utils.data.DataLoader(dataset, batch_size=batchsize, shuffle= True)
            for batch_ndx, sample in enumerate(loader):
              print ("==>>> batch num:  "  + str(batch_ndx))
              optimizer.zero_grad()
              batch_x, batch_y_class = sample
              loss = self.forward(batch_x,batch_y_class) #* (N/batchsize)#calls forward function
              loss.backward() #loss anusar gradient calculate garchha
              optimizer.step() # changes parameters according to the gradient
            if(np.abs(loss.data-old_loss)/np.abs(loss.data)<1e-6):
                break
            old_loss = loss.data
            if(verbose):
                print('==>>> epoch: {}, train loss: {:.6f}'.format(epoch, loss.data))
        print('==>>> epoch: {}, train loss: {:.6f}'.format(epoch, loss.data))

def main():
    #preprocessing steps
    features = np.load("feat.npy", allow_pickle=True)
    path = np.load("path.npy", allow_pickle=True)
    train_path_label = pd.read_csv("train.csv")
    #dictionary mapping path to train label
    path_label_dict= pd.Series(train_path_label.word.values,index=train_path_label.path.values).to_dict()

    test_paths = (pd.read_csv("test.csv"))[['path']].to_numpy()
    # divide feature matrix into test and train set,
    y_train = [] #train labels
    test_indices =[0] * len(test_paths)
    train_indices = [] #paths in train

    for n in range(len(path)):
        if path[n] in path_label_dict:
            y_train.append(path_label_dict[path[n]])
            train_indices.append(n)
        elif path[n] in test_paths:
            idx_in_test= np.where(test_paths==path[n])
            test_indices[idx_in_test[0][0]] = n

    X_train = np.take(features,train_indices, axis= 0)
    X_test = np.take(features, test_indices, axis= 0)

    #map each class to a number
    unique_labels = np.unique(y_train)
    unique_label_to_number = {}
    number_to_unique_label = {}
    for i in range(len(unique_labels)):
        unique_label_to_number[unique_labels[i]]= i
        number_to_unique_label[i] = unique_labels[i]
    # convert train labels to their mapped nums
    y_train_num =np.array([unique_label_to_number[lab] for lab in y_train])
    clf = MLP()
    clf.fit(X_train,y_train_num, verbose=True)

    # predict probability

    X_test_torch = torch.zeros(X_test.shape[0], 1, X_test[0].shape[0], X_test[0].shape[1])
    # create pytorch vector for the test data
    for i in range(len(X_test)):
        X_test_torch[i][0] = torch.from_numpy(clf.getPaddedData(X_test[i])).float()

    predictions = clf.predict_proba(X_test_torch)

    prediction_labels=[]
    for p in predictions:
        prediction_labels.append(number_to_unique_label[p])

    res = pd.read_csv("test.csv")
    res['word'] = prediction_labels
    res.to_csv("res.csv", sep=',')

if __name__ == '__main__':
    main()
