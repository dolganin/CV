import matplotlib.pyplot as plt
from os import listdir
from os.path import join
import pandas as pd
#from cv2 import imread, dnn, cvtColor, COLOR_BGR2GRAY
#import numpy as np
import torch
import torch.nn as func
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torchvision.io import read_image


rootdir = "simpsons_dataset"
rate_learning = 1e-4
epochs = 10

class NeuralNetwork(func.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = func.Flatten()
        self.linear_relu_stack = func.Sequential(
        func.Linear(28*28, 256),
        func.ReLU(),
        func.Linear(256, 196),
        func.ReLU(),
        func.Linear(196, 42),
    )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

class Data(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


def dataload(train, test):
    trainloader = DataLoader(train, batch_size=16)
    testloader = DataLoader(test, batch_size=16)
    return trainloader, testloader


def grad(func):
    return None

def splitting(data):
    length = data.__len__()
    test_length = length - int(0.8*length)
    train_length = int(0.8*length)
    (train, test) = torch.utils.data.random_split(data, [train_length, test_length])
    return (train, test)

def model_create():
    model = NeuralNetwork()
    if torch.cuda.is_available():
        model = model.cuda()
    loss = func.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=rate_learning)
    return model, optimizer, loss

def train_step(model, optim, trainloader, loss_func):
    for e in range(epochs):
        train_loss = 0.0
        for data, labels in trainloader:
            if torch.cuda.is_available():
                data, labels = data.cuda(), labels.cuda()
            optim.zero_grad()
            target = model(data)
            #pred_probabily = func.Softmax(dim=1)(target)
            #pred = pred_probabily.argmax(1)
            loss = loss_func(target, labels)
            loss.backward()
            optim.step()
            temp = loss.item()
            train_loss += temp
            print("The current epoch is " +str(e)+"\n"+"The current loss is "+ str(temp)+"\n", "The global loss is "+ str(train_loss))
    return model, train_loss

def learning(x_train, y_train, criterion, optim, model):
    for i in range(epochs):
        opt.zero_grad()
        pred = model.forward(x_train)
        loss = criterion(pred, y_train)
        loss.backward()
        optim.step()
        train_loss += loss.item()


def predict(x_test, y_test):
    logit = model(x_train, y_train)
    pred_probabily = func.Softmax(dim=1)(logit)
    y_pred = pred_probabily.argmax(1)
    return y_pred





