from datetime import datetime
from os.path import join

import pandas as pd
import torch
import torch.nn as func
import torch.optim as optim
import torchmetrics.classification.accuracy
from torch.utils.data import DataLoader, Dataset
from torchvision.io import read_image

rootdir = "simpsons_dataset"
rate_learning = 1e-4
epochs = 80

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
        #logits = self.out(logits)
        #logits = logits.argmax(1)
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


def splitting(data):
    length = data.__len__()
    test_length = length-16768
    train_length = 16768
    (train, test) = torch.utils.data.random_split(data, [train_length, test_length])
    return (train, test)

def model_create():
    model = NeuralNetwork()
    if torch.cuda.is_available():
        model = model.cuda()
    loss = func.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=rate_learning)
    return model, optimizer, loss

def train_model(model, optim, trainloader, loss_func):
    output = open('log_learning.txt', 'w', encoding="utf-8")
    if torch.cuda.is_available():
        accuracy = torchmetrics.Accuracy(task="multiclass", num_classes= 42, k_func= 1).to('cuda')
    else:
        accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=42, k_func=1)
    for e in range(epochs):
        train_loss = 0.0
        for data, labels in trainloader:
            if torch.cuda.is_available():
                data, labels = data.cuda(), labels.cuda()
            optim.zero_grad()
            target = model(data)
            loss = loss_func(target, labels)
            loss.backward()
            optim.step()
            temp = loss.item()
            train_loss += temp
            output.write("The current epoch is " +str(e)+"\n"+"The current loss is "+ str(temp)+"\n")
            output.write("Accuracy = {}\n".format(accuracy(target, labels)))
            output.write("The time is "+ str(datetime.now())+'\n\n')
    return model

def test_model(test_set, model):
    output = open('log_testing.txt', 'w', encoding="utf-8")

    if torch.cuda.is_available():
        accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=42, k_func=1).to('cuda')
    else:
        accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=42, k_func=1)

    cnt = 0
    sum = 0
    for data, labels in test_set:
        if torch.cuda.is_available():
            data, labels = data.cuda(), labels.cuda()
        target = model(data)
        acc = accuracy(target, labels)
        sum += acc
        cnt += 1
        output.write("The current accuracy is = {}".format(acc)+'\n')
    output.write("Summary accuracy is = {}".format(str(sum/cnt)))
    return None


    #logit = model(x_train, y_train)
    #pred_probabily = func.Softmax(dim=1)(logit)
    #y_pred = pred_probabily.argmax(1)
    #return y_pred






