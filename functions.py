import matplotlib.pyplot as plt
from os import listdir
from cv2 import imread
from cv2 import dnn
from cv2 import cvtColor
from cv2 import COLOR_BGR2GRAY
import numpy as np
import torch
import torch.nn as func
import torch.optim as optim


rootdir = "simpsons_dataset"

class NeuralNetwork(func.Module):
    def __init__(self):
        self.flatten = func.Flatten()
        self.linear_relu_stack = func.Sequential(
        func.Linear(28*28,256),
        func.ReLU(),
        func.Linear(256,196),
        func.ReLU(),
        func.Linear(196, 42),
        #func.Softmax()
    )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits



def grad(func):
    return None


def get_persons_list():
    pers_dict = dict()
    persons = listdir(rootdir)
    for i in range(len(persons)):
        pers_dict[persons[i]] = i
    return pers_dict


def read_full_set():
    count_elems = 0
    for directory in listdir(rootdir):
        count_elems+= len(listdir(rootdir+'\\'+directory))
    dataset = np.zeros((count_elems, 28, 28))
    labels = np.zeros(count_elems)

    counter = 0
    for directory in listdir(rootdir):
        label = 0
        for image in listdir(rootdir + '\\' + directory):
            rawimage = imread(rootdir + '\\' + directory + '\\' + image)
            rawimage = cvtColor(rawimage, COLOR_BGR2GRAY)
            blob = dnn.blobFromImage(rawimage, 1/255.0, (28,28), swapRB = True, crop = False)
            blob = blob.reshape(1,28,28)
            dataset[counter] = blob
            labels[counter] = label
            counter += 1
            label +=1
    return dataset, labels, count_elems


def splitting(data, label, cnt):
    test_length = cnt-int(0.8*cnt)
    train_length = int(0.8*cnt)
    (x_train, x_test) = torch.utils.data.random_split(data, [train_length, test_length])
    (y_train, y_test) = torch.utils.data.random_split(label, [train_length, test_length])
    return (x_train, y_train), (x_test, y_test)

def model_create():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NeuralNetwork()
    NeuralNetwork = NeuralNetwork.to(device)
    loss = func.CrossEntropyLoss
    optimizer = optim.adam

    return model, optimizer, loss






