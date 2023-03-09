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
    model =  func.Sequential(
        func.Conv2d(1,1, kernel_size=4),
        func.ReLU(),
        func.Conv2d(1,1, kernel_size= 2),
        func.Linear(196, 42),
        func.Softmax()
    )
    loss = func.CrossEntropyLoss
    optimizer = optim.adam
    return model, optimizer, loss




