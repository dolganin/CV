import matplotlib.pyplot as plt
from os import listdir
from cv2 import imread
from cv2 import dnn
from cv2 import cvtColor
from cv2 import COLOR_BGR2GRAY
import numpy as np
import torch


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
    test_length = cnt-15000
    train_length = 15000
    test_length *= 2
    train_length *= 2
    return torch.utils.data.random_split(data, label, [train_length, test_length])

