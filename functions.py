import torch
import matplotlib.pyplot as plt
from os import listdir
from cv2 import imread
from cv2 import dnn
from cv2 import cvtColor
from cv2 import COLOR_BGR2GRAY
from tensorflow import keras
import tensorflow as tf
import numpy as np

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
    dataset=[]
    for directory in listdir(rootdir):
        lst_1_person = np.zeros((len(listdir(rootdir+'\\'+directory)), 3, 28, 28))
        i = 0
        for image in listdir(rootdir+'\\'+directory):
            rawimage = imread(rootdir+'\\'+directory+'\\'+image)
            rawimage = cvtColor(rawimage, COLOR_BGR2GRAY)
            blob = dnn.blobFromImage(rawimage, 1/255.0, (28,28), swapRB = True, crop = False)
            blob = blob.reshape(1,28,28)
            lst_1_person[i] = blob
            i += 1
        dataset.append(lst_1_person)
    return dataset




