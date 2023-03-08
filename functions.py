import torch
import matplotlib.pyplot as plt
from os import listdir
from cv2 import imread
from cv2 import dnn


rootdir = "simpsons_dataset"


def grad(func):
    return None


def get_persons_list():
    pers_dict = dict()
    persons = listdir(rootdir)
    for i in range(len(persons)):
        pers_dict[persons[i]] = i
    return pers_dict


def read_train_set():
    dataset=dict()
    for directory in listdir(rootdir):
        lst_1_person = []
        for image in listdir(rootdir+'\\'+directory):
            rawimage = imread(rootdir+'\\'+directory+'\\'+image)
            blob = dnn.blobFromImage(rawimage, 1/255.0, (28,28), swapRB = True, crop = False)
            lst_1_person.append(rawimage)
        dataset[directory] = lst_1_person
    return dataset

