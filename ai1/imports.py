# Python libraries for navigation in directories(e.g. iteration there).
from os import listdir
from os.path import join
from datetime import datetime

# Standard libraries for ML, we will use it permanently.
import numpy as np
import pandas as pd
import torch

import torchmetrics.classification.accuracy # It may be bad idea to use this for metrics, but different tests gave me not bad result.
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim

from torchmetrics.classification import MulticlassStatScores
from torchvision.datasets import ImageFolder #M ethod that allow us to use name of directories as labels.
from torchvision.io import read_image
from torch.utils.data import DataLoader
from torchvision.transforms import transforms # I don't like albumentations library because of my classes in university...
from tqdm import tqdm

#Constants that we will use in the next cells.
rootdir = "simpsons_dataset" #This is where dataset located. Change it to the relevant.
rate_learning = 1e-3
epochs = 30  # After 30th epoch we can see the beginning of overfitting at this parameters. I guess there could be a bit more complexity of model than it need.
classnum = 42 # As you will use this dataset for DL, don't forget to delete duplicate of simpson_dataset in simpson_dataset.
bs = 128 # Change this parameter according to hardware.
k_prop = 0.8 # Testset in this dataset sucks.
wd = 1e-3 # Weight decay for weight regularization
classlist = listdir(rootdir)
counter = 20933

dropout_rate = 0.2 #A little bit increase of this probabilty will occur as bad converge
loss_list_train = []
loss_list_test = []

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
