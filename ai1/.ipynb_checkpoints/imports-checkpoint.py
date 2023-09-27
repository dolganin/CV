# Python libraries for navigation in directories(e.g. iteration there).
from os import listdir
from os.path import join
from datetime import datetime

# Standard libraries for ML, we will use it permanently.
import numpy as np
import pandas as pd
import torch

import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim

from torchmetrics.classification import MulticlassStatScores
from torchvision.datasets import ImageFolder #M ethod that allow us to use name of directories as labels.
from torchvision.io import read_image
from torch.utils.data import DataLoader
from torchvision.transforms import transforms # I don't like albumentations library because of my classes in university...
from tqdm import tqdm

from yaml_reader import yaml_reader

#Constants that we will use in the next cells.
config = yaml_reader()

rootdir = config["dataset_parameters"]["rootdir"] #This is where dataset located. Change it to the relevant.
classlist = listdir(rootdir)
classnum = len(classlist) # As you will use this dataset for DL, don't forget to delete duplicate of simpson_dataset in simpson_dataset.

rate_learning = config["training_parameters"]["learning_rate"]
epochs = config["training_parameters"]["num_epochs"]  # After 30th epoch we can see the beginning of overfitting at this parameters. I guess there could be a bit more complexity of model than it need.
bs = config["training_parameters"]["batch_size"] # Change this parameter according to hardware.
k_prop = config["training_parameters"]["k_prop"] # Testset in this dataset sucks.
dropout_rate = config["model_parameters"]["dropout_rate"] #A little bit increase of this probabilty will occur as bad converge
wd = config["model_parameters"]["weight_decay"] # Weight decay for weight regularization

counter = ImageFolder(rootdir).__len__()

output_dir = config["output_parameters"]["out_directory"]

loss_list_train = []
loss_list_test = []

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
