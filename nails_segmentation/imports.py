# Python libraries for navigation in directories(e.g. iteration there).
import os
import numpy as np
import time
from datetime import datetime

# Standard library for CV, we will use it permanently.
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
from torchvision.io import read_image, ImageReadMode

import segmentation_models_pytorch as smp
from tqdm import tqdm
from yaml_reader import yaml_reader


config = yaml_reader()
sigma = nn.Sigmoid()

labels = config["dataset_parameters"]["labels"] #This is where dataset located. Change it to the relevant.
images = config["dataset_parameters"]["images"]
h = config["dataset_parameters"]["heigth"]
w = config["dataset_parameters"]["width"]
channels = config["dataset_parameters"]["channels"]

rate_learning = config["training_parameters"]["learning_rate"]
epochs = config["training_parameters"]["num_epochs"]  # After 20th epoch we can see the beginning of overfitting at this parameters. I guess there could be a bit more complexity of model than it need.
bs = config["training_parameters"]["batch_size"] # Change this parameter according to hardware.
k_prop = config["training_parameters"]["k_prop"] # Testset in this dataset sucks.
wd = config["model_parameters"]["weight_decay"] # Weight decay for weight regularization


output_model_dir = config["output_parameters"]["out_model_directory"]
output_graphics_dir = config["output_parameters"]["out_graphics_directory"]
output_inference_dir = config["output_parameters"]["out_inference_directory"]
parts = config["output_parameters"]["part_of_partitions"]

loss_list_train = []
loss_list_test = []

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'