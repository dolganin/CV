from imports import *

def focused_loss():
    return torch.Tensor([1-(len(listdir(join(traindir, el))))/counter for el in classlist]).to(device)