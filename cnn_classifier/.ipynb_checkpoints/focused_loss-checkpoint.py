from imports import *

def focused_loss():
    return torch.Tensor([1-(len(listdir(join(rootdir, el))))/counter for el in classlist]).to(device)