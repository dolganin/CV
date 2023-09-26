from imports import *

def focused_loss():
    class_weights = []
    for el in classlist:
        class_weights.append(1-(len(listdir(join(rootdir, el))))/counter)
    class_weights = torch.Tensor(class_weights).to(device)
    return class_weights