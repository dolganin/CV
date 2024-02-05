from imports import *
def calculate_loss(target, labels, loss_func, mode="train"):
    """
    Calculate a loss with given loss_func.
    """
    loss = loss_func(target, labels)
    loss_temp = loss
    if mode == "train":
        loss_list_train.append(loss_temp.item())
    else:
        loss_list_test.append(loss_temp.item())
    return loss