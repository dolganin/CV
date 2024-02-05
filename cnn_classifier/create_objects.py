from imports import *
from model import NeuralNetwork
def model_create(weights):
    stats = MulticlassStatScores(num_classes=classnum, average=None).to(device)  # Create metrics for every class. 
    # This method creates object that accumulate TP, TN, FN, FP for every class.
    model = NeuralNetwork()  # Create our CNN model.
    model.to(device)  # Transfer it to device.
    loss = nn.CrossEntropyLoss(weight = weights)  #  Why weights? It's solution of some problem. What's the matter - in this dataset we can 
    # see clearly disbalanced class. So, in this case often we can see solution with focused loss. About it you can read below.
    optimizer = optim.AdamW(model.parameters(), lr=rate_learning, weight_decay=wd)
    return model, optimizer, loss, stats