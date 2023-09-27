from create_objects import model_create
from load_split import load_and_split, dataload
from model import NeuralNetwork
from test import test_model
from train import run_model
from focused_loss import focused_loss
from imports import epochs
import torch

def main():
    train, test = load_and_split()
    
    (train_loader, test_loader) = dataload(train, test)
    
    class_weights = focused_loss()
    
    model, optimizer, loss, statistic = model_create(class_weights)
    
    if __name__ == "__main__":
        for i in range(epochs):
            print("Epoch #"+(str(i)))
            run_model(model, optimizer, train_loader, test_loader, loss)
    
    accuracy, precision, recall = test(model, test_loader, statistic)

    torch.save(model.state_dict(), path=output_dir)


main()
