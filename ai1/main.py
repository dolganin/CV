from create_objects import model_create
from load_split import load_and_split, dataload
from loss import calculate_loss
from model import NeuralNetwork
from test import test
from train import run_model

def main():
    train, test = load_and_split()
    
    (train_loader, test_loader) = dataload(train, test)
    
    class_weights = focused_loss()
    
    model, optimizer, loss, statistic = model_create(class_weights)
    
    if __name__ == "__main__":
        for i in range(epochs):
            run_model(model, optimizer, train_loader, test_loader, loss)
    
    accuracy, precision, recall = test(model, test_loader, statistic)
