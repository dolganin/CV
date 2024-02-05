from create_objects import model_create
from load_split import load_and_split, dataload
from model import NeuralNetwork
from test import test_model
from train import run_model
from focused_loss import focused_loss
from imports import epochs, output_model_dir, output_graphics_dir
from graphics import loss_graphics, metric_bars
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

        print("Finished training for whole epochs")
    
    print("Started testing model")
    
    accuracy, precision, recall = test_model(model, test_loader, statistic)

    torch.save(model.state_dict(), output_model_dir+"model.pt")
    loss_graphics(output_graphics_dir)
    metric_bars(output_graphics_dir, accuracy, precision, recall)

main()
