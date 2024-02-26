from create_objects import model_create
from load_and_split import  splitting, dataload, transformers
from test import test_model
from train import run_model
from imports import epochs, output_model_dir, output_graphics_dir
from graphics import loss_graphics
from dataset import SegmentationDataset
from imports import torch

def main():
    
    train_transform, test_transform = transformers()
    
    nails = SegmentationDataset(train_transform, test_transform)
    
    (train, test) = splitting(nails)
    
    train_loader, test_loader = dataload(train, test)

    model, optimizer, loss = model_create()
    
    if __name__ == "__main__":
        for i in range(epochs):
            print("Epoch #"+(str(i)))
            run_model(model, optimizer, train_loader, test_loader, loss)
    
        print("Finished training for whole epochs")
    
    print("Started testing model")
    
    test_model(model, test_loader)
    
    torch.save(model.state_dict(), output_model_dir+"model.pt")
    loss_graphics(output_graphics_dir)
    
main()
        