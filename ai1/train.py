from imports import *
from loss import calculate_loss
# Standard train for models.
def run_model(model, optim, trainloader, testloader, loss_func):
    batch_number = 0 
    print("Started training at "+str(datetime.now()))
    for data, labels in tqdm(trainloader):
        model.train()  # Evaluate our model in test mode. In default - model is in this mode.
        data, labels = data.to(device), labels.to(device)  # Copy data to the GPU.
        optim.zero_grad()  # Zero gradients - if  we don't zero him, then our batch size equals num_of_repats_with_no_zero_grad * batch_size.
        # This is one of the tricks in learning to make batch bigger.
        target = model(data)  # Forward the data through network.
        loss = calculate_loss(target, labels, loss_func)
        loss.backward()
        optim.step()
    print("Finished training at "+str(datetime.now()))

    print("Started testing at "+str(datetime.now()))
    for data, labels in tqdm(testloader):
        model.eval()  # Change model's mode to test. What the difference: in this mode Pytorch don't accumulate gradients in calculation graphs, also BN
        # doesn't accumulate expectation and variance. And dropouts are off.
        
        data, labels = data.to(device), labels.to(device)
        
        target = model(data)

        calculate_loss(target, labels, loss_func, mode="test")
    print("Finished testing at "+str(datetime.now()))
        
    return None