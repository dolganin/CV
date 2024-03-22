from imports import *
# Standard train for models.
def run_model(model, optim, trainloader, testloader, loss_func):
    batch_number = 0
    train_loss = []
    print(f"Started training at "+ str(datetime.now()))
    for data, labels in tqdm(trainloader):

        model.train()  # Evaluate our model in test mode. In default - model is in this mode.
        data, labels = data.to(device), labels.to(device)  # Copy data to the GPU.
        optim.zero_grad()  # Zero gradients - if  we don't zero him, then our batch size equals num_of_repats_with_no_zero_grad * batch_size.
        # This is one of the tricks in learning to make batch bigger.
        target = model(data)  # Forward the data through network.
        loss = loss_func(target, labels)
        
        loss.backward()
        optim.step()
        
        train_loss.append(loss.item())
    loss_list_train.append(np.array(train_loss).mean())
        
        
    print(f"Finished training at "+ str(datetime.now()))
    print(f"Mean loss at this stage is "+ str((np.array(train_loss).mean())))

    print(f"Started testing at "+str(datetime.now()))
    test_loss = []
    for data, labels in tqdm(testloader):
        model.eval()  # Change model's mode to test. What the difference: in this mode Pytorch don't accumulate gradients in calculation graphs, also BN
        # doesn't accumulate expectation and variance. And dropouts are off.
        
        data, labels = data.to(device), labels.to(device)
        
        target = model(data)

        loss = loss_func(target, labels)
            
        test_loss.append(loss.item())

    loss_list_test.append(np.array(test_loss).mean())
    print(f"Mean loss at this stage is "+ str((np.array(test_loss).mean())))
    print(f"Finished testing at "+ str(datetime.now()))
        
    return None