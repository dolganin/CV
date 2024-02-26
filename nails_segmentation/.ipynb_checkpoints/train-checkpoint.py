from imports import *
def run_model(model, optim, trainloader, testloader, loss_func):
    
    train_loss = []
    print("Started training at "+ str(datetime.now()))
    for data, labels in tqdm(trainloader):
        model.train()  # Evaluate our model in train mode. In default - model is in this mode.
        data, labels = data.to(device), labels.to(device)
        optim.zero_grad()  # Zero gradients - if  we don't zero him, then our batch size equals num_of_repats_with_no_zero_grad * batch_size.
        target = model(data)
        
        loss = loss_func(target, labels)
        
        loss.backward()
        optim.step()
        
        train_loss.append(loss.item())
        
    loss_list_train.append(np.array(train_loss).mean())
    print("Finished training at "+ str(datetime.now()))
    print("Mean loss at this stage is "+ str((np.array(train_loss).mean())))

    
    print("Started testing at "+str(datetime.now()))
    test_loss = []
    
    for data, labels in tqdm(testloader):
        model.eval()
        data, labels = data.to(device), labels.to(device)

        target = model(data)
    
        target = sigma(target)
        
        test_loss.append(loss.item())

    loss_list_test.append(np.array(test_loss).mean())
    print("Mean loss at this stage is "+ str((np.array(test_loss).mean())))
    print("Finished testing at "+ str(datetime.now()))
    

    return None