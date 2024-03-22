from imports import *
from collections import Counter
#Summary count of images in dataset is 20933.
#Splitting dataset in standard proportion - we take 16k images for train and 4k images for test. It is enough for our purpose.
def create_transformer():
    train_transformer = transforms.Compose([
                                  transforms.ToTensor(),
                                  #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                  transforms.Normalize(mean=[0.5], std=[0.25]),
                                  transforms.ConvertImageDtype(dtype=torch.float32),
                                  transforms.Resize([150, 150], antialias=True),
                                  transforms.RandomHorizontalFlip(p=0.5),
                                  transforms.RandomRotation(15),
                                  transforms.RandomResizedCrop(size=(80, 80), antialias=True),
                                  transforms.RandomChannelPermutation()]
                                )
    test_transformer = transforms.Compose([
                              transforms.ToTensor(),
                              #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                              transforms.Normalize(mean=[0.5], std=[0.25]),
                              transforms.ConvertImageDtype(dtype=torch.float32),
                              transforms.Resize([100, 100], antialias=True)]
                            )
    
    return train_transformer, test_transformer


def load_and_split():
    train_transformer, test_transformer = create_transformer()
    train = ImageFolder(traindir, transform = train_transformer)
    test = ImageFolder(testdir, transform = test_transformer)
    return train, test 

#DataLoader from torch with shuffle. We recieve batch of images with size of variable bs. These will increase rate of model's converge.
def dataload(train, test):
    trainloader = DataLoader(train, batch_size=bs, shuffle=True)  # If shuffle == False, then pictures will go through the pipeline in order. It's bad, when your
    # dataset is sorted.
    testloader = DataLoader(test, batch_size=bs, shuffle=True)
    return trainloader, testloader