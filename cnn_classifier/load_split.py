from imports import *
from testset import balanced_train_test_split
from collections import Counter
#Summary count of images in dataset is 20933.
#Splitting dataset in standard proportion - we take 16k images for train and 4k images for test. It is enough for our purpose.
def create_transformer():
    train_transformer = transforms.Compose([
                                  transforms.ToTensor(),
                                  transforms.ConvertImageDtype(dtype=torch.float32),
                                  transforms.Resize([100, 100], antialias=True),
                                  transforms.RandomHorizontalFlip(p=0.7),
                                  transforms.RandomRotation(15),
                                  transforms.Normalize(mean=[0.5], std=[0.25]),
                                  transforms.RandomResizedCrop(size=(80, 80), antialias=True)]
                                )
    test_transformer = transforms.Compose([
                              transforms.ToTensor(),
                              transforms.ConvertImageDtype(dtype=torch.float32),
                              transforms.Resize([100, 100], antialias=True),
                              transforms.Normalize(mean=[0.5], std=[0.25])]
                            )
    
    return train_transformer, test_transformer


def load_and_split():
    data = ImageFolder(rootdir)
    lst = []
    (train, test) = balanced_train_test_split(data, 1-k_prop) # Split the data with the next proportion - 80% of dataset are train, and remaining 20% are test.
    for _, label in test:
        lst.append(label)
    lst = Counter(lst)
    print(lst)
    return train, test 

#DataLoader from torch with shuffle. We recieve batch of images with size of variable bs. These will increase rate of model's converge.
def dataload(train, test):
    train_transformer, test_transformer = create_transformer()
    train.dataset.transform = train_transformer
    test.dataset.transform = test_transformer
    trainloader = DataLoader(train, batch_size=bs, shuffle=True)  # If shuffle == False, then pictures will go through the pipeline in order. It's bad, when your
    # dataset is sorted.
    testloader = DataLoader(test, batch_size=bs, shuffle=True)
    return trainloader, testloader