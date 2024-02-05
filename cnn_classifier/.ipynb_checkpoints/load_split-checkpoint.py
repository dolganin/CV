from imports import *
#Summary count of images in dataset is 20933.
#Splitting dataset in standard proportion - we take 16k images for train and 4k images for test. It is enough for our purpose.
def create_transformer():
    transformer = transforms.Compose([
                                  transforms.ToTensor(),
                                  transforms.ConvertImageDtype(dtype=torch.float32),
                                  transforms.Resize([100, 100], antialias=True),
                                  transforms.RandomHorizontalFlip(p=0.5),
                                  transforms.RandomRotation(15),
                                  transforms.Normalize(mean=[0.5], std=[0.25])]
                                )
    return transformer


def load_and_split():
    data = ImageFolder(rootdir, transform=create_transformer())
    (train, test) = torch.utils.data.random_split(data, [0.8, 0.2]) # Split the data with the next proportion - 80% of dataset are train, and remaining 20% are test.
    return train, test 

#DataLoader from torch with shuffle. We recieve batch of images with size of variable bs. These will increase rate of model's converge.
def dataload(train, test):
    trainloader = DataLoader(train, batch_size=bs, shuffle=True)  # If shuffle == False, then pictures will go through the pipeline in order. It's bad, when your
    # dataset is sorted.
    testloader = DataLoader(test, batch_size=bs, shuffle=True)
    return trainloader, testloader