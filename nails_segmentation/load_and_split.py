from imports import *
#splitting
def transformers():
    train_transform = transforms.Compose([
    transforms.ConvertImageDtype(dtype=torch.float32),
    transforms.Resize([h, w]),
    transforms.Normalize(mean=[0.5], std=[0.25])
    ])
    
    test_transform = transforms.Compose([
    transforms.ConvertImageDtype(dtype=torch.float32),
    transforms.Resize([h, w]),
    transforms.Normalize(mean=[0.5], std=[0.25])
    ])
    return train_transform, test_transform


def splitting(data):
    (train, test) = torch.utils.data.random_split(data, [1-k_prop, k_prop])
    return train, test


# dataload function
def dataload(train, test):
    train_loader = DataLoader(train, batch_size=bs, shuffle=True)
    test_loader = DataLoader(test, batch_size=bs, shuffle=True )
    return train_loader, test_loader