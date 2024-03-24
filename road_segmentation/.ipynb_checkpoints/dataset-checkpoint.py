from imports import *

class SegmentationDataset(Dataset):
    def __init__(self, transform=None, target_transform=None):
        self.images  = os.listdir(images)
        self.labels = os.listdir(labels)
        
        self.transform  = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(images, self.images[index])
        mask_path = os.path.join(labels, self.images[index])
        image = read_image(img_path)
        label = read_image(mask_path, mode=ImageReadMode.GRAY)
        
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
            label = torch.cat([label], dim=0)
        
        return image, label