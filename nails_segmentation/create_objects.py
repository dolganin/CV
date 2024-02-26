from imports import *

#create model
def model_create():
    model = smp.Unet(encoder_name='resnet18', in_channels=channels, classes=1).to(device)
    loss = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=rate_learning, weight_decay=wd)
    return model, optimizer, loss