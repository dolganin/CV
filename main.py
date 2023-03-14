import torch
from functions import  splitting, model_create, predict, Data, dataload, train_step
from torchvision.transforms import transforms

transformer = transforms.Compose([transforms.ConvertImageDtype(dtype=torch.float32),transforms.Resize([28,28]), transforms.Normalize((0,0,0),(1,1,1)), transforms.Grayscale(num_output_channels=1)])

simpsons = Data('labels.csv', 'simpsons_data', transform= transformer)

(train, test) = splitting(simpsons)

(train_loader, test_loader) = dataload(train, test)

model, optimizer, loss =  model_create()

(model, train_loss) = train_step(model, optimizer, train_loader, loss)







