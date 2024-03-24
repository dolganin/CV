from imports import *
from metrics import calculate_iou, calculate_dice
def test_model(model, testloader):
    dice_list = []
    iou_list = []
    for data, labels in testloader:
        model.eval()
        
        data, labels = data.to(device), labels.to(device)
        
        target = model(data)
        dice_list.append(calculate_dice(labels, target))
        iou_list.append(calculate_iou(labels, target))         
        
    print(np.array(dice_list))
    print(np.array(iou_list))
    return None