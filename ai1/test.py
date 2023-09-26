from imports import *
# Calculate TP, TN, FN, FP for every class through MultiClassStatScores
def test(model, testloader, stats):
    for data, labels in testloader:
        model.eval()
        
        data, labels = data.to(device), labels.to(device)
        
        target = model(data)
        
        stats.update(target, labels) # This method calculates confusion matrix.
    tp, fp, tn, fn = stats._final_state() # And then we extract values from it.
    tp, fp, tn, fn = tp.cpu(), fp.cpu(), tn.cpu(), fn.cpu()
    acc, precision, recall  = torch.nan_to_num(torch.div(tp+tn, tp+fp+tn+fn)), torch.nan_to_num(torch.div(tp, tp+fp)), torch.nan_to_num(torch.div(tp, tp+fn))
    
    return acc, precision, recall