from imports import *
def calculate_iou(y_true, y_pred):
    """
    Calculate Intersection over Union (IoU) for binary segmentation.

    Parameters:
    y_true (numpy.ndarray): Ground truth segmentation mask.
    y_pred (numpy.ndarray): Predicted segmentation mask.

    Returns:
    float: IoU score.
    """
    
    intersection = np.logical_and(y_true.cpu().detach().numpy(), y_pred.cpu().detach().numpy())
    union = np.logical_or(y_true.cpu().detach().numpy(), y_pred.cpu().detach().numpy())
    wh = np.prod(y_true.shape)
    iou_score = np.exp(np.sum(intersection/wh)) / np.exp(np.sum(union/wh))
    return iou_score

def calculate_dice(y_true, y_pred):
    return 0
        