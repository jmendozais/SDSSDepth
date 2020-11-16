import cv2
import numpy as np

def resize_like(x, y):
    x_resized = []

    for i in range(len(x)):
        x_resized.append(
            cv2.resize(x[i], 
            (y[i].shape[1], y[i].shape[0]), 
            interpolation=cv2.INTER_LINEAR))

    return x_resized

def predict(test_loader, model):
    raise NotImplementedError

def save_optical_flows(preds, paths, out_dir):
    raise NotImplementedError

def compute_metrics(pred_ofs, gt_ofs):
    '''
    Args:
        pred_ofs:
        gt_ofs:
    '''
    assert pred_ofs.shape[0] == gt_ofs.shape[0]
    
    pred_ofs = np.transpose (pred_ofs, (0, 2, 3, 1))
    gt_ofs = np.transpose (gt_ofs, (0, 2, 3, 1))

    pred_ofs_resized = resize_like(pred_ofs, gt_ofs)
    pred_ofs_resized = np.array(pred_ofs_resized)

    end_point_error = np.linalg.norm(pred_ofs_resized - gt_ofs, axis=-1)
    gt_ofs_norm = np.linalg.norm(gt_ofs, axis=-1)

    error_rate = np.logical_and(end_point_error > 3.0, end_point_error > 0.05 * gt_ofs_norm).astype(np.float32)

    metrics = {'EPE' : np.mean(end_point_error), 'ER' : np.mean(error_rate)}
    return metrics