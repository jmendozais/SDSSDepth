import cv2
import numpy as np
import gc

import torch
import torch.nn.functional as F

def resize_like(x, y):
    x_resized = []
    for i in range(len(x)):
        x_resized.append(
            cv2.resize(x[i], 
                (y[i].shape[1], y[i].shape[0]), 
                interpolation=cv2.INTER_LINEAR))

    return x_resized

# TODO change name to resize_like_torch
def resize_like_gpu(x, y):
    _, _, h, w = y.size()
    return F.interpolate(x, (h, w))

def predict(test_loader, model):
    raise NotImplementedError

def save_optical_flows(preds, paths, out_dir):
    raise NotImplementedError

def compute_of_metrics(pred_ofs, gt_ofs):
    '''
    Args: 
        pred_ofs: A batch containing the predicted optical flow of shape [b, c, h, w]
        gt_ofs: A batch containing the GT optical flow. shape [b, c, h, w] 
    '''
    gt_ofs = gt_ofs.to(pred_ofs.device)
    _, _, gt_h, gt_w = gt_ofs.size()
    _, _, pred_h, pred_w = pred_ofs.size()

    pred_ofs_resized = resize_like_gpu(pred_ofs, gt_ofs)

    pred_ofs_resized[:,0,:,:] *= (gt_w-1)/(pred_w-1)
    pred_ofs_resized[:,1,:,:] *= (gt_h-1)/(pred_h-1)

    epe = torch.norm(pred_ofs_resized - gt_ofs, dim=1)
    gt_ofs_norm = torch.norm(gt_ofs, dim=1)
    error_rate = torch.logical_and(epe > 3.0, epe > 0.05 * gt_ofs_norm).float()

    metrics = {'EPE': torch.mean(epe).item(),
                'ER': torch.mean(error_rate).item()}

    return metrics

def accumulate_metrics(metrics, batch_metrics):
    for k, v in batch_metrics.items():
        if k not in metrics.keys():
            metrics[k] = [v]
        else:
            metrics[k].append(v)

# Deprecated
def compute_metrics_cpu(pred_ofs, gt_ofs):
    '''
    Args:
        pred_ofs: List containing the batches. Each batch has a shape [b, c, h, w]
        gt_ofs: Same structure as pref ofs.
    '''
    assert len(pred_ofs) == len(gt_ofs)

    #gt_ofs = np.transpose (gt_ofs, (0, 2, 3, 1))
    #pred_ofs = np.transpose (pred_ofs, (0, 2, 3, 1))

    _, gt_h, gt_w, _ = gt_ofs[0].shape
    _, pred_h, pred_w, _ = pred_ofs[0].shape
    
    end_point_errors = []
    error_rates = []
    for i in range(len(pred_ofs)):
        pred_ofs[i] = np.transpose(pred_ofs[i], (0, 2, 3, 1))

        pred_ofs_resized = resize_like(pred_ofs[i], gt_ofs[i])
        pred_ofs_resized = np.array(pred_ofs_resized)

        pred_ofs_resized[:,:,:,0] *= (gt_w-1)/(pred_w-1)
        pred_ofs_resized[:,:,:,1] *= (gt_h-1)/(pred_h-1)

        epe = np.linalg.norm(pred_ofs_resized - gt_ofs[i], axis=-1)
        gt_ofs_norm = np.linalg.norm(gt_ofs[i], axis=-1)

        error_rates.append(np.logical_and(epe > 3.0, epe > 0.05 * gt_ofs_norm).astype(np.float32))
        end_point_errors.append(epe)

    metrics = {'EPE' : np.mean(end_point_errors), 'ER' : np.mean(error_rates)}
    return metrics
