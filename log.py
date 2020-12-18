import random
import numpy as np

import torch
from torchvision.utils import make_grid
from tensorboardX import SummaryWriter

import util
import loss
from eval import kitti_depth_eval_utils as kitti_utils
from eval import depth_eval_utils as depth_utils
from eval import of_utils

# Verbosity levels
LOG_MINIMAL = 0
LOG_STANDARD = 1

def log_results(writer, seq_len, results, res, err, min_err, loss_op, epoch, log_depth=True, log_flow=True, data=None):
    batch_size = results.tgt_imgs_pyr[0].size(0)
    cols = min(batch_size, 4)

    imgs = util.denormalize(results.tgt_imgs_pyr[0][:cols])
    imgs_grid = make_grid(imgs)

    _, _, h, w = imgs.size()

    rec_t = torch.transpose(results.recs_pyr[0].cpu(), 0, 1)
    mask_t = torch.transpose(results.mask_pyr[0].cpu(), 0, 1)
    
    writer.add_image('tgt_imgs', imgs_grid, epoch)

    if log_depth:
        depth_colors = util.gray_to_rgb(1.0/results.depths_pyr[0][:cols], 'rainbow') # 1/depth = disparities
        depths_grid = make_grid(depth_colors)

        rigid_rec = util.denormalize(rec_t[:seq_len-1,:cols].reshape(-1, 3, h, w))
        rigid_rec_grid = make_grid(rigid_rec, nrow=cols)

        rigid_mask = mask_t[:seq_len-1,:cols].reshape(-1, 1, h, w)
        rigid_mask_grid = make_grid(rigid_mask, nrow=cols)

        writer.add_image('depths_pred', depths_grid, epoch)
        writer.add_image('rigid_rec', rigid_rec_grid, epoch)
        writer.add_image('rigid_mask', rigid_mask_grid, epoch)

    if log_flow:
        idx = [2*i for i in range(cols)] + [2*i + 1 for i in range(cols)]
        flows_color = util.optical_flow_to_rgb(results.ofs_pyr[0][idx]) 
        flows_grid = make_grid(flows_color, nrow=cols)

        _, _, gt_h, gt_w = data['flow'].shape
        flows_gt = of_utils.resize_like_gpu(data['flow'][:cols], results.ofs_pyr[0])
        flows_gt[:,0,:,:] *= (w-1)/(gt_w-1)
        flows_gt[:,1,:,:] *= (h-1)/(gt_h-1)
        flows_gt_color = util.optical_flow_to_rgb(flows_gt) 
        flows_gt_grid = make_grid(flows_gt_color, nrow=cols)

        flow_rec = util.denormalize(rec_t[seq_len-1:,:cols].reshape(-1, 3, h, w))
        flow_rec_grid = make_grid(flow_rec, nrow=cols) 

        flow_mask = mask_t[seq_len-1:,:cols].reshape(-1, 1, h, w)
        flow_mask_grid = make_grid(flow_mask, nrow=cols)
                 
        writer.add_image('flows', flows_grid, epoch)
        writer.add_image('flows_gt', flows_gt_grid, epoch)
        writer.add_image('flow_rec', flow_rec_grid, epoch)
        writer.add_image('flow_mask', flow_mask_grid, epoch)

    if res != None:
        res_t = torch.transpose(res[0], 0, 1)
        err_t = torch.transpose(err[0], 0, 1)
        lb_err, ub_err = torch.min(err[0]), torch.max(err[0])
        if log_depth:
            rigid_res = res_t[:seq_len-1,:cols].reshape(-1, 1, h, w)
            rigid_res = util.gray_to_rgb(rigid_res, 'coolwarm')
            rigid_res_grid = make_grid(rigid_res, nrow=cols)

            rigid_err = err_t[:seq_len-1,:cols].reshape(-1, 1, h, w)
            rigid_err = util.gray_to_rgb(rigid_err, 'coolwarm', lb=lb_err, ub=ub_err)
            rigid_err_grid = make_grid(rigid_err, nrow=cols)

            writer.add_image('rigid_res', rigid_res_grid, epoch)
            writer.add_image('rigid_err', rigid_err_grid, epoch)

        if log_flow:
            offset = seq_len - 1 if log_depth else 0
            flow_res = res_t[offset:offset + seq_len - 1,:cols].reshape(-1, 1, h, w)

            flow_res = util.gray_to_rgb(flow_res, 'coolwarm')
            flow_res_grid = make_grid(flow_res, nrow=cols)

            flow_err = err_t[offset:offset + seq_len - 1,:cols].reshape(-1, 1, h, w)
            flow_err = util.gray_to_rgb(flow_err, 'coolwarm', lb=lb_err, ub=ub_err)
            flow_err_grid = make_grid(flow_err, nrow=cols)
            
            writer.add_image('flow_res', flow_res_grid, epoch)
            writer.add_image('flow_err', flow_err_grid, epoch)

        writer.add_histogram('err/residual', res[0].cpu().numpy(), epoch)
        writer.add_histogram('err/error', err[0].cpu().numpy(), epoch)

        if min_err is not None:
            min_err_values, min_err_weights = min_err[0] # pair at the highest resolution
            min_err_colors = util.gray_to_rgb(min_err_values[:cols].unsqueeze(1), 'coolwarm', lb=lb_err, ub=ub_err)
            min_err_grid = make_grid(min_err_colors, nrow=cols)

            writer.add_image('min_err', min_err_grid, epoch)
            writer.add_histogram('err/min_err', min_err_values.cpu().numpy(), epoch)
            if min_err_weights is not None:
                writer.add_histogram('err/min_err_weights', min_err_weights.cpu().numpy(), epoch)
                weights_t = torch.transpose(min_err_weights.cpu(), 0, 1)

                if log_depth:
                    rigid_weights = weights_t[:seq_len-1,:cols].reshape(-1, 1, h, w)
                    rigid_weights = util.gray_to_rgb(rigid_weights, 'coolwarm')
                    rigid_weights = make_grid(rigid_weights, nrow=cols)

                    writer.add_image('rigid_weights', rigid_weights, epoch)
                    writer.add_histogram('err/rigid_weights', res[0].cpu().numpy(), epoch)

                if log_flow:
                    flow_weights = weights_t[seq_len-1:,:cols].reshape(-1, 1, h, w)
                    flow_weights = util.gray_to_rgb(flow_weights, 'coolwarm')
                    flow_weights = make_grid(flow_weights, nrow=cols)

                    writer.add_image('flow_weights', flow_weights, epoch)
                    writer.add_histogram('err/flow_weights', err[0].cpu().numpy(), epoch)

        # Log loss params (num_params == 1)
    if loss_op.params_type != loss.PARAMS_NONE:
        assert loss_op.num_params == 1

        if loss_op.params_type == loss.PARAMS_PREDICTED:
            params = results.extra_out_pyr[0]

            params = torch.transpose(params, 0, 1).cpu()
            lb, ub = torch.min(params), torch.max(params)

            if log_depth:
                rigid_params = params[:seq_len-1,:cols].reshape(-1, loss_op.num_params, h, w)
                rigid_params = util.gray_to_rgb(rigid_params, lb=lb, ub=ub)
                rigid_pg = make_grid(rigid_params, nrow=cols)
                writer.add_image('rigid_params', rigid_pg, epoch)

            if log_flow:
                flow_params = params[seq_len-1:,:cols].reshape(-1, loss_op.num_params, h, w)
                flow_params = util.gray_to_rgb(flow_params, lb=lb, ub=ub)
                flow_pg = make_grid(flow_params, nrow=cols)
                writer.add_image('flow_params', flow_pg, epoch)

            writer.add_histogram('err/params', util.cpu_softplus(params.numpy()), epoch)

        elif loss_op.params_type == loss.PARAMS_VARIABLE:

            params = loss_op.params_pyr[0]
            params = torch.transpose(params.cpu(), 0, 1)

            if log_depth:
                rigid_params = params[:seq_len-1,:1].reshape(-1, loss_op.num_params, h, w)
                rigid_params = util.gray_to_rgb(rigid_params)
                rigid_pg = make_grid(rigid_params, nrow=1)
                writer.add_image('rigid_params', rigid_pg, epoch)

            if log_flow:
                flow_params = params[seq_len-1:,:1].reshape(-1, loss_op.num_params, h, w)
                flow_params = util.gray_to_rgb(flow_params)
                flow_pg = make_grid(flow_params, nrow=1)
                writer.add_image('flow_params', flow_pg, epoch)

            writer.add_histogram('err/params', util.cpu_softplus(params.numpy()), epoch)


def log_depth_metrics(writer, gt_depths, pred_depths, min_depth=1e-3, max_depth=80, epoch=None):
    pred_depths = kitti_utils.resize_like(pred_depths, gt_depths)

    scale_factor = depth_utils.compute_scale_factor(pred_depths, gt_depths, min_depth, max_depth)
    metrics_sing_scale = depth_utils.compute_metrics(pred_depths, gt_depths, min_depth, max_depth, scale_factor)

    metrics_amb_scale = depth_utils.compute_metrics(pred_depths, gt_depths, min_depth, max_depth)

    for metric, value in metrics_sing_scale.items():

        writer.add_scalars('depth_metrics/' + metric, {'consistent': metrics_sing_scale[metric],
                                                       'ambiguous': metrics_amb_scale[metric]
                                                       }, 
                                                       epoch)

    writer.add_scalar('depth_scale', scale_factor, epoch)

    idx = random.randint(0, len(gt_depths) - 1)
    mask = np.logical_and(gt_depths[idx] > min_depth, gt_depths[idx] < max_depth)

    writer.add_histogram('gt_depths', gt_depths[idx][mask], epoch, bins=20)
    writer.add_histogram('pred_depths', pred_depths[idx][mask], epoch, bins=20)

    error_metrics1 = {
                     'consistent are': metrics_sing_scale['abs_rel'],
                     'sqre': metrics_sing_scale['sq_rel'],
                     'rmse': metrics_sing_scale['rmse'],
                     'lrmse': metrics_sing_scale['log_rmse']
                    }

    error_metrics2 = {
                     'ambiguous are': metrics_amb_scale['abs_rel'],
                     'sqre': metrics_amb_scale['sq_rel'],
                     'rmse': metrics_amb_scale['rmse'], 
                     'lrmse': metrics_amb_scale['log_rmse']
                    }


    acc_metrics1 = {
                    'a1': metrics_sing_scale['a1'],
                    'a2': metrics_sing_scale['a2'],
                    'a3': metrics_sing_scale['a3']
                  }

    acc_metrics2 = {
                    'a1': metrics_amb_scale['a1'],
                    'a2': metrics_amb_scale['a2'],
                    'a3': metrics_amb_scale['a3']
                  }

    return [error_metrics1, acc_metrics1, error_metrics2, acc_metrics2]


def log_of_metrics_cpu(writer, gt_flows, pred_flows, epoch):
    metrics = oflow_utils.compute_metrics(pred_flows, gt_flows)

    for k, v in metrics.items():
        writer.add_scalar('oflow/' + k, v, epoch)

    return [metrics]

def log_of_metrics(writer, metrics, epoch):
    for k, v in metrics.items():
        writer.add_scalar('oflow/' + k, v, epoch)

def log_model_sanity_checks(writer, model, it):
    writer.add_histogram('dn conv1 weight grads', model.depth_net.enc.net._conv_stem.weight.grad.cpu().numpy(), it)
    writer.add_histogram('dn bn weight', model.depth_net.enc.net._bn0.weight.data.cpu().numpy(), it)
    writer.add_histogram('dn bn weight grad', model.depth_net.enc.net._bn0.weight.grad.cpu().numpy(), it)
    writer.add_histogram('dn bn bias', model.depth_net.enc.net._bn0.bias.data.cpu().numpy(), it)
    writer.add_histogram('dn bn bias grad', model.depth_net.enc.net._bn0.bias.grad.cpu().numpy(), it)
    writer.add_histogram('dn bn rmean', model.depth_net.enc.net._bn0.running_mean.cpu().numpy(), it)
    writer.add_histogram('dn bn rvar', model.depth_net.enc.net._bn0.running_var.cpu().numpy(), it)

def log_params(writer, modules, it):
    for module in modules:
        for name, parameter in module.named_parameters():
            writer.add_histogram(name, parameter.detach().cpu().numpy(), it)            
            if parameter.grad != None:
                writer.add_histogram(name+"_grad", parameter.grad.cpu().numpy(), it)            

def print_metric_groups(metric_groups):
    out = ""
    for metrics in metric_groups:
        for k in metrics.keys():
            out += k + " "
            
        for v in metrics.values():
            #print(k, type(v))
            if isinstance(v, (float, np.float32)):
                out += "{:.4f} ".format(v)
            else:
                out += "{} ".format(v)

    print(out)            

