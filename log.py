import random

import numpy as np
import matplotlib.pyplot as plt

import torch
from torchvision.utils import make_grid
import cv2

import util
from eval import kitti_depth_eval_utils as kitti_utils
from eval import depth_eval_utils as depth_utils
from eval import of_utils

# Verbosity levels
LOG_MINIMAL = 0
LOG_STANDARD = 1


def save_depth(filename, depth, vmin=None, vmax=None):
    assert len(depth.shape) == 2

    depth = depth.detach().cpu().numpy()
    plt.imsave(filename, 1.0 / depth, vmin=vmin, vmax=vmax, cmap='rainbow')


def save_img(filename, img):
    assert len(img.shape) == 3

    img = img.cpu().numpy().transpose(1, 2, 0)
    img = util.denormalize_cpu(img)
    plt.imsave(filename, img)


def save_gray(filename, img, vmin=None, vmax=None):
    assert len(img.shape) == 2

    img = img.detach().cpu().numpy()
    plt.imsave(filename, img, vmin=vmin, vmax=vmax, cmap='gray')


# TODO: decouple log flow, scene flow and depth in deparated functions.
def log_results(writer, seq_len, results, res, err, min_err, loss_op,
                epoch, log_depth=True, log_flow=True, log_sf=False, data=None):

    bs = results.tgt_imgs_pyr[0].size(0)
    num_snippets = (bs // 2) // (seq_len - 1)
    cols = min(num_snippets, 4)
    step = num_snippets // cols

    # if backward direction has t->t+1 flows
    # idx = list(range(bs//2, bs, step * (seq_len - 1)))
    # if forward has t->t+1 flows
    idx = list(range(0, bs // 2, step * (seq_len - 1)))

    imgs = util.denormalize(results.tgt_imgs_pyr[0][idx, 0])
    imgs_grid = make_grid(imgs)

    _, _, h, w = imgs.size()

    rigid_reps = results.rigid_reps

    rec_t = torch.transpose(rigid_reps.recs_pyr[0].cpu(), 0, 1)
    mask_t = torch.transpose(rigid_reps.masks_pyr[0].cpu(), 0, 1)

    writer.add_image('tgt_imgs', imgs_grid, epoch)

    if log_depth:
        assert len(results.tgt_depths_pyr[0].shape) == 5

        depths = results.tgt_depths_pyr[0][idx, 0]
        depth_colors = util.gray_to_rgb(
            1.0 / depths, 'rainbow')  # 1/depth = disparities
        depths_grid = make_grid(depth_colors)

        rigid_rec = util.denormalize(
            rec_t[:seq_len - 1, idx].reshape(-1, 3, h, w))
        rigid_rec_grid = make_grid(rigid_rec, nrow=len(idx))

        rigid_mask = mask_t[:seq_len - 1, idx].reshape(-1, 1, h, w)
        rigid_mask_grid = make_grid(rigid_mask, nrow=len(idx))

        proj_coords_t = torch.transpose(
            rigid_reps.proj_coords_pyr[0][:, :, :3].cpu(), 0, 1)
        proj_coords_t = proj_coords_t[:seq_len - 1, idx].reshape(-1, 3, h, w)
        sampled_coords_t = torch.transpose(
            rigid_reps.sampled_coords_pyr[0][:, :, :3].cpu(), 0, 1)
        sampled_coords_t = sampled_coords_t[:seq_len -
                                            1, idx].reshape(-1, 3, h, w)

        rflows_color = util.optical_flow_to_rgb(
            results.rigid_flows_pyr[0][idx])
        rflows_grid = make_grid(rflows_color, nrow=cols)

        lb_coords, _ = torch.min(proj_coords_t, dim=0, keepdim=True)
        lb_coords, _ = torch.min(lb_coords, dim=2, keepdim=True)
        lb_coords, _ = torch.min(lb_coords, dim=3, keepdim=True)
        lb_coords2, _ = torch.min(sampled_coords_t, dim=0, keepdim=True)
        lb_coords2, _ = torch.min(lb_coords2, dim=2, keepdim=True)
        lb_coords2, _ = torch.min(lb_coords2, dim=3, keepdim=True)

        ub_coords, _ = torch.max(proj_coords_t, dim=0, keepdim=True)
        ub_coords, _ = torch.max(ub_coords, dim=2, keepdim=True)
        ub_coords, _ = torch.max(ub_coords, dim=3, keepdim=True)
        ub_coords2, _ = torch.max(sampled_coords_t, dim=0, keepdim=True)
        ub_coords2, _ = torch.max(ub_coords2, dim=2, keepdim=True)
        ub_coords2, _ = torch.max(ub_coords2, dim=3, keepdim=True)

        proj_coords_color = util.vect3d_to_rgb(
            proj_coords_t, lb=lb_coords, ub=ub_coords)
        proj_coords_grid = make_grid(proj_coords_color, nrow=cols)

        sampled_coords_color = util.vect3d_to_rgb(
            sampled_coords_t, lb=lb_coords2, ub=ub_coords2)
        sampled_coords_grid = make_grid(sampled_coords_color, nrow=cols)

        writer.add_image('depths_pred', depths_grid, epoch)
        writer.add_image('rigid/proj_coords', proj_coords_grid, epoch)
        writer.add_image('rigid/sampled_coords', sampled_coords_grid, epoch)
        writer.add_image('rigid/rigid_rec', rigid_rec_grid, epoch)
        writer.add_image('rigid/rigid_mask', rigid_mask_grid, epoch)
        writer.add_image('rigid/rigid_flow', rflows_grid, epoch)

        delta = rigid_reps.proj_coords_pyr[0] - \
            rigid_reps.sampled_coords_pyr[0]
        writer.add_histogram(
            'err/x_delta', delta[:, :, 0].cpu().numpy(), epoch)
        writer.add_histogram(
            'err/y_delta', delta[:, :, 1].cpu().numpy(), epoch)
        writer.add_histogram(
            'err/z_delta', delta[:, :, 2].cpu().numpy(), epoch)

        writer.add_histogram(
            'err/x_proj', rigid_reps.proj_coords_pyr[0][:, :, 0].cpu().numpy(), epoch)
        writer.add_histogram(
            'err/y_proj', rigid_reps.proj_coords_pyr[0][:, :, 1].cpu().numpy(), epoch)
        writer.add_histogram(
            'err/z_proj', rigid_reps.proj_coords_pyr[0][:, :, 2].cpu().numpy(), epoch)

        writer.add_histogram(
            'err/x_samp', rigid_reps.sampled_coords_pyr[0][:, :, 0].cpu().numpy(), epoch)
        writer.add_histogram(
            'err/y_samp', rigid_reps.sampled_coords_pyr[0][:, :, 1].cpu().numpy(), epoch)
        writer.add_histogram(
            'err/z_samp', rigid_reps.sampled_coords_pyr[0][:, :, 2].cpu().numpy(), epoch)

    if log_flow:
        assert len(results.flows_pyr[0][idx].shape) == 4

        flows_color = util.optical_flow_to_rgb(results.flows_pyr[0][idx])
        flows_norm = torch.linalg.norm(
            results.flows_pyr[0][idx], ord=2, dim=1, keepdim=True)
        flows_grid = make_grid(flows_color, nrow=cols)

        writer.add_image('flow/optical', flows_grid, epoch)
        writer.add_histogram('flow/opticalflow_norm_hist',
                             flows_norm.cpu().numpy(), epoch)
        writer.add_histogram('flow/opticalflow_vectw_hist',
                             (results.flows_pyr[0][idx])[:, 0].cpu().numpy(), epoch)
        writer.add_histogram('flow/opticalflow_vecth_hist',
                             (results.flows_pyr[0][idx])[:, 1].cpu().numpy(), epoch)
        writer.add_scalar('flow/opticalflow_meanw',
                          (results.flows_pyr[0][idx])[:, 0].mean().item(), epoch)
        writer.add_scalar('flow/opticalflow_meanh',
                          (results.flows_pyr[0][idx])[:, 1].mean().item(), epoch)
        writer.add_scalar('flow/opticalflow_stdw',
                          (results.flows_pyr[0][idx])[:, 0].std().item(), epoch)
        writer.add_scalar('flow/opticalflow_stdh',
                          (results.flows_pyr[0][idx])[:, 1].std().item(), epoch)

        if 'flow' in data.keys():
            _, _, gt_h, gt_w = data['flow'].shape
            flows_gt = of_utils.resize_like_gpu(
                data['flow'][idx], results.flows_pyr[0])
            flows_gt[:, 0, :, :] *= (w - 1) / (gt_w - 1)
            flows_gt[:, 1, :, :] *= (h - 1) / (gt_h - 1)

            flows_gt_color = util.optical_flow_to_rgb(flows_gt)
            flows_gt_grid = make_grid(flows_gt_color, nrow=cols)

            writer.add_image('flow/flows_gt', flows_gt_grid, epoch)

        flow_rec = torch.transpose(
            results.nonrigid_reps.recs_pyr[0].cpu(), 0, 1)
        flow_rec = util.denormalize(
            flow_rec[:seq_len - 1, idx].reshape(-1, 3, h, w))
        flow_rec_grid = make_grid(flow_rec, nrow=cols)

        flow_mask = torch.transpose(
            results.nonrigid_reps.masks_pyr[0].cpu(), 0, 1)
        flow_mask = flow_mask[:seq_len - 1, idx].reshape(-1, 1, h, w)
        flow_mask_grid = make_grid(flow_mask, nrow=cols)

        writer.add_image('flow/flow_rec', flow_rec_grid, epoch)
        writer.add_image('flow/flow_mask', flow_mask_grid, epoch)

        if len(results.rigid_mask_pyr) > 0:
            rigid_mask = torch.transpose(results.rigid_mask_pyr[0].cpu(), 0, 1)
            rigid_mask = rigid_mask[:seq_len - 1, idx].reshape(-1, 1, h, w)
            rigid_mask_grid = make_grid(rigid_mask, nrow=cols)
            writer.add_image('flow/rigid_mask', rigid_mask_grid, epoch)

    if log_sf:
        assert len(results.flows_pyr[0][idx].shape) == 4

        flows_norm = torch.linalg.norm(
            results.flows_pyr[0][idx], ord=2, dim=1, keepdim=True)
        flows_lb, flows_ub = torch.min(flows_norm), torch.max(flows_norm)
        flows_color = util.gray_to_rgb(
            flows_norm, 'coolwarm', lb=flows_lb, ub=flows_ub)
        flows_grid = make_grid(flows_color, nrow=cols)
        writer.add_image('sflow/full', flows_grid, epoch)

        rigid_flows_norm = torch.linalg.norm(
            results.rigid_flows_pyr[0][idx], dim=1, keepdim=True)

        res_flows_norm = torch.linalg.norm(
            results.res_flows_pyr[0][idx], ord=2, dim=1, keepdim=True)
        res_flows_color = util.gray_to_rgb(
            res_flows_norm, 'coolwarm', lb=flows_lb, ub=flows_ub)
        res_flows_grid = make_grid(res_flows_color, nrow=cols)
        writer.add_image('sflow/residual', res_flows_grid, epoch)

        writer.add_histogram(
            'sflow/full_hist', flows_norm.cpu().numpy(), epoch)
        writer.add_histogram('sflow/residual_hist',
                             res_flows_norm.cpu().numpy(), epoch)
        writer.add_histogram('sflow/rigid_hist',
                             rigid_flows_norm.cpu().numpy(), epoch)

    if res is not None:
        res_t = torch.transpose(res[0], 0, 1)
        err_t = torch.transpose(err[0], 0, 1)
        lb_err, ub_err = torch.min(err[0]), torch.max(err[0])

        merged_res = res_t[:seq_len - 1, idx].reshape(-1, 1, h, w)
        merged_res = util.gray_to_rgb(merged_res, 'coolwarm')
        merged_res_grid = make_grid(merged_res, nrow=cols)

        merged_err = err_t[:seq_len - 1, idx].reshape(-1, 1, h, w)
        merged_err = util.gray_to_rgb(
            merged_err, 'coolwarm', lb=lb_err, ub=ub_err)
        merged_err_grid = make_grid(merged_err, nrow=cols)

        writer.add_image('merged/merged_res', merged_res_grid, epoch)
        writer.add_image('merged/merged_err', merged_err_grid, epoch)

        writer.add_histogram('merged/residual', res[0].cpu().numpy(), epoch)
        writer.add_histogram('merged/error', err[0].cpu().numpy(), epoch)

        if min_err is not None:
            # pair at the highest resolution
            min_err_values, min_err_weights = min_err[0]
            min_err_colors = util.gray_to_rgb(
                min_err_values[idx], 'coolwarm', lb=lb_err, ub=ub_err)
            min_err_grid = make_grid(min_err_colors, nrow=cols)

            writer.add_image('merged/min_err', min_err_grid, epoch)
            writer.add_histogram(
                'merged/min_err', min_err_values.cpu().numpy(), epoch)

            if min_err_weights is not None:
                writer.add_histogram(
                    'merged/min_err_weights', min_err_weights.cpu().numpy(), epoch)
                weights_t = torch.transpose(min_err_weights.cpu(), 0, 1)

                rigid_weights = weights_t[:seq_len -
                                          1, idx].reshape(-1, 1, h, w)
                rigid_weights = util.gray_to_rgb(rigid_weights, 'coolwarm')
                rigid_weights = make_grid(rigid_weights, nrow=cols)

                writer.add_image('merged/weights', rigid_weights, epoch)
                writer.add_histogram(
                    'merged/weights', res[0].cpu().numpy(), epoch)


def log_dict(writer, metrics, label, it=None):
    for metric, value in metrics.items():
        writer.add_scalar('{}/{}'.format(label, metric), value, it)


def log_of_metrics_cpu(writer, gt_flows, pred_flows, epoch):
    metrics = oflow_utils.compute_metrics(pred_flows, gt_flows)

    for k, v in metrics.items():
        writer.add_scalar('oflow/' + k, v, epoch)

    return [metrics]


def log_of_metrics(writer, metrics, epoch):
    for k, v in metrics.items():
        writer.add_scalar('oflow/' + k, v, epoch)


def log_model_sanity_checks(writer, model, it):
    writer.add_histogram('dn conv1 weight grads',
                         model.depth_net.enc.net._conv_stem.weight.grad.cpu().numpy(), it)
    writer.add_histogram(
        'dn bn weight', model.depth_net.enc.net._bn0.weight.data.cpu().numpy(), it)
    writer.add_histogram(
        'dn bn weight grad', model.depth_net.enc.net._bn0.weight.grad.cpu().numpy(), it)
    writer.add_histogram(
        'dn bn bias', model.depth_net.enc.net._bn0.bias.data.cpu().numpy(), it)
    writer.add_histogram(
        'dn bn bias grad', model.depth_net.enc.net._bn0.bias.grad.cpu().numpy(), it)
    writer.add_histogram(
        'dn bn rmean', model.depth_net.enc.net._bn0.running_mean.cpu().numpy(), it)
    writer.add_histogram(
        'dn bn rvar', model.depth_net.enc.net._bn0.running_var.cpu().numpy(), it)


def log_params(writer, modules, it):
    for module in modules:
        for name, parameter in module.named_parameters():
            writer.add_histogram(name, parameter.detach().cpu().numpy(), it)
            if parameter.grad is not None:
                writer.add_histogram(
                    name + "_grad", parameter.grad.cpu().numpy(), it)


def print_metric_groups(metric_groups):
    out = ""
    for metrics in metric_groups:
        for k in metrics.keys():
            out += k + " "

        for i, v in enumerate(metrics.values()):
            if i > 0:
                out += ", "
            if isinstance(v, (float, np.float32)):
                out += "{:.4f}".format(v)
            else:
                out += "{}".format(v)

        out += " "

    print(out)
