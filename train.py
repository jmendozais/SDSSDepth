''' Validation. We perform validation without looking to the depth results for simplicity. We use validation loss for model selection. 

TODO: After obtaining results we can implement a validation procedure with depth maps to check if the validation loss have the same behaviour as the depth metrics with video wise depth map normalization.
'''
import os
import time
import argparse
import random
import math

import numpy as np

import torch
from torch import nn, optim
from torchvision.utils import make_grid
from tensorboardX import SummaryWriter
import dill

import model
import loss
from data import *
import util
from eval import kitti_depth_eval_utils as kitti_utils
from eval import depth_eval_utils as depth_utils

from pytorch3d import transforms as transforms3D

def gt_snippets_from_tgt_imgs(tgt_imgs, seq_len):
    ''' Repeats each target imgs in a batch num_sources times'''
    gt_pyr = []
    num_scales = len(tgt_imgs)

    for i in range(num_scales):
        b, c, h, w = tgt_imgs[i].size()
        imgs = torch.unsqueeze(tgt_imgs[i], 1)
        imgs = imgs.expand(b, seq_len - 1, c, h, w)
        imgs = torch.cat([imgs, imgs], axis=1) # duplicate for flow and rigid
        gt_pyr.append(imgs)

    return gt_pyr

def log_results(writer, results, res, err, min_err, loss_op, epoch):
    batch_size = results.tgt_imgs_pyr[0].size(0)
    cols = min(batch_size, 4)

    imgs = util.denormalize(results.tgt_imgs_pyr[0][:cols])

    _, _, h, w = imgs.size()
    imgs_grid = make_grid(imgs)

    depth_colors = util.gray_to_rgb(1.0/results.depths_pyr[0][:cols], 'rainbow')
    #depth_colors = util.gray_to_rgb(depths[0][:cols], 'rainbow')
    depths_grid = make_grid(depth_colors)

    of_colors = util.optical_flow_to_rgb(results.ofs_pyr[0][:cols*2]) 
    flows_grid = make_grid(of_colors, nrow=cols)

    if res != None:
        res_t = torch.transpose(res[0].cpu(), 0, 1)
        #_, _, h, w = res_t.size()

        rigid_res = res_t[:seq_len-1,:cols].reshape(-1, 1, h, w)
        rigid_res = util.gray_to_rgb(rigid_res, 'coolwarm')
        rigid_res_grid = make_grid(rigid_res, nrow=cols)

        flow_res = res_t[seq_len-1:,:cols].reshape(-1, 1, h, w)
        flow_res = util.gray_to_rgb(flow_res, 'coolwarm')
        flow_res_grid = make_grid(flow_res, nrow=cols)

        err_t = torch.transpose(err[0].cpu(), 0, 1)

        rigid_err = err_t[:seq_len-1,:cols].reshape(-1, 1, h, w)
        rigid_err = util.gray_to_rgb(rigid_err, 'coolwarm')
        rigid_err_grid = make_grid(rigid_err, nrow=cols)

        flow_err = err_t[seq_len-1:,:cols].reshape(-1, 1, h, w)
        flow_err = util.gray_to_rgb(flow_err, 'coolwarm')
        flow_err_grid = make_grid(flow_err, nrow=cols)

        writer.add_histogram('err/residual', res[0].cpu().numpy(), epoch)
        writer.add_histogram('err/error', err[0].cpu().numpy(), epoch)

        writer.add_image('rigid_res', rigid_res_grid, epoch)
        writer.add_image('flow_res', flow_res_grid, epoch)
        writer.add_image('rigid_err', rigid_err_grid, epoch)
        writer.add_image('flow_err', flow_err_grid, epoch)

    rec_t = torch.transpose(results.recs_pyr[0].cpu(), 0, 1)

    rigid_rec = util.denormalize(rec_t[:seq_len-1,:cols].reshape(-1, 3, h, w))
    rigid_rec_grid = make_grid(rigid_rec, nrow=cols)

    flow_rec = util.denormalize(rec_t[seq_len-1:,:cols].reshape(-1, 3, h, w))
    flow_rec_grid = make_grid(flow_rec, nrow=cols)

    if min_err is not None:
        min_err_values, min_err_weights = min_err[0]
        min_err_colors = util.gray_to_rgb(min_err_values[:cols].unsqueeze(1), 'coolwarm')
        min_err_grid = make_grid(min_err_colors, nrow=cols)

        writer.add_image('min_err', min_err_grid, epoch)
        writer.add_histogram('err/min_err', min_err_values.cpu().numpy(), epoch)
        if min_err_weights is not None:
            writer.add_histogram('err/min_err_weights', min_err_weights.cpu().numpy(), epoch)

    writer.add_image('tgt_imgs', imgs_grid, epoch)
    writer.add_image('depths', depths_grid, epoch)
    writer.add_image('flows', flows_grid, epoch)
    writer.add_image('rigid_rec', rigid_rec_grid, epoch)
    writer.add_image('flow_rec', flow_rec_grid, epoch)

    # Log loss params (num_params == 1)
    if loss_op.params_type != loss.PARAMS_NONE:
        assert loss_op.num_params == 1

        if loss_op.params_type == loss.PARAMS_PREDICTED:
            params = results.extra_out_pyr[0]
            params = torch.transpose(params.cpu(), 0, 1)
            assert params.size(0) == 2*(seq_len-1), "params tensor with shape {} does not ave 2*(seq_len-2) param maps for rigid and flow reconstructions".format(params.shape)
            rigid_params = params[:seq_len-1,:cols].reshape(-1, loss_op.num_params, h, w)

            rigid_params = util.gray_to_rgb(rigid_params)
            rigid_pg = make_grid(rigid_params, nrow=cols)
            flow_params = params[seq_len-1:,:cols].reshape(-1, loss_op.num_params, h, w)
            flow_params = util.gray_to_rgb(flow_params)
            flow_pg = make_grid(flow_params, nrow=cols)

            writer.add_image('rigid_params', rigid_pg, epoch)
            writer.add_image('flow_params', flow_pg, epoch)
            writer.add_histogram('err/params', np.log(np.exp(params.numpy()) + 1), epoch)

        elif loss_op.params_type == loss.PARAMS_VARIABLE:
            #params = loss_op.params_pyr[0].cpu()
            #params_rgb = util.gray_to_rgb(params)
            #writer.add_image('params', params_rgb.view(3, h, w), epoch)
            #writer.add_histogram('err/params', np.log(np.exp(params.numpy()) + 1), epoch)

            params = loss_op.params_pyr[0]
            params = torch.transpose(params.cpu(), 0, 1)
            assert params.size(0) == 2*(seq_len-1), "params tensor with shape {} does not ave 2*(seq_len-2) param maps for rigid and flow reconstructions".format(params.shape)
            rigid_params = params[:seq_len-1,:1].reshape(-1, loss_op.num_params, h, w)

            rigid_params = util.gray_to_rgb(rigid_params)
            rigid_pg = make_grid(rigid_params, nrow=1)
            flow_params = params[seq_len-1:,:1].reshape(-1, loss_op.num_params, h, w)
            flow_params = util.gray_to_rgb(flow_params)
            flow_pg = make_grid(flow_params, nrow=1)

            writer.add_image('rigid_params', rigid_pg, epoch)
            writer.add_image('flow_params', flow_pg, epoch)
            writer.add_histogram('err/params', np.log(np.exp(params.numpy()) + 1), epoch)


def log_depth_metrics(writer, gt_depths, pred_depths, min_depth=1e-3, max_depth=80, epoch=None):
    pred_depths = kitti_utils.resize_like(pred_depths, gt_depths)
    scale_factor = depth_utils.compute_scale_factor(pred_depths, gt_depths, min_depth, max_depth)
    metrics = depth_utils.compute_metrics(pred_depths, gt_depths, min_depth, max_depth, scale_factor)
    for metric, value in metrics.items():
        writer.add_scalar('depth_metrics/' + metric, value, epoch)
        writer.add_scalar('depth/scale_factor', scale_factor, epoch)

    idx = random.randint(0, len(gt_depths) - 1)
    mask = np.logical_and(gt_depths[idx] > min_depth, gt_depths[idx] < max_depth)

    writer.add_histogram('gt_depths', gt_depths[idx][mask], epoch, bins=20)
    writer.add_histogram('pred_depths', pred_depths[idx][mask], epoch, bins=20)
    return metrics

def log_model_sanity_checks(writer, model):
    writer.add_histogram('data', data[0].cpu().numpy(), it)
    writer.add_histogram('dn conv1 weight grads', model.depth_net.enc.net._conv_stem.weight.grad.cpu().numpy(), it)
    writer.add_histogram('dn bn weight', model.depth_net.enc.net._bn0.weight.data.cpu().numpy(), it)
    writer.add_histogram('dn bn weight grad', model.depth_net.enc.net._bn0.weight.grad.cpu().numpy(), it)
    writer.add_histogram('dn bn bias', model.depth_net.enc.net._bn0.bias.data.cpu().numpy(), it)
    writer.add_histogram('dn bn bias grad', model.depth_net.enc.net._bn0.bias.grad.cpu().numpy(), it)
    writer.add_histogram('dn bn rmean', model.depth_net.enc.net._bn0.running_mean.cpu().numpy(), it)
    writer.add_histogram('dn bn rvar', model.depth_net.enc.net._bn0.running_var.cpu().numpy(), it)

if __name__ == '__main__':
    random.seed(101)
    torch.backends.cudnn.benchmark = True # faster with fixed size inputs

    # Create data loader for training set 
    parser = argparse.ArgumentParser()

    ''' TODO 
    parser.add_argument('-s', '--seq_len', type=int, default=3)
    parser.add_argument('-h', '--height', type=int, default=128)
    parser.add_argument('-w', '--width' type=int, default=416)
    '''

    parser.add_argument('-d', '--dataset-dir', default='/data/ra153646/datasets/KITTI/raw_data')
    parser.add_argument('-t', '--train-file', default='/home/phd/ra153646/robustness/robustdepthflow/data/kitti/train.txt')
    parser.add_argument('-v', '--val-file', default='//home/phd/ra153646/robustness/robustdepthflow/data/kitti/val.txt')

    parser.add_argument('-b', '--batch-size', type=int, default=12)
    parser.add_argument('-l', '--learning-rate', type=float, default=5e-5)

    parser.add_argument('--epochs', type=int, default=20)

    parser.add_argument('--weight-ds', type=float, default=1e-2)
    parser.add_argument('--weight-ofs', type=float, default=1e-3)
    parser.add_argument('--weight-dc', type=float, default=1e-1)
    parser.add_argument('--weight-fc', type=float, default=1e-1)
    parser.add_argument('--weight-sc', type=float, default=1e-1)
    parser.add_argument('--weight-pl', type=float, default=1e-1)

    parser.add_argument('--weight-ec', type=float, default=1e-1)
    parser.add_argument('--ec-mode', type=str, default='alg', choices=['alg', 'samp'])

    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--ckp-freq', type=int, default=5)
    parser.add_argument('--log', type=str, default='sample-exp')

    parser.add_argument('--flow-ok', action='store_true')
    parser.add_argument('--learn-intrinsics', action='store_true')
    parser.add_argument('--rep-cons', action='store_true')
    parser.add_argument('--softmin-beta', type=float, default='inf')
    parser.add_argument('--norm', default='bn')

    parser.add_argument('--loss', default='l1') # l1, student, charbonnier, cauchy, general adaptive
    parser.add_argument('--loss-params-type', default=None) # none, 'var', 'func'

    args = parser.parse_args()

    if args.loss_params_type == 'var':
        args.loss_params_type = loss.PARAMS_VARIABLE
    elif args.loss_params_type == 'fun':
        args.loss_params_type = loss.PARAMS_PREDICTED
    else:
        args.loss_params_type = loss.PARAMS_NONE

    if args.ec_mode == 'alg':
        args.ec_mode = loss.EPIPOLAR_ALGEBRAIC
    elif args.ec_mode == 'samp':
        args.ec_mode = loss.EPIPOLAR_SAMPSON


    os.makedirs(args.log, exist_ok=True)

    #torch.autograd.set_detect_anomaly(True)

    num_scales=4
    seq_len=3
    height=128
    width=416
    num_recs = (seq_len - 1) * (2 if args.flow_ok else 1)

    train_set = Kitti(args.dataset_dir, args.train_file, height=height, width=width, num_scales=num_scales, seq_len=seq_len, is_training=True)
    val_set = Kitti(args.dataset_dir, args.val_file, height=height, width=width, num_scales=num_scales, seq_len=seq_len, is_training=False, load_depth=True)

    train_loader = torch.utils.data.DataLoader(train_set, args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True, drop_last=True)

    val_loader = torch.utils.data.DataLoader(val_set, args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True, drop_last=True)

    norm_op = loss.create_norm_op(args.loss, params_type=args.loss_params_type, num_recs=num_recs, height=height, width=width)
    norm_op.to(args.device)

    model = model.Model(args.batch_size, num_scales, seq_len, height, width, num_extra_channels=norm_op.num_pred_params, learn_intrinsics=args.learn_intrinsics, norm=args.norm)
    model = model.to(args.device)

    optimizer = optim.Adam(list(model.parameters()) + list(norm_op.parameters()) , lr=args.learning_rate)

    writer = SummaryWriter(os.path.join(args.log, 'tb_log'))
    start = time.perf_counter()

    it = 0

    start_training = time.perf_counter()
    start = start_training
    val_gt_depths = []
    for epoch in range(1, args.epochs + 1):
        start_epoch = time.perf_counter()

        model.train()
        train_loss = 0
        for i, data in enumerate(train_loader, 0):
            #print('load', time.perf_counter() - start)
            #start = time.perf_counter()

            for j in range(num_scales):
                data[j] = data[j].to(args.device)

            optimizer.zero_grad()

            #tgt_imgs, recs, depths, proj_depths, sampled_depths, proj_coords, sampled_coords, feats, sampled_feats, ofs, T, K, inv_K = model(data)
            results = model(data)

            results.gt_imgs_pyr = gt_snippets_from_tgt_imgs(results.tgt_imgs_pyr, seq_len)

            if args.rep_cons:
                rec_loss = loss.representation_consistency(results, args.weight_dc, args.weight_fc, args.weight_sc, args.softmin_beta, norm_op)
            else:
                rec_loss, rec_terms = loss.baseline_consistency(results, args.weight_dc, args.weight_fc, args.weight_sc, color_nonsmoothness=args.softmin_beta, flow_ok=args.flow_ok)

            for j in range(num_scales):
                _, _, c, h, w = data[j].size()
                data[j] = data[j].view(-1, c, h, w)

            ds_loss = loss.depth_smoothness(results.depths_pyr, data, order=2)
            ofs_loss = loss.flow_smoothness(results.ofs_pyr, results.tgt_imgs_pyr, order=1)
            coords = [apply_flow.coords for apply_flow in model.ms_applyflow]
            ec_loss = loss.epipolar_constraint(coords, results, args.ec_mode)

            ds_loss *= args.weight_ds 
            ofs_loss *= args.weight_ofs 
            ec_loss *= args.weight_ec

            if args.weight_pl > 0:
                percept_loss = loss.LPIPS_loss(results.gt_imgs_pyr, results.recs_pyr)
                percept_loss *= args.weight_pl
            else:
                percept_loss = 0

            batch_loss = rec_loss + ds_loss + ofs_loss + ec_loss + percept_loss # HERE! EC LOSS IS DEACTIVATED
            batch_loss.backward()

            #log_model_sanity_checks(writer, model)

            optimizer.step()
            train_loss += batch_loss.item()

            # Log stats
            writer.add_scalars('loss/batch', {'rec':rec_loss.item(), 
                                                'ds': ds_loss.item(),
                                                'ofs': ofs_loss.item(),
                                                'ec': ec_loss.item(),
                                                'all': batch_loss.item()}, it)
            if not args.rep_cons:
                writer.add_scalars('loss/batch', rec_terms, it) 

            for j in range(len(results.K_pyr)):
                writer.add_scalars('intrinsics/{}'.format(j), 
                                                {'fx': results.K_pyr[j][0,0,0].item(),
                                                'fy': results.K_pyr[j][0,1,1].item()}, it)

            A = transforms3D.rotation_conversions.matrix_to_euler_angles(results.T[:,:3,:3], "XYZ")
            writer.add_scalars('pose',
                                                {'e1': A[0,0].item(),
                                                'e2': A[0,1].item(),
                                                'e3': A[0,2].item()}, it)
            
            # Log loss scale parameters
            if args.loss_params_type != loss.PARAMS_NONE:
                params_mean = 0
                for s in range(num_scales):
                    if args.loss_params_type == loss.PARAMS_PREDICTED:
                        params_mean += torch.mean(results.extra_out_pyr[s]).item()
                    elif args.loss_params_type == loss.PARAMS_VARIABLE:
                        params_mean += torch.mean(norm_op.params_pyr[s]).item()
                params_mean /= num_scales
                writer.add_scalars('loss/batch/params', {'mean': math.log(1 + math.exp(params_mean))}, it)

            mem = {}
            for j in range(torch.cuda.device_count()):
                mem['alloc-cuda:' + str(j)] = torch.cuda.memory_allocated(j)
                mem['reserv-cuda:' + str(j)] = torch.cuda.memory_reserved(j)
            writer.add_scalars('mem', mem, it)

            it += 1

            #if i > 6:
                #break

            del batch_loss
            del results

        train_loss /= i

        model.eval()

        #log_idx = len(val_loader) - 1
        #log_idx = random.randint(0, len(val_loader) - 1)
        log_idx = random.randint(0, 6)
        val_loss = 0
        best_val_loss = 1e10
        val_pred_depths = []
        for i, data in enumerate(val_loader, 0):
            #print('load val', time.perf_counter() - start)
            #start = time.perf_counter()
            with torch.no_grad():
                for j in range(num_scales):
                    data[j] = data[j].to(args.device)

                results = model(data)

                results.gt_imgs_pyr = gt_snippets_from_tgt_imgs(results.tgt_imgs_pyr, seq_len)

                results.tgt_depths_pyr = []
                for j in range(num_scales):
                    tmp = [results.depths_pyr[j][k*seq_len] for k in range(args.batch_size)]
                    results.tgt_depths_pyr.append(torch.stack(tmp))

                if i == log_idx:
                    #rec_loss, res, min_res = loss.representation_consistency(gt_imgs, recs, proj_depths, sampled_depths, feats, sampled_feats, mode='min', return_residuals=True)
                    if args.rep_cons:
                        rec_loss, res, err, min_err = loss.representation_consistency(results, args.weight_dc, args.weight_fc, args.weight_sc, args.softmin_beta, norm_op, return_residuals=True)
                        log_results(writer, results, res, err, min_err, norm_op, epoch=epoch)
                    else:
                        rec_loss, rec_terms = loss.baseline_consistency(results, args.weight_dc, args.weight_fc, args.weight_sc, color_nonsmoothness=args.softmin_beta, flow_ok=args.flow_ok)
                        log_results(writer, results, res=None, err=None, min_err=None, loss_op=norm_op, epoch=epoch)
                else:
                    if args.rep_cons:
                        rec_loss = loss.representation_consistency(results, args.weight_dc, args.weight_fc, args.weight_sc, args.softmin_beta, norm_op)
                    else:
                        rec_loss, rec_terms = loss.baseline_consistency(results, args.weight_dc, args.weight_fc, args.weight_sc, color_nonsmoothness=args.softmin_beta, flow_ok=args.flow_ok)

                if epoch == 1:
                    val_gt_depths.append(data[-1]) # dict[-1] = depth

                val_pred_depths.append(results.tgt_depths_pyr[0])
                
                for j in range(num_scales):
                    _, _, c, h, w = data[j].size()
                    data[j] = data[j].view(-1, c, h, w)

                del data[-1] # -1 keeps the groundtruth depth map

                ds_loss = loss.depth_smoothness(results.depths_pyr, data, order=2)
                ofs_loss = loss.flow_smoothness(results.ofs_pyr, results.tgt_imgs_pyr, order=1)
                coords = [apply_flow.coords for apply_flow in model.ms_applyflow]
                ec_loss = loss.epipolar_constraint(coords, results, args.ec_mode)

                ds_loss *= args.weight_ds 
                ofs_loss *= args.weight_ofs 
                ec_loss *= args.weight_ec

                if args.weight_pl > 0:
                    percept_loss = loss.LPIPS_loss(results.gt_imgs_pyr, results.recs_pyr)
                    percept_loss *= args.weight_pl
                else:
                    percept_loss = 0

                batch_loss = rec_loss + ds_loss + ofs_loss + ec_loss + percept_loss
                val_loss += batch_loss.item()

            #print('ops val', time.perf_counter() - start)
            #start = time.perf_counter()

            #if i > 6:
                #break

        val_loss /= i

        writer.add_scalars('loss', {'train':train_loss, 'val':val_loss}, epoch)

        #TODO: fix val for batch size
        if epoch == 1:
            val_gt_depths = torch.cat(val_gt_depths, dim=0).squeeze(1)

        val_pred_depths = torch.cat(val_pred_depths, dim=0).squeeze(1)
        metrics = log_depth_metrics(writer, val_gt_depths.numpy(), val_pred_depths.cpu().numpy(), epoch=epoch)

        has_improved = val_loss < best_val_loss - 1e-6
        save_checkpoint = epoch % args.ckp_freq == 0

        if save_checkpoint or has_improved:
            checkpoint = {}
            checkpoint['model'] = model
            checkpoint['optimizer'] = optimizer
            checkpoint['epoch'] = epoch
            if save_checkpoint:
                torch.save(checkpoint, os.path.join(args.log, 'checkpoint-{}.tar'.format(epoch)), pickle_module=dill)
            if has_improved:
                torch.save(checkpoint, os.path.join(args.log, 'best_model_val.tar'), pickle_module=dill)
        
        elapsed_time = time.perf_counter() - start_training 
        epoch_time = time.perf_counter() - start_epoch
        avg_epoch_time = elapsed_time // epoch 
        expected_training_time = avg_epoch_time * args.epochs
        print("Ep {}, tr loss {:.4f}, val loss {:.4f}, imp? {} are sqre rmse lrmse {:.4f} {:.4f} {:.4f} {:.4f}, a1 a2 a3 {:.4f} {:.4f} {:.4f}, time {} ({}/{})".format(epoch, train_loss, val_loss, has_improved, metrics['abs_rel'], metrics['sq_rel'], metrics['rmse'], metrics['log_rmse'], metrics['a1'], metrics['a2'], metrics['a3'], util.human_time(epoch_time), util.human_time(elapsed_time), util.human_time(expected_training_time)))

    writer.close()
        
        #val_metrics = compute_val_metrics(model, val_data)
        #print("Epoch {}: train loss {}, val loss {}".format(epoch, train_loss, val_metrics['loss']))
    
    print("training time", time.perf_counter() - start_training)
