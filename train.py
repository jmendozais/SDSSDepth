''' Validation. We perform validation without looking to the depth results for simplicity. We use validation loss for model selection. 

TODO: After obtaining results we can implement a validation procedure with depth maps to check if the validation loss have the same behaviour as the depth metrics with video wise depth map normalization.
'''
import os
import time
import configargparse
#import argparse
import random
import math

import numpy as np

import torch
from torch import nn, optim
import dill
from tensorboardX import SummaryWriter

import model
import loss
from data import *
from loss import *
from log import *
import util

from pytorch3d import transforms as transforms3D

def print_epoch_stats(epoch, train_loss, val_loss, metric_groups):
    #out = "Ep {}, tr loss {:.4f}, val loss {:.4f}, ".format(epoch, train_loss, val_loss)
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

def load_model(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    return checkpoint['model']

def save_checkpoint(model, optimizer, epoch, save_chkp=True, save_best=False):
    checkpoint = {}
    checkpoint['model'] = model
    checkpoint['optimizer'] = optimizer
    checkpoint['epoch'] = epoch

    if save_chkp:
        torch.save(checkpoint, os.path.join(args.log, 'checkpoint-{}.tar'.format(epoch)), pickle_module=dill)
    if save_best:
        torch.save(checkpoint, os.path.join(args.log, 'best_model_val.tar'), pickle_module=dill)

def compute_loss(model, data, results, rec_mode, rep_cons, num_scales,
    weight_ds, ds_at_level, weight_ofs, ofs_sm_alpha,
    weight_dc, weight_fc, weight_sc, softmin_beta, norm_op, 
    weight_ec, ec_mode, weight_pl, 
    log_misc, log_depth, log_flow, 
    it, epoch, writer):

    if rep_cons:
        rec_loss, res, err, pooled_err = loss.representation_consistency(results, weight_dc, weight_fc, weight_sc, softmin_beta, norm_op, rec_mode=rec_mode, return_residuals=True)
        if log_depth or log_flow:
            log_results(writer, seq_len, results, res, err, pooled_err, norm_op, epoch=epoch, log_depth=log_depth, log_flow=log_flow)
    else:
        rec_loss, rec_terms = loss.baseline_consistency(results, weight_dc, weight_fc, weight_sc, color_nonsmoothness=softmin_beta, rec_mode=rec_mode)
        if log_depth or log_flow:
            log_results(writer, seq_len, results, res=None, err=None, min_err=None, loss_op=norm_op, epoch=epoch, log_depth=log_depth, log_flow=log_flow)


    for j in range(num_scales):
        _, _, c, h, w = data[j].size()
        data[j] = data[j].view(-1, c, h, w)

    batch_loss = rec_loss

    ds_loss = torch.zeros((1,), dtype=torch.float)
    if args.weight_ds > 0:
        ds_loss = loss.disp_smoothness(results.disps_pyr, data, args.ds_at_level, num_scales, order=2)
        ds_loss *= args.weight_ds 
        batch_loss += ds_loss

    ofs_loss = torch.zeros((1,), dtype=torch.float)
    if args.weight_ofs > 0:
        ofs_loss = loss.flow_smoothness(results.ofs_pyr, results.tgt_imgs_pyr, num_scales, order=1, alpha=ofs_sm_alpha)
        ofs_loss *= args.weight_ofs 
        batch_loss += ofs_loss

    ec_loss = torch.zeros((1,), dtype=torch.float)
    if args.weight_ec > 0:
        coords = [apply_flow.coords for apply_flow in model.ms_applyflow]
        ec_loss = loss.epipolar_constraint(coords, results, ec_mode)
        ec_loss *= args.weight_ec
        batch_loss += ec_loss

    percept_loss = torch.zeros((1,), dtype=torch.float)
    if args.weight_pl > 0:
        percept_loss = loss.LPIPS_loss(results.gt_imgs_pyr, results.recs_pyr)
        percept_loss *= args.weight_pl
        batch_loss += percept_loss

    if log_misc:
        writer.add_scalars('loss/batch', {'rec':rec_loss.item(), 
                                            'ds': ds_loss.item(),
                                            'ofs': ofs_loss.item(),
                                            'ec': ec_loss.item(),
                                            'pl': percept_loss.item(),
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

    return batch_loss

if __name__ == '__main__':
    random.seed(101)
    torch.backends.cudnn.benchmark = True # faster with fixed size inputs

    # Create data loader for training set 
    #parser = argparse.ArgumentParser()
    parser = configargparse.ArgParser()

    ''' TODO 
    parser.add_argument('-s', '--seq_len', type=int, default=3)
    parser.add_argument('-h', '--height', type=int, default=128)
    parser.add_argument('-w', '--width' type=int, default=416)
    '''
    parser.add_argument('--config-file', is_config_file=True)

    parser.add_argument('--dataset', default='kitti', choices=['kitti', 'sintel'])
    parser.add_argument('--dataset-dir', default='/data/ra153646/datasets/KITTI/raw_data')
    parser.add_argument('--train-file', default='/home/phd/ra153646/robustness/robustdepthflow/data/kitti/train.txt')
    parser.add_argument('--val-file', default='/home/phd/ra153646/robustness/robustdepthflow/data/kitti/val.txt')
    parser.add_argument('--test-file', default=None) # just for compatibility with the config file

    parser.add_argument('-b', '--batch-size', type=int, default=12)
    parser.add_argument('-l', '--learning-rate', type=float, default=5e-5)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--dropout', type=float, default=1e-2)

    parser.add_argument('--depth-backbone', type=str, default='resnet')
    parser.add_argument('--flow-backbone', type=str, default='resnet')

    parser.add_argument('--pred-disp', action='store_true')
    parser.add_argument('--weight-ds', type=float, default=1e-2)
    parser.add_argument('--ds-at-level', type=int, default=-1) # -1 = all levels

    parser.add_argument('--multi-flow', action='store_true') #multi-frame
    parser.add_argument('--stack-flows', action='store_true') #stack motion ins
    parser.add_argument('--flow-sm-alpha', type=int, default=1)
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
    parser.add_argument('--verbose', type=int, default=LOG_STANDARD)
    parser.add_argument('--load-model', type=str, default=None)

    parser.add_argument('--rec-mode', type=str, default='joint', choices=['joint', 'depth', 'flow'])
    parser.add_argument('--learn-intrinsics', action='store_true')
    parser.add_argument('--rep-cons', action='store_true')
    parser.add_argument('--softmin-beta', type=float, default='inf')
    parser.add_argument('--bidirectional', action='store_true')

    # Architecture config
    parser.add_argument('--norm', default='bn')

    parser.add_argument('--loss', default='l1') # l1, student, charbonnier, cauchy, general adaptive
    parser.add_argument('--loss-params-type', default='none') # 'none', 'var', 'func'
    parser.add_argument('--loss-params-lb', type=float, default=1e-5) # none, 'var', 'func'

    parser.add_argument('--debug-model', action='store_true') # debug in/out using hooks
    parser.add_argument('--debug-training', action='store_true') # debug trainig process with a few iterations
    # TODO:Enable this for laplacian experiments
    parser.add_argument('--debug-params', action='store_true') # log params and gradients
    parser.add_argument('--debug-step', type=int, default=500)

    # Testing improvements
    parser.add_argument('--upscale-pred', action='store_true')
    parser.add_argument('--larger-pose', action='store_true')
    parser.add_argument('--loss-noaug', action='store_true')

    args = parser.parse_args()

    print("Arguments.")
    print(args)

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

    log_depth = args.rec_mode in ['joint', 'depth'] and args.verbose != LOG_MINIMAL
    log_flow = args.rec_mode in ['joint', 'flow'] and args.verbose != LOG_MINIMAL
   
    os.makedirs(args.log, exist_ok=True)

    #torch.autograd.set_detect_anomaly(True)

    num_scales=4
    seq_len=3
    height=128
    width=416
    num_recs = (seq_len - 1) * (2 if args.rec_mode == 'joint' else 1)

    train_set = create_dataset(args.dataset, args.dataset_dir, args.train_file, height=height, width=width, num_scales=num_scales, seq_len=seq_len, is_training=True)
    val_set = create_dataset(args.dataset, args.dataset_dir, args.val_file, height=height, width=width, 
        num_scales=num_scales, seq_len=seq_len, is_training=False, load_depth=log_depth, load_flow=log_flow)

    train_loader = torch.utils.data.DataLoader(train_set, args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_set, args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True, drop_last=True)

    norm_op = loss.create_norm_op(args.loss, params_type=args.loss_params_type, params_lb=args.loss_params_lb, num_recs=num_recs, height=height, width=width)
    norm_op.to(args.device)

    if args.load_model is not None:
        model = load_model(args.load_model)
    else:
        model = model.Model(args.batch_size, num_scales, seq_len, height, width, 
            multiframe_of=args.multi_flow,
            stack_flows=args.stack_flows,
            num_extra_channels=norm_op.num_pred_params, 
            learn_intrinsics=args.learn_intrinsics, 

            norm=args.norm, debug=args.debug_model, 
            depth_backbone=args.depth_backbone, flow_backbone=args.flow_backbone,
            dropout=args.dropout, 
            loss_noaug=args.loss_noaug, larger_pose=args.larger_pose,
            pred_disp=args.pred_disp)

    model = model.to(args.device)

    optimizer = optim.Adam(list(model.parameters()) + list(norm_op.parameters()) , lr=args.learning_rate)

    writer = SummaryWriter(os.path.join(args.log, 'tb_log'))
    start = time.perf_counter()

    it = 0

    start_training = time.perf_counter()
    start = start_training
    val_gt_depths = []
    val_gt_flows = []
    for epoch in range(1, args.epochs + 1):
        start_epoch = time.perf_counter()

        model.train()
        train_loss = 0
        for i, data_pair in enumerate(train_loader, 0):
            #print("ep", epoch, "it", i)
            data, data_noaug = data_pair

            for j in range(num_scales):
                data[j] = data[j].to(args.device)
                data_noaug[j] = data_noaug[j].to(args.device)

            optimizer.zero_grad()

            results = model(data, data_noaug)

            batch_loss = compute_loss(model, data, results, args.rec_mode, args.rep_cons, num_scales,
                args.weight_ds, args.ds_at_level, args.weight_ofs, args.flow_sm_alpha,
                args.weight_dc, args.weight_fc, args.weight_sc, args.softmin_beta, norm_op, 
                args.weight_ec, args.ec_mode, args.weight_pl, 
                log_misc=True, log_depth=False, log_flow=False, 
                it=it, epoch=epoch, writer=writer)

            batch_loss.backward()

            #log_model_sanity_checks(writer, model)

            optimizer.step()
            train_loss += batch_loss.item()

            # Log iter stats
            # Log loss parameters

            if args.loss_params_type != loss.PARAMS_NONE:
                params_mean = 0
                for s in range(num_scales):
                    if args.loss_params_type == loss.PARAMS_PREDICTED:
                        params_mean += torch.mean(results.extra_out_pyr[s]).item()
                    elif args.loss_params_type == loss.PARAMS_VARIABLE:
                        params_mean += torch.mean(norm_op.params_pyr[s]).item()

                params_mean /= num_scales
                writer.add_scalars('loss/batch/params', {'mean': util.cpu_softplus(params_mean)}, it)

            mem = {}

            for j in range(torch.cuda.device_count()):
                mem['alloc-cuda:' + str(j)] = torch.cuda.memory_allocated(j)
                mem['reserv-cuda:' + str(j)] = torch.cuda.memory_reserved(j)

            writer.add_scalars('mem', mem, it)

            if args.debug_params and it % args.debug_step == 0:
                log_params(writer, [model, norm_op], it)


            if args.debug_training and i > 6:
                it += 1
                break

            it += 1

            del batch_loss
            del results

        train_loss /= i

        model.eval()

        if args.debug_training:
            log_idx = random.randint(0, 6)
        else:
            log_idx = random.randint(0, len(val_loader) - 1)

        val_loss = 0
        best_val_loss = 1e10

        val_pred_depths = []
        val_pred_flows = []

        for i, data_pair in enumerate(val_loader, 0):
            #print('load val', time.perf_counter() - start)
            #start = time.perf_counter()
            data, data_noaug = data_pair

            with torch.no_grad():
                
                for j in range(num_scales):
                    data[j] = data[j].to(args.device)
                    data_noaug[j] = data_noaug[j].to(args.device)

                results = model(data, data_noaug)

                results.tgt_depths_pyr = []
                for j in range(num_scales):
                    tmp = [results.depths_pyr[j][k*seq_len] for k in range(args.batch_size)]
                    results.tgt_depths_pyr.append(torch.stack(tmp))

                log_flow_results = args.rec_mode in ['joint', 'flow']
                log_depth_results = args.rec_mode in ['joint', 'depth']

                batch_loss = compute_loss(model, data, results, args.rec_mode, args.rep_cons, num_scales,
                    args.weight_ds, args.ds_at_level, args.weight_ofs, args.flow_sm_alpha, 
                    args.weight_dc, args.weight_fc, args.weight_sc, args.softmin_beta, norm_op, 
                    args.weight_ec, args.ec_mode, args.weight_pl, 
                    log_misc=False, log_depth=log_depth_results and i == log_idx, log_flow=log_flow_results and i == log_idx,
                    it=it, epoch=epoch, writer=writer)

                # Keeping values for logging at epoch level

                val_loss += batch_loss.item()

                if log_depth: 
                    if epoch == 1:
                        val_gt_depths.append(data['depth']) # dict[-1] = depth

                    val_pred_depths.append(results.tgt_depths_pyr[0])
                 
                if log_flow:
                    if epoch == 1:
                        val_gt_flows.append(data['flow'])

                    idx_fw_flow = [int((seq_len - 1)/2 + (seq_len - 1) * i) for i in range(args.batch_size)] # just the forward flows
                    val_pred_flows.append(results.ofs_pyr[0][idx_fw_flow])

            #print('ops val', time.perf_counter() - start)
            #start = time.perf_counter()

            if args.debug_training and i > 6:
                break
        
        # Log the train and val losses

        val_loss /= i
        writer.add_scalars('loss', {'train':train_loss, 'val':val_loss}, epoch)

        # Log depth and optical flow metrics

        #TODO: fix val for batch size
        metrics = [{"Ep": epoch}, {"tr loss": train_loss, "val loss": val_loss}]
        if log_depth:
            if epoch == 1:
                val_gt_depths = torch.cat(val_gt_depths, dim=0).squeeze(1)

            val_pred_depths = torch.cat(val_pred_depths, dim=0).squeeze(1)
            metrics += log_depth_metrics(writer, val_gt_depths.numpy(), val_pred_depths.cpu().numpy(), epoch=epoch)
            
        if log_flow:
            if epoch == 1:
                val_gt_flows = torch.cat(val_gt_flows, dim=0)#.squeeze(1)

            val_pred_flows = torch.cat(val_pred_flows, dim=0)#.squeeze(1)
            metrics += log_oflow_metrics(writer, val_gt_flows.numpy(), val_pred_flows.cpu().numpy(), epoch=epoch)
        
        # Save checkpoint

        has_improved = val_loss < best_val_loss - 1e-6
        save_chkp = epoch % args.ckp_freq == 0

        if save_chkp or has_improved:
            save_checkpoint(model, optimizer, epoch, save_chkp, has_improved)
       
        elapsed_time = time.perf_counter() - start_training 
        epoch_time = time.perf_counter() - start_epoch
        avg_epoch_time = elapsed_time // epoch 
        remaining_time = (args.epochs - epoch) * avg_epoch_time
        expected_training_time = avg_epoch_time * args.epochs

        time_metrics = {'epoch' : util.human_time(epoch_time), 'elap':util.human_time(elapsed_time), 'rem': util.human_time(remaining_time), 'tot':util.human_time(expected_training_time)}
        metrics.append(time_metrics)

        print_epoch_stats(epoch, train_loss, val_loss, metrics)

    writer.close()
        
    #val_metrics = compute_val_metrics(model, val_data)
    #print("Epoch {}: train loss {}, val loss {}".format(epoch, train_loss, val_metrics['loss']))
    
    print("training time", time.perf_counter() - start_training)
