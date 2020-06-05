'''
Validation. We perform validation without looking to the depth results for simplicity. We use validation loss for model selection. 

TODO: After obtaining results we can implement a validation procedure with depth maps to check if the validation loss have the same behaviour as the depth metrics with video wise depth map normalization.
'''

import os
import time
import argparse

import torch
from torch import nn, optim
from torchvision.utils import make_grid
from tensorboardX import SummaryWriter
import dill

import model
import loss
from data import *
import util

def snippet_from_target_imgs(imgs, seq_len):
    ''' Repeats each target imgs in a batch num_sources times'''
    gt = []
    num_scales = len(imgs)
    for i in range(num_scales):
        b, c, h, w = tgt_imgs[i].size()
        imgs = torch.unsqueeze(tgt_imgs[i], 1)
        imgs = imgs.expand(b, seq_len - 1, c, h, w)
        imgs = torch.cat([imgs, imgs], axis=1) # duplicate for flow and rigid
        gt.append(imgs)
    return gt

if __name__ == '__main__':
    # Create data loader for training set 
    parser = argparse.ArgumentParser()

    parser.add_argument('--weight-ds', type=float, default=1e-2)
    parser.add_argument('--weight-ofs', type=float, default=1e-3)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--ckp-freq', type=int, default=1)
    parser.add_argument('-b', '--batch-size', type=int, default=12)
    parser.add_argument('--log', type=str, default='sample-exp')
    parser.add_argument('--epochs', type=int, default=20)

    args = parser.parse_args()

    os.makedirs(args.log, exist_ok=True)
    #torch.autograd.set_detect_anomaly(True)

    num_scales=4
    seq_len=3
    height=128
    width=416

    train_set = Dataset('/data/ra153646/datasets/KITTI/raw_data', 'data/kitti/train.txt', height=height, width=width, num_scales=num_scales, seq_len=seq_len, is_training=True)
    val_set = Dataset('/data/ra153646/datasets/KITTI/raw_data', 'data/kitti/val.txt', height=height, width=width, num_scales=num_scales, seq_len=seq_len, is_training=False)

    train_loader = torch.utils.data.DataLoader(train_set, args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True, drop_last=True)

    val_loader = torch.utils.data.DataLoader(train_set, args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True, drop_last=True)

    model = model.Model(args.batch_size, num_scales, seq_len, height, width)
    model = model.to(args.device)

    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    writer = SummaryWriter(os.path.join(args.log, 'tb_log'))
    start = time.perf_counter()

    it = 0

    start_training = time.perf_counter()
    start = start_training
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        for i, data in enumerate(train_loader, 0):

            #print('load', time.perf_counter() - start)
            start = time.perf_counter()

            for j in range(len(data)):
                data[j] = data[j].to(args.device)

            optimizer.zero_grad()

            tgt_imgs, recs, depths, ofs = model(data)

            gt = snippet_from_target_imgs(tgt_imgs, seq_len)
            rec_loss = loss.joint_rec(gt, recs, mode='min')

            weights_x, weights_y = loss.exp_gradient(tgt_imgs)
            ds_loss = loss.depth_smoothness(depths, weights=(weights_x, weights_y), order=2)
            ofs_loss = loss.flow_smoothness(ofs, weights=(weights_x, weights_y), order=1)

            ds_loss *= args.weight_ds * ds_loss
            ofs_loss *= args.weight_ofs * ofs_loss
            batch_loss = rec_loss + ds_loss + ofs_loss
            batch_loss.backward()

            optimizer.step()
            train_loss += batch_loss.item()

            # Log stats
            writer.add_scalars('loss/batch', {'rec':rec_loss.item(), 
                                                'ds': ds_loss.item(),
                                                'ofs': ofs_loss.item(),
                                                'all': batch_loss.item()}, it)

            #print('ops', time.perf_counter() - start)
            #start = time.perf_counter()
            mem = {}
            for j in range(torch.cuda.device_count()):
                mem['alloc-cuda:' + str(j)] = torch.cuda.memory_allocated(j)
                mem['reserv-cuda:' + str(j)] = torch.cuda.memory_reserved(j)
            writer.add_scalars('mem', mem, it)

            it += 1
            #if i > 2:
            #    break

        train_loss /= i

        model.eval()
        val_loss = 0
        for i, data in enumerate(val_loader, 0):
            #print('load val', time.perf_counter() - start)
            #start = time.perf_counter()
            with torch.no_grad():
                for j in range(len(data)):
                    data[j] = data[j].to(args.device)

                tgt_imgs, recs, depths, ofs = model(data)
                gt = snippet_from_target_imgs(tgt_imgs, seq_len)

                if i == 0:
                    rec_loss, res, min_res = loss.joint_rec(gt, recs, mode='min', return_residuals=True)
                    cols = min(args.batch_size, 4)
                    imgs_grid = make_grid(tgt_imgs[0][:cols])

                    depth_colors = util.gray_to_rgb(depths[0][:cols], 'rainbow')
                    depths_grid = make_grid(depth_colors)

                    of_colors = util.optical_flow_to_rgb(ofs[0][:cols*2]) 
                    flows_grid = make_grid(of_colors, nrow=cols)

                    

                    res_t = torch.transpose(res[0].cpu(), 0, 1)
                    _, _, h, w = res_t.size()

                    rigid_res = res_t[:seq_len-1,:cols].reshape(-1, 1, h, w)
                    rigid_res = util.gray_to_rgb(rigid_res, 'coolwarm')
                    rigid_res_grid = make_grid(rigid_res, nrow=cols)

                    flow_res = res_t[seq_len-1:,:cols].reshape(-1, 1, h, w)
                    flow_res = util.gray_to_rgb(flow_res, 'coolwarm')
                    flow_res_grid = make_grid(flow_res, nrow=cols)

                    rec_t = torch.transpose(recs[0].cpu(), 0, 1)

                    rigid_rec = rec_t[:seq_len-1,:cols].reshape(-1, 3, h, w)
                    rigid_rec_grid = make_grid(rigid_rec, nrow=cols)

                    flow_rec = rec_t[seq_len-1:,:cols].reshape(-1, 3, h, w)
                    flow_rec_grid = make_grid(flow_rec, nrow=cols)

                    min_res_colors = util.gray_to_rgb(min_res[0][:cols].unsqueeze(1), 'coolwarm')
                    min_res_grid = make_grid(min_res_colors, nrow=cols)

                    writer.add_image('tgt_imgs', imgs_grid, epoch)
                    writer.add_image('depths', depths_grid, epoch)
                    writer.add_image('flows', flows_grid, epoch)
                    writer.add_image('rigid_res', rigid_res_grid, epoch)
                    writer.add_image('flow_res', flow_res_grid, epoch)
                    writer.add_image('rigid_rec', rigid_rec_grid, epoch)
                    writer.add_image('flow_rec', flow_rec_grid, epoch)
                    writer.add_image('min_res', min_res_grid, epoch)

                    writer.add_histogram('residual/res', res[0].cpu().numpy(), epoch)
                    writer.add_histogram('residual/min_res', min_res[0].cpu().numpy(), epoch)
                else:
                    rec_loss = loss.joint_rec(gt, recs, mode='min')

                weights_x, weights_y = loss.exp_gradient(tgt_imgs)
                ds_loss = loss.depth_smoothness(depths, weights=(weights_x, weights_y), order=2)
                ofs_loss = loss.flow_smoothness(ofs, weights=(weights_x, weights_y), order=1)
                batch_loss = rec_loss + args.weight_ds * ds_loss + args.weight_ofs * ofs_loss

                val_loss += batch_loss.item()


            #print('ops val', time.perf_counter() - start)
            #start = time.perf_counter()

            #if i > 2:
            #    break

        val_loss /= i

        writer.add_scalars('loss', {'train':train_loss, 'val':val_loss}, epoch)

        print("epoch {}, train loss {}, val loss {}".format(epoch, train_loss, val_loss))

        if epoch % args.ckp_freq == 0:
            checkpoint = {}
            checkpoint['model'] = model
            checkpoint['optimizer'] = optimizer
            checkpoint['epoch'] = epoch
            torch.save(checkpoint, os.path.join(args.log, 'checkpoint-{}.tar'.format(epoch)), pickle_module=dill)

    writer.close()
        
        #val_metrics = compute_val_metrics(model, val_data)
        #print("Epoch {}: train loss {}, val loss {}".format(epoch, train_loss, val_metrics['loss']))

    print("training time", time.perf_counter() - start_training)
