'''
Validation. We perform validation without looking to the depth results for simplicity. We use validation loss for model selection. 

TODO: After obtaining results we can implement a validation procedure with depth maps to check if the validation loss have the same behaviour as the depth metrics with video wise depth map normalization.
'''

import torch
from torch import nn, optim
from tensorboardX import SummaryWriter
import time
import argparse

import model
import loss
from data import *

def snippet_from_target_imgs(imgs, seq_len):
    ''' Repeats each target imgs in a batch num_sources times'''
    gt = []
    num_scales = len(imgs)
    for i in range(num_scales):
        b, c, h, w = tgt_imgs[i].size()
        imgs = torch.unsqueeze(tgt_imgs[i], 1)
        imgs = imgs.expand(b, seq_len - 1, 3, h, w)
        imgs = torch.cat([imgs, imgs], axis=1) # duplicate for flow and rigid
        gt.append(imgs)
    return gt

if __name__ == '__main__':
    # Create data loader for training set 
    parser = argparse.ArgumentParser()

    parser.add_argument('--weight-ds', type=float, default=1e-2)
    parser.add_argument('--weight-ofs', type=float, default=1e-3)
    parser.add_argument('--log-dir', type=str, default=None)

    args = parser.parse_args()

    torch.autograd.set_detect_anomaly(True)

    num_epochs=10
    batch_size=2
    num_scales=4
    seq_len=3
    height=64#128
    width=192#416

    train_set = Dataset('./data/moving_exp/sample/ytwalking_frames', './data/moving_exp/sample/ytwalking_frames/ytwalking.txt', height=height, width=width, num_scales=num_scales, seq_len=seq_len)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)

    val_loaders = []

    val_loader = torch.utils.data.DataLoader(train_set, batch_size, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)

    model = model.Model(batch_size, num_scales, seq_len, height, width)

    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    writer = SummaryWriter(args.log_dir)
    start = time.perf_counter()

    it = 0
    for epoch in range(num_epochs):
        train_loss = 0
        for i, data in enumerate(train_loader, 0):
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
            writer.add_scalars('loss/batch', {'rec':rec_loss.item(), 
                                                'ds': ds_loss.item(),
                                                'ofs': ofs_loss.item(),
                                                'all': batch_loss.item()}, it)

            it += 1

            if i > 2:
                break

        train_loss /= i

        val_loss = 0
        for i, data in enumerate(val_loader, 0):
            with torch.no_grad():
                tgt_imgs, recs, depths, ofs = model(data)

                gt = snippet_from_target_imgs(tgt_imgs, seq_len)
                rec_loss = loss.joint_rec(gt, recs, mode='min')

                weights_x, weights_y = loss.exp_gradient(tgt_imgs)
                ds_loss = loss.depth_smoothness(depths, weights=(weights_x, weights_y), order=2)
                ofs_loss = loss.flow_smoothness(ofs, weights=(weights_x, weights_y), order=1)
                batch_loss = rec_loss + args.weight_ds * ds_loss + args.weight_ofs * ofs_loss

                val_loss += batch_loss.item()

            if i > 2:
                break

        val_loss /= i

        writer.add_scalars('loss', {'train':train_loss, 'val':val_loss}, epoch)
        print("epoch {}, train loss {}, val loss {}".format(epoch, train_loss, val_loss))

    writer.close()
        
        #val_metrics = compute_val_metrics(model, val_data)
        #print("Epoch {}: train loss {}, val loss {}".format(epoch, train_loss, val_metrics['loss']))

    print("training time", time.perf_counter() - start)
