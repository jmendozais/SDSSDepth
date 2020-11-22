import time
import numpy as np
from absl.testing import absltest

import torch
import torch.nn
import torchvision.transforms.functional as T
from net.base_net import *

from data import Dataset

import loss
import model
import torch.utils.data as data

class DatasetMock(data.Dataset):

    def __init__(self, seq_len, height, width, scales=4):
        super(DatasetMock, self).__init__()
        self.seq_len = seq_len
        self.height = height
        self.width = width
        self.num_scales = 4

        self.size = 32

    def __getitem__(self, idx):
        snippet = torch.zeros((self.seq_len, 3, self.height, self.width))
       
        ms_snippets = {}

        ms_snippets[0] = snippet
        for i in range(1, self.num_scales):
            size = (self.width//(2**i), self.height//(2**i))
            ms_snippets[i] = snippet[:,:,:size[1],:size[0]]

        return ms_snippets

    def __len__(self):

        return self.size


class ModelTest(absltest.TestCase):

    def _get_model_uflow(self, device='cpu', batch_size=2, flow_backbone='uflow_lite'):
        seq_len = 3
        height = 128
        width = 416

        norm_op = loss.create_norm_op(
            name='l1', 
            params_type=2, # Params None
            params_lb=0,
            num_recs=seq_len - 1, 
            height=height, 
            width=width)

        norm_op.to(device)

        model_ = model.Model(
            batch_size=batch_size,
            num_scales=4,
            seq_len=seq_len, 
            height=height, 
            width=width, 
            multiframe_of=False,
            stack_flows=False,
            num_extra_channels=norm_op.num_pred_params, 
            learn_intrinsics=False, 
            norm='bn', 
            debug=False, 
            depth_backbone='resnet', 
            flow_backbone=flow_backbone,
            dropout=0, 
            loss_noaug=False, 
            larger_pose=True,
            pred_disp=True)

        model_ = model_.to(device)

        return model_

    def _get_model_multiframe_stacked(self, device='cpu', batch_size=2):
        seq_len = 3
        height = 128
        width = 416

        norm_op = loss.create_norm_op(
            name='l1', 
            params_type=2, # Params None
            params_lb=0,
            num_recs=seq_len - 1, 
            height=height, 
            width=width)

        norm_op.to(device)

        model_ = model.Model(
            batch_size=batch_size,
            num_scales=4,
            seq_len=seq_len, 
            height=height, 
            width=width, 
            multiframe_of=True,
            stack_flows=True,
            num_extra_channels=norm_op.num_pred_params, 
            learn_intrinsics=False, 
            norm='bn', 
            debug=False, 
            depth_backbone='resnet', 
            flow_backbone='resnet',
            dropout=0, 
            loss_noaug=False, 
            larger_pose=True,
            pred_disp=True)

        model_ = model_.to(device)

        return model_

    def _test_forward(self):#
        model = self._get_model()
        dataset = DatasetMock(model.seq_len, model.height, model.width)
        dataloader = torch.utils.data.DataLoader(dataset, 
            batch_size=model.batch_size, 
            shuffle=True, 
            num_workers=0, 
            pin_memory=True, 
            drop_last=True)
        
        it = iter(dataloader)
        batch = next(it)
           
        _ = model(batch, batch)

    def _mock_batch_processing(self, model):
        dataset = DatasetMock(model.seq_len, model.height, model.width)
        dataloader = torch.utils.data.DataLoader(dataset, 
            batch_size=model.batch_size, 
            shuffle=True, 
            num_workers=0, 
            pin_memory=True, 
            drop_last=True)
        
        it = iter(dataloader)
        batch = next(it)

        start = time.perf_counter()
        # Forward
        for k, v in batch.items():
            batch[k] = v.to('cuda')
        model = model.to('cuda')
        res = model(batch, batch)

        loss = 0
        for i in range(len(res.gt_imgs_pyr)):
            b, ns, c, h, w = res.gt_imgs_pyr[i].shape
            loss += torch.mean(torch.abs(res.gt_imgs_pyr[i].view(-1, c, h, w) - res.recs_pyr[i].view(-1, c, h, w)))

        loss.backward()
        print("Forward-backward time:", time.perf_counter() - start)
        #loss = compute_loss(res, something)

        #loss.backward()

        # test

    def test_batch_proc_multiframe(self):
        model = self._get_model_multiframe_stacked(batch_size=12)
        self._mock_batch_processing(model)

    def _test_batch_proc_uflow(self):
        model = self._get_model_uflow(batch_size=12, flow_backbone='uflow')
        self._mock_batch_processing(model)

    def test_batch_proc_uflow_lite(self):
        model = self._get_model_uflow(device='cuda', batch_size=12, flow_backbone='uflow_lite')
        self._mock_batch_processing(model)

if __name__ == '__main__':
    absltest.main()
    #m = ModelTest()
    #m.test_batch_proc_uflow_lite()
