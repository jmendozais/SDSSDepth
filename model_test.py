from absl.testing import absltest
import numpy as np

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

    def _get_model(self, device='cpu', batch_size=2):
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
            flow_backbone='uflow',
            dropout=0, 
            loss_noaug=False, 
            larger_pose=True,
            pred_disp=True)

        model_ = model_.to(device)

        return model_


    def test_batch_proc(self):#
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

    def test_gpu_memo(self):
        model = self._get_model(batch_size=12)
        dataset = DatasetMock(model.seq_len, model.height, model.width)
        dataloader = torch.utils.data.DataLoader(dataset, 
            batch_size=model.batch_size, 
            shuffle=True, 
            num_workers=0, 
            pin_memory=True, 
            drop_last=True)
        
        it = iter(dataloader)
        batch = next(it)
           
        res = model(batch, batch)
        model = model.cuda()

        #loss = compute_loss(res, something)

        #loss.backward()

        # test



if __name__ == '__main__':
    absltest.main()
