import time

from absl.testing import absltest
from argparse import Namespace

import numpy as np
import torch
import torch.nn
import torch.utils.data as data
import torchvision.transforms.functional as T

from net.base_net import *
from data import Dataset
import loss
import model
import train
import opts

DEVICE = 'cpu' 


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
        ms_snippets['K'] = []

        K = np.array([[241.2800, 0.0000, 208.0000, 0.0000],
                      [0.0000, 245.7600, 64.0000, 0.0000],
                      [0.0000, 0.0000, 1.0000, 0.0000],
                      [0.0000, 0.0000, 0.0000, 1.0000]]).astype(np.float32)

        ms_snippets['K'] = K

        for i in range(1, self.num_scales):
            size = (self.width // (2**i), self.height // (2**i))
            ms_snippets[i] = snippet[:, :, :size[1], :size[0]]

        return ms_snippets

    def __len__(self):

        return self.size


class ModelTest(absltest.TestCase):

    def _get_model_uflow(self, device='cpu', batch_size=2,
                         flow_backbone='uflow_lite'):
        seq_len = 3
        height = 128
        width = 416

        norm_op = loss.create_norm_op(
            name='l1',
            params_type=2,  # Params None
            params_lb=0,
            num_recs=seq_len - 1,
            height=height,
            width=width)

        norm_op.to(device)

        params = Namespace()
        params.batch_size = batch_size,
        params.num_scales = 4,
        params.seq_len = seq_len,
        params.height = height,
        params.width = width,
        params.rec_mode = 'depth',
        params.multiframe_of = False,
        params.stack_flows = False,
        params.learn_intrinsics = False,
        params.norm = 'bn',
        params.debug = False,
        params.depth_backbone = 'resnet',
        params.flow_backbone = flow_backbone,
        params.dropout = 0.1,
        params.loss_noaug = False
        params.depthnet_out = 'disp',
        params.pose_layers = 0

        model_ = model.Model(
            params=params,
            num_extra_channels=norm_op.num_pred_params)

        model_ = model_.to(device)

        return model_, norm_op

    def _get_model_multiframe_stacked(self, device='cpu', batch_size=2):
        seq_len = 3
        height = 128
        width = 416

        norm_op = loss.create_norm_op(
            name='cauchy2d_nll_v2',
            num_recs=seq_len - 1)

        norm_op.to(device)

        params = Namespace()
        params.batch_size = batch_size
        params.num_scales = 4
        params.seq_len = seq_len
        params.height = height
        params.width = width
        params.rec_mode = 'depth'
        params.multiframe_of = True
        params.stack_flows = True
        params.learn_intrinsics = False
        params.norm = 'bn'
        params.debug = False
        params.depth_backbone = 'cpcv2'
        params.flow_backbone = 'cpcv2'
        params.dropout = 0.1
        params.loss_noaug = False
        params.depthnet_out = 'disp'
        params.pose_layers = 3
        params.pose_dp = 0.0

        model_ = model.Model(
            params=params,
            num_extra_channels=norm_op.num_pred_params,
            dim_extra=norm_op.dim
        )

        return model_, norm_op

    def _test_forward(self):
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

    def _mock_batch_processing(self, model, params, norm_op, device='cpu'):
        dataset = DatasetMock(model.seq_len, model.height, model.width)
        dataloader = torch.utils.data.DataLoader(dataset,
                                                 batch_size=model.batch_size,
                                                 shuffle=True,
                                                 num_workers=0,
                                                 pin_memory=False,
                                                 drop_last=True)

        it = iter(dataloader)
        batch = next(it)

        start = time.perf_counter()
        # Forward
        for i in range(model.num_scales):
            batch[i] = batch[i].to(device)
        model = model.to(device)
        res = model(batch, batch)

        batch_loss = train.compute_loss(
            model=model,
            data=batch,
            results=res,
            params=params,
            norm_op=norm_op,
            log_misc=False,
            log_depth=False,
            log_flow=False,
            it=-1,
            epoch=-1,
            writer=None)

        batch_loss.backward()
        print("Forward-backward time:", time.perf_counter() - start)

    def _test_batch_proc_multiframe(self):
        model, norm_op = self._get_model_multiframe_stacked(batch_size=2)
        self._mock_batch_processing(model, norm_op)

    def test_seq_2(self):
        seq_len = 2
        height = 128
        width = 416
        device = 'cpu'
        batch_size = 2

        norm_op = loss.create_norm_op(
            name='l1',
            num_recs=seq_len - 1,
        )

        norm_op.to(device)

        params = opts.parse_args()

        params.batch_size = batch_size
        params.num_scales = 4
        params.seq_len = seq_len
        params.height = height
        params.width = width
        params.nonrigid_mode = 'opt'
        params.merge_op = 'sum'
        params.motion_mode = 'by_pair'
        params.learn_intrinsics = False
        params.norm = 'bn'
        params.debug = False
        params.depth_backbone = 'resnet'
        params.depth_occ = 'outfov'
        params.flow_backbone = 'resnet'
        params.dropout = 0.1
        params.loss_noaug = False
        params.depthnet_out = 'disp'
        params.pose_layers = 3
        params.pose_dp = 0.0
        params.bidirectional = True

        model_ = model.Model(
            params=params,
            num_extra_channels=norm_op.num_pred_params,
            dim_extra=norm_op.dim)

        self._mock_batch_processing(model_, params, norm_op)

    def _test_batch_proc_uflow(self):
        model = self._get_model_uflow(batch_size=2, flow_backbone='uflow')
        self._mock_batch_processing(model)

    def _test_batch_proc_uflow_lite(self):
        model, norm_op = self._get_model_uflow(
            device='cpu', batch_size=2, flow_backbone='uflow_lite')
        self._mock_batch_processing(model, norm_op)


if __name__ == '__main__':
    absltest.main()
