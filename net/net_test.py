from absl.testing import absltest
import numpy as np

import torch
import torch.nn

from net.base_net import *

class BaseNetTest(absltest.TestCase):

    def _get_input(self):
        return torch.ones((2, 3, 128, 512))

    def test_uflownet_smaller_output(self):
        encoder = UFlowEncoder()
        decoder = UFlowDecoder(same_resolution=False)

        x1 = self._get_input()
        x2 = self._get_input()
        feats1 = encoder(x1)
        feats2 = encoder(x2)

        flow_pyr = decoder(feats1, feats2)

        _, _, h_in, w_in = x1.shape
        _, _, h, w = flow_pyr[0].shape

        assert h_in == 4 * h and w_in == 4 * w

    def test_uflownet_same_resolution(self):#
        encoder = UFlowEncoder()
        decoder = UFlowDecoder(same_resolution=True)

        x1 = self._get_input()
        x2 = self._get_input()
        feats1 = encoder(x1)
        feats2 = encoder(x2)
        flow_pyr = decoder(feats1, feats2)

        _, _, h_in, w_in = x1.shape
        _, _, h, w = flow_pyr[0].shape

        assert h_in == h and w_in == w

    def test_uflownet_extra_channels(self):#
        encoder = UFlowEncoder()
        decoder = UFlowDecoder(same_resolution=True, num_ch_out=3)

        x1 = self._get_input()
        x2 = self._get_input()
        feats1 = encoder(x1)
        feats2 = encoder(x2)
        flow_pyr = decoder(feats1, feats2)

        _, _, h_in, w_in = x1.shape
        _, _, h, w = flow_pyr[0].shape

        assert h_in == h and w_in == w

if __name__ == '__main__':
    absltest.main()
