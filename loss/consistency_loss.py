from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as VF
import torch.linalg as LA
import torch.nn.functional as F
from pytorch3d import transforms as transforms3D
import numpy as np

from .adaptive_loss import *
from log import log_results
from util.convert import any_nan
from util.rec import grid_sample


def l1(x, y, normalized=False):

    abs_diff = torch.abs(x - y)
    if normalized:
        abs_diff /= torch.abs(x) + torch.abs(y) + 1e-6

    return abs_diff.mean(1, keepdim=True)


def normalized_l1(x, y):
    return l1(x, y, normalized=True)


def dssim(x, y):
    x = F.pad(x, (1, 1, 1, 1), "reflect")
    y = F.pad(y, (1, 1, 1, 1), "reflect")
    mu_x = F.avg_pool2d(x, 3, 1)
    mu_y = F.avg_pool2d(y, 3, 1)
    mu_x_2 = mu_x**2
    mu_y_2 = mu_y**2
    mu_x_y = mu_x * mu_y
    sig_x = F.avg_pool2d(x**2, 3, 1) - mu_x_2
    sig_y = F.avg_pool2d(y**2, 3, 1) - mu_y_2
    sig_xy = F.avg_pool2d(x * y, 3, 1) - mu_x_y

    c1 = 0.01**2
    c2 = 0.03**2
    ssim_n = (2 * mu_x_y + c1) * (2 * sig_xy + c2)
    ssim_d = (mu_x_2 + mu_y_2 + c1) * (sig_x + sig_y + c2)

    return torch.clamp((1 - ssim_n / ssim_d) / 2, 0, 1).mean(1, True)


def normalize(x, mode="L2_feat"):
    if mode == "L2_feat":
        return x / (LA.norm(x, ord=2, dim=1, keepdim=True) + 1e-6)
    else:
        raise NotImplementedError("Normalization mode {} not implemented".format(mode))


def feature_distance(x, y, mode="L1_unitsum"):

    if mode == "L1_unitsum":
        diff = torch.abs(x - y)
        diff /= torch.abs(x) + torch.abs(y) + 1e-6
        diff.mean(1, keepdim=True)
        return diff

    elif mode == "squared_L2":
        diff = (x - y) ** 2
        return diff.mean(1, keepdim=True)

    elif mode == "L1":
        diff = torch.abs(x - y)
        return diff.mean(1, keepdim=True)

    else:
        raise NotImplementedError(
            "Feature distance mode {} not implemented".format(mode)
        )


def softmin(x, beta):
    values = F.softmin(beta * x, dim=1)
    return torch.sum(x * values.detach(), dim=1), values


def masked_mean(x, mask, dim=1):
    '''
    Args:
        x: a tensor of shape [b, c, h, w]
        mask: a visibility mask. 1 indicates that the value is vissible, otherwise is no vissible.
    '''
    assert x.shape == mask.shape
    return torch.sum(x * mask, dim=1, keepdim=True) / (
        torch.sum(mask, dim=1) + 1e-7
    ), torch.any(mask, dim=1)


def masked_min(x, mask, dim=1):
    '''
    Args:
        x: a tensor of shape [b, c, h, w]
        mask: a vissibility mask. 1 indicates that the value is vissible, otherwise is no vissible.

    Returns:
        ans: shape [d1, d2, .., d_dim=1, ...] ?
    '''

    assert x.shape == mask.shape

    almost_inf = (1e6 * (1.0 - mask)).detach()
    nx = (
        x + almost_inf
    )  # if mask is false in all views the gradient is the same, no values

    _, idx = torch.min(nx, dim=dim, keepdim=True)
    ans = torch.gather(x, dim, idx)

    min_mask = torch.zeros(x.shape).to(x.device)
    min_mask.scatter_(dim, idx, value=1)

    return ans, min_mask


def get_pooling_op(beta=1):
    if beta == float("inf"):
        return masked_min
    elif beta == 0:
        return masked_mean
    else:
        raise NotImplementedError(
            "Masking not implemented for softmin, beta {}".format(beta)
        )


def create_norm_op(
    name,
    num_recs,
    height=-1,
    width=-1,
    params_type=PARAMS_NONE,
    params_lb=1e-4,
    aux_weight=1,
):

    if name == "l1":
        if params_type != PARAMS_NONE:
            raise Exception("L1 does not support parameters")

        return L1()

    elif name == "laplacian2d_nll":
        return LaplacianNLL2D(
            params_type=params_type,
            params_lb=params_lb,
            aux_weight=aux_weight,
            num_recs=num_recs,
            height=height,
            width=width,
        )

    elif name == "laplacian2d_nll_v2":
        return LaplacianNLL2Dv2(
            params_type=params_type,
            params_lb=params_lb,
            aux_weight=aux_weight,
            num_recs=num_recs,
            height=height,
            width=width,
        )

    elif name == "laplacian0d_nll":
        return LaplacianNLL0D(params_lb, aux_weight)

    elif name == "laplacian0d_nll_v2":
        return LaplacianNLL0Dv2(params_lb, aux_weight)

    elif name == "charbonnier2d_nll":
        return CharbonnierNLL(
            params_type=params_type,
            params_lb=params_lb,
            num_recs=num_recs,
            height=height,
            width=width,
        )

    elif name == "charbonnier0d_nll":
        return CharbonnierNLL0D(params_lb)

    elif name == "cauchy2d_nll":
        return CauchyNLL(
            params_type=params_type,
            params_lb=params_lb,
            num_recs=num_recs,
            height=height,
            width=width,
        )

    elif name == "cauchy2d_nll_v2":
        return CauchyNLL2Dv2(
            params_type=params_type,
            params_lb=params_lb,
            num_recs=num_recs,
            height=height,
            width=width,
        )

    elif name == "cauchy0d_nll_v2":
        return CauchyNLL0Dv2(params_lb)

    else:
        raise NotImplementedError(name + " not implemented")


def handle_outliers(res, mask, mode="trim", percentile=0.1):
    b, nr, h, w = res.shape

    low_percentile = percentile / 2
    high_percentile = 1 - percentile / 2
    res = res.view(b, nr, -1)
    mask = mask.view(b, nr, -1)
    low = torch.quantile(res, low_percentile, dim=2, keepdim=True)
    high = torch.quantile(res, high_percentile, dim=2, keepdim=True)

    if mode == "trim":
        mask = torch.logical_and(mask, res > low)
        mask = torch.logical_and(mask, res < high)
    elif mode == "wisorize":
        res = torch.clamp(res, low, high)

    return res.view(b, nr, h, w), mask.view(b, nr, h, w)


def representation_consistency_scale(
    results, reps, params, i=0, norm=L1(), loss_terms_log=None
):
    pooling = get_pooling_op(params.softmin_beta)

    batch_size, num_recs, c, h, w = reps.recs_pyr[i].size()
    _, _, _, h_rep, w_rep = reps.proj_coords_pyr[i].size()

    full_res_rec = (h_rep != h) or (w_rep != w)
    if full_res_rec:
        tgt_imgs = results.tgt_imgs_pyr[0].reshape(-1, c, h, w)
        src_imgs = results.src_imgs_pyr[0].reshape(-1, c, h, w)
    else:
        tgt_imgs = results.tgt_imgs_pyr[i].reshape(-1, c, h, w)
        src_imgs = results.src_imgs_pyr[i].reshape(-1, c, h, w)

    # recs shape depends on loss_full_res
    recs = reps.recs_pyr[i].view(-1, c, h, w)
    loss_terms_log = {}

    res = (1.0 - params.weight_ssim) * l1(tgt_imgs, recs)
    if params.weight_ssim > 0:
        res += params.weight_ssim * dssim(tgt_imgs, recs)

    if params.auto_mask:
        res_noproj = l1(tgt_imgs, src_imgs)
        if params.weight_ssim > 0:
            res_noproj += params.weight_ssim * dssim(tgt_imgs, src_imgs)
        auto_mask = res < res_noproj

        if full_res_rec:
            mask = VF.resize(reps.masks_pyr[i], (h, w), VF.InterpolationMode.BILINEAR)
        else:
            mask = reps.masks_pyr[i]
        reps.masks_pyr[i] = mask * auto_mask.view(-1, num_recs, h, w).float()

    mask = reps.masks_pyr[i]  # [rec_bs, num_src, h, w]

    if loss_terms_log is None:
        mask_detached = mask.detach()
        loss_terms_log[("cc", i)].append(((res.detach() * mask_detached)).mean().item())

    if params.weight_sc > 0:
        _, _, sc, _, _ = reps.proj_coords_pyr[i].size()
        coord_res = normalized_l1(
            reps.proj_coords_pyr[i].view(-1, sc, h_rep, w_rep),
            reps.sampled_coords_pyr[i].view(-1, sc, h_rep, w_rep),
        )
        coord_res = coord_res.view(batch_size, num_recs, h_rep, w_rep)
        coord_res = coord_res.view(-1, 1, h_rep, w_rep)

        if full_res_rec:
            coord_res = VF.resize(coord_res, (h, w), VF.InterpolationMode.BILINEAR)

        assert coord_res.size() == res.size()
        res += params.weight_sc * coord_res

        if loss_terms_log is None:
            loss_terms_log.append(
                (coord_res.detach() * mask_detached).mean().item() * params.weight_sc
            )

    if params.weight_fc > 0 and i > 0:
        _, _, fc, _, _ = reps.feats_pyr[i - 1].size()

        fa = reps.feats_pyr[i - 1].view(-1, fc, h_rep, w_rep)
        fb = reps.sampled_feats_pyr[i - 1].view(-1, fc, h_rep, w_rep)

        if params.fc_detach:
            fa = fa.detach()
            fb = fb.detach()

        if params.fc_norm is not None:
            fa = normalize(fa, params.fc_norm)
            fb = normalize(fb, params.fc_norm)

        feat_res = feature_distance(fa, fb, params.fc_diff)
        feat_res = feat_res.view(batch_size, num_recs, h_rep, w_rep)
        feat_res = feat_res.view(-1, 1, h_rep, w_rep)
        if full_res_rec:
            feat_res = VF.resize(feat_res, (h, w), VF.InterpolationMode.BILINEAR)

        assert feat_res.size() == res.size()
        res += params.weight_fc * feat_res

        if loss_terms_log is None:
            loss_terms_log.append(
                (feat_res.detach() * mask_detached).mean().item() * params.weight_fc
            )

    res = res.view(batch_size, num_recs, h, w)

    if params.uncertainty:
        extra = reps.tgt_uncerts_pyr[i]

        if params.loss_outliers_qt > 0.0:
            res, mask = handle_outliers(
                res, mask, params.loss_outliers_mode, params.loss_outliers_qt
            )

        res = res.reshape(-1, 1, h, w)
        _, _, num_ext_channels, eh, ew = extra.shape
        assert num_ext_channels == 1
        extra = extra.reshape(-1, 1, num_ext_channels, eh, ew)

        err = norm(res, extra)
    else:

        if params.loss_outliers_qt > 0.0:
            res, mask = handle_outliers(
                res, mask, params.loss_outliers_mode, params.loss_outliers_qt
            )

        res = res.reshape(-1, 1, h, w)
        err = norm(res, scale_idx=i)

    err = err.view(batch_size, -1, h, w)
    mask = mask.view(batch_size, -1, h, w)

    pooled_err, pooling_weights = pooling(err, mask)

    return pooled_err, pooling_weights, err, res


def representation_consistency(
    results,
    params,
    norm=L1(),
    return_residuals=False,
):

    num_scales = len(results.tgt_imgs_pyr)

    total_loss = 0
    res_pyr = []
    err_pyr = []
    pooled_err_pyr = []

    for i in range(num_scales):
        if params.weight_rigid > 0:
            (
                rigid_pooled_err,
                rigid_pooling_weights,
                rigid_err,
                rigid_res,
            ) = representation_consistency_scale(
                results, results.rigid_reps, params, i=i, norm=norm
            )

        if params.weight_nonrigid > 0:
            (
                full_pooled_err,
                full_pooling_weights,
                full_err,
                full_res,
            ) = representation_consistency_scale(
                results, results.nonrigid_reps, params, i=i, norm=norm
            )

        if params.merge_op == "min":
            assert params.weight_nonrigid > 0 and params.weight_rigid > 0

            pooled_err, idx = torch.min(
                torch.stack([rigid_pooled_err, full_pooled_err]), dim=0, keepdim=True
            )
            pooled_err = pooled_err.squeeze(0)
            res = torch.gather(
                torch.stack([rigid_res, full_res]), dim=0, index=idx
            ).squeeze(0)
            err = torch.gather(
                torch.stack([rigid_err, full_err]), dim=0, index=idx
            ).squeeze(0)
            pooling_weights = torch.gather(
                torch.stack([rigid_pooling_weights, full_pooling_weights]),
                dim=0,
                index=idx,
            ).squeeze(0)

        elif params.merge_op == "sum":
            res = 0
            err = 0
            pooled_err = 0
            pooling_weights = 0

            if params.weight_rigid > 0:
                res += params.weight_rigid * rigid_res
                err += params.weight_rigid * rigid_err
                pooled_err += params.weight_rigid * rigid_pooled_err
                pooling_weights += params.weight_rigid * rigid_pooling_weights

            if params.weight_nonrigid > 0:
                res += params.weight_nonrigid * full_res
                err += params.weight_nonrigid * full_err
                pooled_err += params.weight_nonrigid * full_pooled_err
                pooling_weights += params.weight_nonrigid * full_pooling_weights
        else:
            raise NotImplementedError(
                "Merge with op {} not implemented", params.merge_op
            )

        # Is this affected by full scale loss
        loss = torch.mean(pooled_err) * 1.0 / (2**i)
        total_loss += loss
        res_pyr.append(res)
        err_pyr.append(err)
        pooled_err_pyr.append((pooled_err, pooling_weights))

    if return_residuals:
        return total_loss, res_pyr, err_pyr, pooled_err_pyr
    else:
        return total_loss


def rotation_consistency(results):
    # we assume T has T_forward and T_backward concatenated around axis=0
    b, _, _, _ = results.T.shape
    T_fwd = results.T[: b // 2]
    T_bwd = results.T[b // 2 :]
    R = T_fwd[:, :, :3, :3]
    R_backward = T_bwd[:, :, :3, :3]

    ones = torch.ones(3)
    identity = torch.diag(ones).unsqueeze(0).unsqueeze(0).to(R.device)

    # We should use sum instead of mean and  we should downscale weight_rc by 3*3
    n = torch.mean(torch.square(torch.matmul(R, R_backward) - identity), dim=(2, 3))
    d = torch.mean(torch.square(R - identity), dim=(2, 3)) + torch.sum(
        torch.square(R_backward - identity), dim=(2, 3)
    )
    ans = n / (1e-14 + d)  # update to 1e-7

    return torch.mean(ans)


def translation_consistency(results):

    T_fwd = results.T.view(-1, 4, 4)
    b, _, _ = T_fwd.size()
    T_bwd = torch.cat((T_fwd[b // 2 :], T_fwd[: b // 2]), axis=0)
    R_backward = T_bwd[:, :3, :3]

    Tr_rigid_fwd = T_fwd[:, :3, 3:4]
    Tr_rigid_bwd = T_bwd[:, :3, 3:4]
    Tr_error_norm = torch.sum(
        torch.square(torch.matmul(R_backward, Tr_rigid_fwd) + Tr_rigid_bwd), dim=1
    )
    Tr_fwd_norm = torch.sum(torch.square(Tr_rigid_fwd), dim=1)
    Tr_bwd_norm = torch.sum(torch.square(Tr_rigid_bwd), dim=1)

    return torch.mean(Tr_error_norm / (Tr_fwd_norm + Tr_bwd_norm + 1e-24))


def translation_consistency_pixelwise(results):
    T_fwd = results.T.view(-1, 4, 4)
    b, _, _ = T_fwd.size()
    T_bwd = torch.cat((T_fwd[b // 2 :], T_fwd[: b // 2]), axis=0)
    R_backward = T_bwd[:, :3, :3]

    ans = 0
    num_scales = len(results.total_flows_pyr)
    assert results.T.size(0) == results.total_flows_pyr[0].size(0)
    for i in range(num_scales):
        Tr_fwd = results.total_flows_pyr[i]
        # assume bidirectional=True
        Tr_bwd = torch.cat(
            (
                results.total_flows_pyr[i][b // 2 :],
                results.total_flows_pyr[i][: b // 2],
            ),
            axis=0,
        )
        Tr_bwd_warped = grid_sample(Tr_bwd, results.warp_pyr[i].detach())

        Tr_bwd_warped = Tr_bwd_warped.view(b, 3, -1)
        Tr_fwd = Tr_fwd.view(b, 3, -1)

        # R_b [b, 3, 3] * Tr_fwd [b, 3, h * w]
        Tr_error_norm = torch.sum(
            torch.square(torch.matmul(R_backward, Tr_fwd) + Tr_bwd_warped), dim=1
        )
        Tr_fwd_norm = torch.sum(torch.square(Tr_fwd), dim=1)
        Tr_bwd_norm = torch.sum(torch.square(Tr_bwd_warped), dim=1)
        ans += (1 / (2**i)) * torch.mean(
            Tr_error_norm / (Tr_fwd_norm + Tr_bwd_norm + 1e-24)
        )


def _gradient_x(img):
    img = F.pad(img, (0, 0, 0, 1), mode="reflect")
    return img[:, :, :-1, :] - img[:, :, 1:, :]


def _gradient_y(img):
    img = F.pad(img, (0, 1, 0, 0), mode="reflect")
    return img[:, :, :, :-1] - img[:, :, :, 1:]


def exp_gradient(imgs, alpha=1):
    '''
    Returns:
      wx, wy: A pair of tensors with the exponent of the negative average gradient. Each tensor has a shape [b*s, 1, h, w]
    '''
    dx = _gradient_x(imgs)
    dy = _gradient_y(imgs)
    wx = torch.exp(-alpha * torch.mean(torch.abs(dx), dim=1, keepdim=True))
    wy = torch.exp(-alpha * torch.mean(torch.abs(dy), dim=1, keepdim=True))
    return wx, wy


def spatial_smoothness(target, order=1, norm=2):
    '''
    Args:
        target: a tensor containing target data where the smoothness
        constraint is applied, at one or multi scales. Each tensor has a shape [b, c>=1, h, w]
    '''
    assert len(target.size()) == 4

    dnx = target
    dny = target
    for j in range(order):
        dnx = _gradient_x(dnx)
        dny = _gradient_y(dny)

    _, num_ch, _, _ = target.size()
    dn = torch.cat((dnx.unsqueeze(2), dny.unsqueeze(2)), axis=2)
    # normalized dimension is removed
    sm = torch.linalg.norm(dn, ord=norm, dim=2)

    return torch.mean(sm)


def motion_sparsity(normalized_T):
    abs_T = torch.abs(normalized_T)
    avg_abs_T = torch.mean(abs_T, dim=(2, 3), keepdim=True).detach() + 1e-12

    return torch.mean(
        avg_abs_T
        * torch.mean(torch.sqrt(abs_T / avg_abs_T + 1), dim=(2, 3), keepdim=True)
    )


def motion_regularization(residual_T, T, w_smoothness, w_sparsity):
    num_scales = len(T)

    loss = 0
    for i in range(num_scales):
        assert len(T[i].shape) == 4
        norm = torch.mean(T[i] ** 2, dim=(1, 2, 3), keepdim=True) * 3
        norm = torch.sqrt(norm + 1e-12)  # the norm can not be -0 (EPS)

        normalized_res_T = residual_T[i] / norm

        loss_i = w_smoothness * spatial_smoothness(normalized_res_T)
        loss_i += w_sparsity * motion_sparsity(normalized_res_T)
        loss += (1.0 / (2**i)) * loss_i

    return loss


def weighted_spatial_smoothness(data, weights, order=2):
    '''
    Args:
        data: a tensor containing the data to be smoothed [b, c, h, w]
        weights: a pair of tensors containing the weights for the x and y components of each pixel to be smoothed.
    '''
    weights_x, weights_y = weights

    dnx = data
    dny = data
    for j in range(order):
        dnx = _gradient_x(dnx)
        dny = _gradient_y(dny)

    return torch.mean(weights_x * torch.abs(dnx) + weights_y * torch.abs(dny))


def normalized_smoothness(target, imgs, order=2, alpha=1):
    '''
    Args:
        disps: a list of tensors containing the predicted disparity maps at multiple
        scales. Each tensor has a shape [batch_size * seq_len, 1, h, w].
    '''
    wx, wy = exp_gradient(imgs, alpha)

    target_mean = torch.mean(target, dim=[2, 3], keepdim=True)
    norm_target = target / (target_mean + 1e-7)

    return weighted_spatial_smoothness(norm_target, (wx, wy), order)


def flow_smoothness(flows, imgs, order=1, alpha=1):

    wx, wy = exp_gradient(imgs, alpha)

    b, c, h, w = imgs.size()
    num_flows = flows.size(0)
    num_src = int(num_flows / b)
    assert num_flows % b == 0

    fwx = wx.repeat(1, num_src, 1, 1)
    fwy = wy.repeat(1, num_src, 1, 1)
    fwx = fwx.view(num_flows, 1, h, w)
    fwy = fwy.view(num_flows, 1, h, w)

    norm = torch.sum(flows**2, dim=(1, 2, 3), keepdim=True)
    norm = torch.sqrt(norm + 1e-7)  # the norm can not be -0 (EPS)
    flows = flows / norm

    return weighted_spatial_smoothness(flows, weights=(fwx, fwy), order=order)


def _skew_symmetric(x, batch_size):
    mat = torch.zeros(batch_size, 3, 3, device=x.device)
    mat[:, 0, 1] = -x[:, 2]
    mat[:, 0, 2] = x[:, 1]
    mat[:, 1, 2] = -x[:, 0]

    return mat - torch.transpose(mat, 1, 2)


EPIPOLAR_ALGEBRAIC = 0
EPIPOLAR_SAMPSON = 1


def epipolar_constraint(coords, results, mode=EPIPOLAR_ALGEBRAIC):
    '''
    Args:

      flows: a list of tensors containing the optical flow from target to source.
      T: a list of transformation from the target to the source frames
    '''
    num_scales = len(coords)
    batch_size = coords[0].size(0)

    R = results.T[:, :3, :3]
    t_skew = _skew_symmetric(results.T[:, :3, 3], batch_size)

    loss = 0
    for i in range(num_scales):

        num_points = coords[i].size(2)
        ones = torch.ones(batch_size, num_points, 1, 1, device=coords[i].device)

        p_ = torch.transpose(coords[i], 1, 2).unsqueeze(3)
        p_ = torch.cat([p_, ones], 2)  # Why?

        # flow from the target to the source planes, on the target coordinate grid 
        proj_coords = coords[i] + results.total_flows_pyr[i].view(batch_size, 2, -1)
        q_ = torch.transpose(proj_coords, 1, 2).unsqueeze(3)
        q_ = torch.cat([q_, ones], 2)

        iK = results.inv_K_pyr[i][:, :3, :3]
        F = torch.matmul(
            torch.transpose(iK, 1, 2), torch.matmul(R, torch.matmul(t_skew, iK))
        ).unsqueeze(1)

        Fp = torch.matmul(F, p_)
        qTFp = torch.matmul(torch.transpose(q_, 2, 3), Fp)

        if mode == EPIPOLAR_ALGEBRAIC:
            loss += (1 / (2**i)) * torch.mean(torch.abs(qTFp))

        elif mode == EPIPOLAR_SAMPSON:
            FTq = torch.matmul(torch.transpose(F, 2, 3), q_)
            loss += (1 / (2**i)) * torch.mean(
                torch.square(qTFp)
                / (
                    torch.sum(torch.square(Fp[:, :, :2]), axis=2, keepdim=True)
                    + torch.sum(torch.square(FTq[:, :, :2]), axis=2, keepdim=True)
                    + 1e-6
                ).detach()
            )

    return loss


def flow_consistency(rigid_flow, nonrigid_flow):
    assert len(rigid_flow.shape) == 4

    diff = torch.sum((nonrigid_flow - rigid_flow.detach()) ** 2, dim=1, keepdim=True)
    unit_mean_diff = diff / torch.mean(diff, dim=(1, 2, 3), keepdim=True)
    sharpness = 4
    rigid_mask = (1 / (1 + torch.pow(unit_mean_diff, sharpness))).detach()
    loss = torch.sum(rigid_mask * diff, dim=(1, 2, 3)) / torch.sum(
        rigid_mask, dim=(1, 2, 3)
    )
    loss = torch.mean(loss)

    return loss, rigid_mask