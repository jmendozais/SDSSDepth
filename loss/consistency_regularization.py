import os
import time

import copy
import torch
from torch import functional as F
from torchvision.transforms import functional as VF

from util.convert import snippet_to_matched_pairs
from log import save_depth, save_gray, save_img

import loss.consistency_loss as consistency_loss


def MSE(real, fake):
    assert len(real.shape) == 4
    assert len(fake.shape) == 4

    return (real - fake) ** 2


def MAE(real, fake):
    assert len(real.shape) == 4
    assert len(fake.shape) == 4

    return torch.abs(real - fake)


def berhu(real, fake, threshold=0.2):
    assert len(real.shape) == 4
    assert len(fake.shape) == 4

    mask = real > 0
    if not fake.shape == real.shape:
        _, _, H, W = real.shape
        fake = F.upsample(fake, size=(H, W), mode="bilinear")

    fake = fake * mask
    diff = torch.abs(real - fake)
    delta = threshold * torch.max(diff).item()

    part1 = -F.threshold(-diff, -delta, 0.0)
    part2 = F.threshold(diff**2 - delta**2, 0.0, -(delta**2.0)) + delta**2
    part2 = part2 / (2.0 * delta)

    loss = part1 + part2
    return loss


def SSI_MAE(real, fake, detach=True):
    assert len(real.shape) == 4
    assert len(fake.shape) == 4

    b, c, h, w = real.shape
    real = real.view(b, -1)
    fake = fake.view(b, -1)
    real_m, _ = real.median(dim=1, keepdim=True)
    fake_m, _ = fake.median(dim=1, keepdim=True)
    real_d = torch.abs(real - real_m).mean(dim=1, keepdim=True)
    fake_d = torch.abs(fake - fake_m).mean(dim=1, keepdim=True)

    if detach:
        real_m = real_m.detach()
        fake_m = fake_m.detach()
        real_d = real_d.detach()
        fake_d = fake_d.detach()

    real_ssi = (real - real_m) / (real_d + 1e-7)
    fake_ssi = (fake - fake_m) / (fake_d + 1e-7)

    return torch.abs(real_ssi - fake_ssi).view(b, c, h, w)


def log_consistency_regularization(
    imgs_w, imgs_s, plabel, pred, mask, err_map, res, batch_mask, scale, it, args
):
    assert len(plabel.shape) == 4
    assert len(imgs_w.shape) == 4

    b_ns, c, h, w = imgs_w.shape

    id = int(int(time.time() * 1e6) % 1000000)
    plt_dir = "{}/plt_log".format(args.log)
    os.makedirs(plt_dir, exist_ok=True)

    for j in range(b_ns):
        pl_fname = "{}/s{}_it{}_{:.4f}_bi{}_plb_ispl{}_id{}.jpg".format(
            plt_dir,
            scale,
            it,
            res[j].detach().item(),
            j,
            batch_mask[j].detach().item(),
            id,
        )
        save_depth(pl_fname, plabel[j, 0])
        pred_fname = "{}/s{}_it{}_{:.4f}_bi{}_predb_ispl{}_id{}.jpg".format(
            plt_dir,
            scale,
            it,
            res[j].detach().item(),
            j,
            batch_mask[j].detach().item(),
            id,
        )
        save_depth(pred_fname, pred[j, 0])
        fname = "{}/s{}_it{}_{:.4f}_bi{}_pl_imgb_ispl{}_id{}.jpg".format(
            plt_dir,
            scale,
            it,
            res[j].detach().item(),
            j,
            batch_mask[j].detach().item(),
            id,
        )
        save_img(fname, imgs_w[j])
        fname = "{}/s{}_it{}_{:.4f}_bi{}_pred_imgb_ispl{}_id{}.jpg".format(
            plt_dir,
            scale,
            it,
            res[j].detach().item(),
            j,
            batch_mask[j].detach().item(),
            id,
        )
        save_img(fname, imgs_s[j])
        mask_fname = "{}/s{}_it{}_{:.4f}_bi{}_masks_ispl{}_id{}.jpg".format(
            plt_dir,
            scale,
            it,
            res[j].detach().item(),
            j,
            batch_mask[j].detach().item(),
            id,
        )
        save_gray(mask_fname, mask[j, 0])
        if err_map is not None:
            emt_fname = "{}/s{}_it{}_{:.4f}_bi{}_err_ispl{}_id{}.jpg".format(
                plt_dir,
                scale,
                it,
                res[j].detach().item(),
                j,
                batch_mask[j].detach().item(),
                id,
            )
            save_depth(emt_fname, err_map[j, 0])


class EMA(torch.nn.Module):
    def __init__(self, model, alpha):
        super(EMA, self).__init__()

        self.model = model
        self.ema_model = copy.deepcopy(model)
        self.alpha = alpha

        model_sd = self.model.state_dict()
        ema_model_sd = self.ema_model.state_dict()
        for k in ema_model_sd:
            ema_model_sd[k] = model_sd[k].detach().clone()

        self.iter = 1

    def forward(self, inputs):
        return self.ema_model(inputs)

    def update(self):
        alpha_corrected = min(1 - 1 / (self.iter + 1), self.alpha)
        for ema_param, param in zip(
            self.ema_model.parameters(), self.model.parameters()
        ):
            ema_param.data.mul_(alpha_corrected).add_(
                param.data.detach(), alpha=1 - alpha_corrected
            )
        self.iter += 1


class ConsistencyRegularizationLoss:
    def __init__(self, args):
        self.scales = args.num_scales

        self.error_fn = args.cr_error_fn
        self.img_thresh = args.cr_img_thresh
        self.img_beta = args.cr_img_beta
        self._img_thresh_value = 0

        self.pix_thresh = args.cr_pix_thresh
        self.pix_beta = args.cr_pix_beta
        self.pix_use_recerr = args.cr_pix_recerr
        self._pix_thresh_value = None

        self.training_step = 0

        # CR mode config
        self.cr_mode = args.cr_mode
        self.detach_pl = self.cr_mode in ["pl", "ema"]

        self.use_disp = args.depthnet_out == "disp"
        self.weight_grad_match = args.cr_weight_gm
        self.weight_egomotion = args.cr_weight_egom
        self.full_res = args.cr_full_res

        self.scale_depth = args.cr_scale_depth
        self.DEPTH_SF = 0.01

    def compute_error(self, plabel, pred):
        if self.error_fn == "MSE":
            err_map = MSE(plabel, pred)
        elif self.error_fn == "MAE":
            err_map = MAE(plabel, pred)
        elif self.error_fn == "berhu":  # this works with depth
            err_map = berhu(plabel, pred)
        elif self.error_fn == "SSI_MAE_detach":  # this works well with disp
            err_map = SSI_MAE(plabel, pred)
        elif self.error_fn == "SSI_MAE_nodetach":  # this work well with disp
            err_map = SSI_MAE(plabel, pred, detach=False)
        else:
            raise NotImplementedError("{} mode not implememnted".format(self.error_fn))

        return err_map

    def compute_pair(
        self,
        wt_results,
        st_data,
        st_results,
        args,
        it,
        is_training,
        log=False,
        writter=None,
    ):

        if is_training:
            self.training_step += 1

        b, ns, c, height, width = wt_results.tgt_imgs_pyr[0].shape

        _, _, err, _ = consistency_loss.representation_consistency_scale(
            wt_results, wt_results.rigid_reps, args, i=0
        )

        err = err.detach()
        err = err.view(b * ns, 1, height, width)

        # Image level filtering
        if self.img_thresh > 0:
            tgt_masks, _ = snippet_to_matched_pairs(
                st_data[("mask_snp", 0)], args.seq_len, args.bidirectional
            )
            img_mask_flat = tgt_masks.view(-1, 1, height, width) > 0
            assert len(img_mask_flat.shape) == len(err.shape)
            assert img_mask_flat.shape[0] == err.shape[0]

            # Calculate image level threshold
            err_masked = torch.masked_select(err, img_mask_flat)
            batch_thresh = torch.quantile(
                err_masked, q=(1.0 - self.img_thresh)
            ).detach()

            if self.img_beta > 0:
                if is_training:
                    self._img_thresh_value = (
                        self._img_thresh_value * self.img_beta
                        + batch_thresh * (1 - self.img_beta)
                    )
                img_thresh_value = self._img_thresh_value / (
                    1 - (self.img_beta**self.training_step)
                )  # Apply bias correction
            else:
                img_thresh_value = batch_thresh

            err_loc = torch.ones((b * ns,), device=err.device)
            for i in range(b * ns):
                err_masked_i = torch.masked_select(err[i], img_mask_flat[i])
                err_loc[i] = torch.quantile(err_masked_i, q=0.5)

            batch_mask = err_loc < img_thresh_value

        else:
            batch_mask = torch.ones((b * ns,), device=err.device) > 0

        # Compute consistency regularization term
        cr_loss = torch.zeros(size=(b * ns,), device=wt_results.tgt_imgs_pyr[0].device)

        if self._pix_thresh_value is None:
            self._pix_thresh_value = torch.zeros((self.scales,), device=err.device)

        for i in range(self.scales):
            b, ns, c, h_i, w_i = wt_results.tgt_depths_pyr[i].shape

            pred_depths = st_results.tgt_depths_pyr[i].view(-1, 1, h_i, w_i)

            if self.full_res:
                tgt_depths, _ = snippet_to_matched_pairs(
                    st_data[("pred_depth_snp", 0)], args.seq_len, args.bidirectional
                )
                tgt_masks, _ = snippet_to_matched_pairs(
                    st_data[("mask_snp", 0)], args.seq_len, args.bidirectional
                )

                pred_depths = VF.resize(
                    pred_depths, (height, width), VF.InterpolationMode.BILINEAR
                )
                h_i = height
                w_i = width

            else:
                tgt_depths, _ = snippet_to_matched_pairs(
                    st_data[("pred_depth_snp", i)], args.seq_len, args.bidirectional
                )
                tgt_masks, _ = snippet_to_matched_pairs(
                    st_data[("mask_snp", i)], args.seq_len, args.bidirectional
                )

            if self.scale_depth:
                tgt_depths *= self.DEPTH_SF
                pred_depths *= self.DEPTH_SF

            if self.use_disp:
                plabel = 1.0 / tgt_depths.view(-1, 1, h_i, w_i)
                pred = 1.0 / pred_depths
            else:
                plabel = tgt_depths.view(-1, 1, h_i, w_i)
                pred = pred_depths

            img_mask_flat = tgt_masks.view(-1, 1, h_i, w_i)
            img_mask_flat = img_mask_flat > 0

            if self.detach_pl:
                plabel = plabel.detach()

            err_map = self.compute_error(plabel, pred)

            err_map_to_thresh = None

            # Pixel level filtering
            if self.pix_thresh > 0.0:
                if self.pix_use_recerr:  # err_map_filtering
                    target_scale = 0 if self.full_res else i
                    err_map_to_thresh, _ = snippet_to_matched_pairs(
                        st_data[("error_snp", target_scale)],
                        args.seq_len,
                        args.bidirectional,
                    )
                    err_map_to_thresh = err_map_to_thresh.view(-1, 1, h_i, w_i)
                else:
                    err_map_to_thresh = err_map

                err_map_masked = torch.masked_select(err_map_to_thresh, img_mask_flat)
                batch_thresh = torch.quantile(
                    err_map_masked, q=(1.0 - self.pix_thresh)
                ).detach()

                if self.pix_beta > 0:
                    if is_training:
                        self._pix_thresh_value[i] = self._pix_thresh_value[
                            i
                        ] * self.pix_beta + batch_thresh * (1 - self.pix_beta)
                    # Apply bias correction
                    pix_thresh_value = self._pix_thresh_value[i] / (
                        1 - (self.pix_beta**self.training_step)
                    )
                else:
                    pix_thresh_value = batch_thresh

                nocr = torch.sum(img_mask_flat > 0)

                img_mask_flat[err_map_to_thresh > pix_thresh_value] = False
                img_mask_flat = img_mask_flat.detach()

                cr = torch.sum(img_mask_flat > 0)
                total = torch.prod(torch.tensor(img_mask_flat.size()))

            tmp = (err_map * img_mask_flat).sum(dim=(1, 2, 3)) / (
                img_mask_flat.sum(dim=(1, 2, 3)) + 1e-7
            )

            loss_i = (err_map * img_mask_flat).sum(dim=(1, 2, 3)) / (
                img_mask_flat.sum(dim=(1, 2, 3)) + 1e-7
            )

            if self.weight_grad_match > 0:
                pred_tgt_disp = 1.0 / st_results.tgt_depths_pyr[i].view(-1, 1, h_i, w_i)
                plabel_tgt_disp = 1.0 / wt_results.tgt_depths_pyr[i].view(
                    -1, 1, h_i, w_i
                )

                R = plabel_tgt_disp.detach() - pred_tgt_disp[i]
                R = R.view(-1, 1, h_i, w_i)

                dx_R = consistency_loss._gradient_x(R)
                dy_R = consistency_loss._gradient_y(R)

                gm = (torch.abs(dx_R) + torch.abs(dy_R)) * img_mask_flat
                loss_i += args.weight_cr_gm * torch.mean(gm, dim=(1, 2, 3))

            if i > 0:
                loss_i *= 1 / (2**i)

            cr_loss += loss_i

            if log and i == 0:
                b, ns, c, img_h, img_w = wt_results.tgt_imgs_pyr[i].shape
                imgs_w = wt_results.tgt_imgs_pyr[i].reshape(b * ns, c, img_h, img_w)
                imgs_s = st_results.tgt_imgs_pyr[i].reshape(b * ns, c, img_h, img_w)

                log_consistency_regularization(
                    imgs_w,
                    imgs_s,
                    plabel,
                    pred,
                    img_mask_flat,
                    err_map_to_thresh,
                    err.mean(dim=(1, 2, 3)),
                    batch_mask,
                    i,
                    it,
                    args,
                )

        if self.weight_egomotion > 0:
            b3, ns3 = wt_results.T.size(0), wt_results.T.size(1)
            assert b3 == b and ns3 == ns
            pose_loss = (1.0 / (b * ns)) * torch.norm(
                wt_results.T.view(b * ns, -1).detach() - st_results.T.view(b * ns, -1),
                p=2,
                dim=(1,),
            )
            cr_loss += self.weight_egomotion * pose_loss

        cr_loss = (cr_loss * batch_mask).sum()
        num_filtered = batch_mask.sum()
        cr_loss = (
            cr_loss / num_filtered
            if num_filtered > 0
            else torch.tensor(0.0, device=err.device)
        )

        return cr_loss, num_filtered
