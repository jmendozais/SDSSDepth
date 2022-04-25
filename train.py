import os
import time
import random
import math
import gc

import configargparse
import dill
import numpy as np
import torch
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import functional as VF

import opts
from data import *
from data import transform as DT
import loss
from loss import *
from log import *
from model import *
from util.convert import matched_pairs_to_snippet
import eval.of_utils as of_utils
import eval.depth_eval_utils as depth_utils
import eval.kitti_depth_eval_utils as kitti_utils


class CosineWithRestarts:
    """A simple wrapper class for learning rate scheduling"""

    def __init__(self, optimizer, lr_max, lr_min, step_size):
        self._optimizer = optimizer
        self.lr_min = lr_min
        self.lr_max = lr_max
        self.step_size = step_size
        self._step = 0
        self.lr = lr_max

    def step(self):
        """Learning rate scheduling per step"""

        self._step += 1
        self.lr = self.lr_min + 0.5 * (self.lr_max - self.lr_min) * (
            1 + math.cos(math.pi * (self._step % self.step_size) / self.step_size)
        )

        for param_group in self._optimizer.param_groups:
            param_group["lr"] = self.lr

    def get_lr(self):
        return self.lr


def save_checkpoint(
    model, optimizer, epoch, save_chkp=True, save_best=False, teacher=None
):
    checkpoint = {}
    checkpoint["model"] = model
    checkpoint["teacher"] = teacher
    checkpoint["optimizer"] = optimizer
    checkpoint["epoch"] = epoch

    if save_chkp:
        torch.save(
            checkpoint,
            os.path.join(args.log, "checkpoint-{}.tar".format(epoch)),
            pickle_module=dill,
        )
        torch.save(
            checkpoint, os.path.join(args.log, "best_model_val.tar"), pickle_module=dill
        )


def train_step(data, model, optimizer, loss, teacher, args, it):
    wt_params = DT.sample_transform_by_label(args.weak_transform, args.batch_size)

    wt_data = DT.apply_transform(data, wt_params)
    DT.normalize(wt_data)

    DT.compute_image_pyramid(
        wt_data, downsample=args.downsample, full_res=args.loss_full_res
    )

    optimizer.zero_grad()

    if args.motion_burning_iters > 0:
        model.motion_weight = it * 1.0 / args.motion_burnin_iters
        model.motion_weight = min(max(0, 2 * model.motion_weight - 1), 1)

    model.is_initialized = it > args.init_iters

    wt_results = model(wt_data)

    batch_loss, err_pyr = compute_loss(
        model,
        wt_data,
        wt_results,
        args,
        norm_op,
        log_misc=True,
        log_depth=False,
        log_flow=False,
        it=it,
        epoch=epoch,
        writer=writer,
    )

    if args.weight_cr > 0:

        if teacher is not None:
            wt_results = teacher(wt_data)

        # Invert geometric subset of the composite weak transform
        wt_geometric_params = {"geometric": wt_params["geometric"].copy()}
        iwt_params = DT.invert_transform(wt_geometric_params)

        mask_cmp = wt_results.rigid_reps.masks_pyr[0]
        wt_target_data = {
            "pred_depth_snp": wt_results.depths_pyr[0].view(
                args.batch_size, args.seq_len, 1, args.height, args.width
            ),
            "mask_snp": matched_pairs_to_snippet(
                mask_cmp, mask_cmp, args.seq_len, args.bidirectional
            ),
            "error_snp": matched_pairs_to_snippet(
                err_pyr[0], err_pyr[0], args.seq_len, args.bidirectional
            ),
        }

        # Apply inverted transform
        iwt_data = DT.apply_transform(wt_target_data, iwt_params)
        data["pred_depth_snp"] = iwt_data["pred_depth_snp"]
        data["mask_snp"] = iwt_data["mask_snp"]
        data["error_snp"] = iwt_data["error_snp"]

        # Apply feature transforms
        if args.cr_feats_noise > 0.0:
            model.depth_net.dec.set_noise(args.cr_feats_noise)
        if args.cr_student_dp > 0.0:
            model.depth_net.dec.set_dropout(args.cr_student_dp)

        cr_loss, cr_count = 0, 0
        for k in range(args.cr_num_calls):
            st_params = DT.sample_transform_by_label(
                args.strong_transform, args.batch_size
            )
            st_data = DT.apply_transform(data, st_params)
            DT.normalize(st_data)

            DT.compute_image_pyramid(
                st_data, downsample=args.downsample, full_res=args.loss_full_res
            )

            st_results = model(st_data)

            cr_loss_i, cr_count_i = pl_loss.compute_pair(
                wt_results, st_data, st_results, args, it, is_training=True, log=False
            )

            cr_loss += cr_loss_i
            cr_count += cr_count_i

        cr_loss /= args.cr_num_calls

        weight_cr = args.weight_cr
        if args.cr_ramp_iters > it:
            weight_cr = args.weight_cr * it / args.cr_ramp_iters

        batch_loss += weight_cr * cr_loss

        # Disable feature transforms on student
        if args.cr_feats_noise > 0.0:
            model.depth_net.dec.set_noise(0.0)
        if args.cr_student_dp > 0.0:
            model.depth_net.dec.set_dropout(0.0)

        writer.add_scalars(
            "loss/batch", {"pl_tr": weight_cr * cr_loss.detach().item()}, it
        )
        writer.add_scalars(
            "loss/pl_count", {"pl_count_tr": cr_count.detach().item()}, it
        )

    batch_loss.backward()

    if args.clip_grad > 0:
        nn.utils.clip_grad_norm(parameters, args.clip_grad)

    optimizer.step()

    # Log iter stats and loss parameters
    if args.loss_params_type != loss.PARAMS_NONE:
        params_mean = 0
        for s in range(args.num_scales):
            if args.loss_params_type == loss.PARAMS_PREDICTED:
                params_mean += torch.mean(wt_results.extra_out_pyr[s]).item()
            elif args.loss_params_type == loss.PARAMS_VARIABLE:
                params_mean += torch.mean(norm_op.params_pyr[s]).item()

        params_mean /= args.num_scales
        writer.add_scalars(
            "loss/batch/params", {"mean": util.cpu_softplus(params_mean)}, it
        )

    mem = {}

    for j in range(torch.cuda.device_count()):
        mem["alloc-cuda:" + str(j)] = torch.cuda.memory_allocated(j)
        mem["reserv-cuda:" + str(j)] = torch.cuda.memory_reserved(j)

    writer.add_scalars("mem", mem, it)

    if args.debug_params and it % args.debug_step == 0:
        log_params(writer, [model, norm_op], it)

    if args.debug_metrics_train:
        tgt_depths_pyr = []
        for j in range(args.num_scales):
            tmp = [
                wt_results.depths_pyr[j][k * args.seq_len]
                for k in range(args.batch_size)
            ]
            tgt_depths_pyr.append(torch.stack(tmp))

        batch_metrics = depth_utils.compute_metrics_batch(
            tgt_depths_pyr[0],
            data["depth"].to(args.device),
            min_depth=args.min_depth,
            crop_eigen=args.crop_eigen,
        )

        for k, v in batch_metrics.items():
            batch_metrics[k] = np.mean(v)

        log_dict(writer, batch_metrics, "depth_metrics/batch", it=it)

    return batch_loss.detach().item()


def val_step(data, model, optimizer, loss, teacher, args, it):

    with torch.no_grad():
        for k in data.keys():
            data[k] = data[k].to(args.device)

        DT.normalize(data)
        DT.compute_image_pyramid(
            data, downsample=args.downsample, full_res=args.loss_full_res
        )

        wt_results = model(data)

        batch_loss, err_pyr = compute_loss(
            model,
            data,
            wt_results,
            args,
            norm_op,
            log_misc=False,
            log_depth=log_depth and i == log_idx[0],
            log_flow=log_flow and i == log_idx[0],
            it=it,
            epoch=epoch,
            writer=writer,
        )

        if args.weight_cr > 0:

            data["pred_depth_snp"] = wt_results.depths_pyr[0].view(
                args.batch_size, args.seq_len, 1, args.height, args.width
            )
            mask_cmp = wt_results.rigid_reps.masks_pyr[0]
            data["mask_snp"] = matched_pairs_to_snippet(
                mask_cmp, mask_cmp, args.seq_len, args.bidirectional
            )
            data["error_snp"] = matched_pairs_to_snippet(
                err_pyr[0], err_pyr[0], args.seq_len, args.bidirectional
            )

            DT.denormalize(data)

            cr_loss, cr_count = 0, 0
            for k in range(args.cr_num_calls):
                st_params = DT.sample_transform_by_label(
                    args.strong_transform, args.batch_size
                )
                st_data = DT.apply_transform(data, st_params)

                DT.normalize(st_data)
                DT.compute_image_pyramid(
                    st_data, downsample=args.downsample, full_res=args.loss_full_res
                )

                if teacher is not None:
                    st_results = teacher(st_data)
                else:
                    st_results = model(st_data)

                cr_loss_i, cr_count_i = pl_loss.compute_pair(
                    wt_results,
                    st_data,
                    st_results,
                    args,
                    it,
                    is_training=False,
                    log=i in log_idx,
                )

                cr_loss += cr_loss_i
                cr_count += cr_count_i

            cr_loss /= args.cr_num_calls

            weight_cr = args.weight_cr
            if args.cr_ramp_iters > it:
                weight_cr = args.weight_cr * it / args.cr_ramp_iters

            batch_loss += weight_cr * cr_loss

        metrics = None
        teacher_metrics = None

        if log_depth and "depth" in data.keys():
            tgt_depths = [
                wt_results.depths_pyr[0][k * args.seq_len]
                for k in range(args.batch_size)
            ]
            tgt_depths = torch.stack(tgt_depths)

            metrics = depth_utils.compute_metrics_batch(
                tgt_depths,
                data["depth"].to(args.device),
                min_depth=args.min_depth,
                max_depth=args.max_depth,
                crop_eigen=args.crop_eigen,
            )

            if teacher is not None:
                teacher_results = teacher(data)
                tgt_depths = [
                    teacher_results.depths_pyr[0][k * args.seq_len]
                    for k in range(args.batch_size)
                ]
                tgt_depths = torch.stack(tgt_depths)

                teacher_metrics = depth_utils.compute_metrics_batch(
                    tgt_depths,
                    data["depth"].to(args.device),
                    min_depth=args.min_depth,
                    max_depth=args.max_depth,
                    crop_eigen=args.crop_eigen,
                )

            if args.weight_cr:
                if cr_loss is not None:
                    writer.add_scalars(
                        "loss/batch",
                        {"pl_val": args.weight_cr * cr_loss.detach().item()},
                        it,
                    )
                    writer.add_scalars(
                        "loss/pl_count", {"pl_count_val": cr_count.detach().item()}, it
                    )

        if log_flow and "flow" in data.keys():
            # TODO: check if this logic works for seq_len = 2
            idx_fw_flow = [
                int((args.seq_len - 1) / 2 + (args.seq_len - 1) * i)
                for i in range(args.batch_size)
            ]  

            batch_metrics = of_utils.compute_of_metrics(
                wt_results.flows_pyr[0][idx_fw_flow], data["flow"].to(args.device)
            )

            util.accumulate_metrics(of_metrics, batch_metrics)

        return batch_loss.detach().item(), metrics, teacher_metrics


# Remove args usage
if __name__ == "__main__":

    random.seed(101)

    # Enable torch.autograd.set_detect_anomaly(True) for detailed logs. It 
    # slows down forward time x3 backward time x10

    torch.backends.cudnn.benchmark = True  # faster with fixed size inputs

    args = opts.parse_args()

    print("Arguments.")

    print(args)

    if args.loss_params_type == "var":
        args.loss_params_type = loss.PARAMS_VARIABLE
    elif args.loss_params_type == "fun":
        args.loss_params_type = loss.PARAMS_PREDICTED
    else:
        args.loss_params_type = loss.PARAMS_NONE

    if args.ec_mode == "alg":
        args.ec_mode = loss.EPIPOLAR_ALGEBRAIC
    elif args.ec_mode == "samp":
        args.ec_mode = loss.EPIPOLAR_SAMPSON

    if args.weight_rc > 0 or args.weight_tc > 0:
        if args.motion_mode != "by_pair" or args.bidirectional == False:
            raise NotImplementedError(
                "Rotation consistency cannot be enforced with motion mode different from by_pair or when bidirectional is disabled"
            )

    log_depth = args.weight_rigid > 0 and args.verbose != LOG_MINIMAL
    log_flow = args.weight_nonrigid > 0 and args.verbose != LOG_MINIMAL

    os.makedirs(args.log, exist_ok=True)

    train_set = create_dataset(
        args.dataset,
        args.dataset_dir,
        args.train_file,
        height=args.height,
        width=args.width,
        num_scales=args.num_scales,
        seq_len=args.seq_len,
        is_training=True,
        load_depth=args.debug_metrics_train,
        load_intrinsics=not args.learn_intrinsics,
    )

    val_set = create_dataset(
        args.dataset,
        args.dataset_dir,
        args.val_file,
        height=args.height,
        width=args.width,
        num_scales=args.num_scales,
        seq_len=args.seq_len,
        is_training=False,
        load_depth=log_depth,
        load_flow=log_flow,
        load_intrinsics=not args.learn_intrinsics,
    )

    pin_memory = args.device != "cpu"
    train_loader = torch.utils.data.DataLoader(
        train_set,
        args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_set,
        args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=pin_memory,
        drop_last=True,
    )

    norm_op = loss.create_norm_op(
        args.loss,
        num_recs=1,  # TODO: check parameter consistency
        height=args.height,
        width=args.width,
        params_type=args.loss_params_type,
        params_lb=args.loss_params_lb,
        aux_weight=args.loss_aux_weight,
    )

    norm_op.to(args.device)

    model = Model(
        params=args,
        num_extra_channels=norm_op.num_pred_params,
        dim_extra=norm_op.dim,
    )

    if args.data_parallel:
        model = nn.DataParallel(model, device_ids=[0, 1])

    if args.load_model is not None:
        model.load_state_dict(
            torch.load(args.load_model, map_location=args.device)["model"].state_dict()
        )

        model.batch_size = args.batch_size

    model = model.to(args.device)

    parameters = list(model.parameters()) + list(norm_op.parameters())

    if args.optimizer == "adam":
        optimizer = optim.Adam(parameters, lr=args.learning_rate)
    elif args.optimizer == "adamw":
        optimizer = optim.AdamW(parameters, lr=args.learning_rate)

    iters_by_epoch = len(train_loader)

    if args.scheduler == "step":
        lr_scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=args.scheduler_step_size, gamma=0.1, verbose=False
        )

    elif args.scheduler == "coswr":
        step_its = max(1, int(args.scheduler_step_size * iters_by_epoch))
        lr_scheduler = CosineWithRestarts(
            optimizer,
            lr_max=args.learning_rate,
            lr_min=args.learning_rate * 0.1,
            step_size=step_its,
        )

    else:
        step_its = max(1, int(args.scheduler_step_size * iters_by_epoch))

        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=step_its, eta_min=args.learning_rate * 0.1
        )

    teacher = None
    if args.weight_cr > 0:
        pl_loss = ConsistencyRegularizationLoss(args)
        if args.cr_mode in ["ema"]:
            teacher = EMA(model, args.mt_ema)

    writer = SummaryWriter(os.path.join(args.log, "tb_log"))
    start = time.perf_counter()

    it = 0

    start_training = time.perf_counter()
    start = start_training

    # Sample indices of examples for debugging.
    DEBUG_DATA_LIMIT = 6
    if args.debug_training:
        log_idx = np.random.choice(range(DEBUG_DATA_LIMIT), size=(3,), replace=False)
    else:
        log_idx = np.random.choice(range(len(val_loader) - 1), size=(3,), replace=False)

    for epoch in range(1, args.epochs + 1):
        start_epoch = time.perf_counter()

        model.train()
        if args.weight_cr > 0 and teacher is not None:
            teacher.train()

        train_loss = 0
        for i, data in enumerate(train_loader, 0):
            for k in data.keys():
                data[k] = data[k].to(args.device)

            train_loss += train_step(data, model, optimizer, loss, teacher, args, it)

            update_teacher = (
                (it + 1) % max(1, int(args.cr_teacher_update * iters_by_epoch))
            ) == 0
            if (args.weight_cr > 0) and (teacher is not None) and update_teacher:
                teacher.update()

            it += 1

            if args.scheduler == "coswr":
                lr_scheduler.step()

            if (args.debug_training or args.debug_valset) and i > DEBUG_DATA_LIMIT:
                break

        train_loss /= i

        if args.scheduler == "step":
            lr_scheduler.step()

        if epoch % args.val_step_size != 0:
            continue

        model.eval()
        if teacher is not None:
            teacher.eval()

        val_loss = 0
        best_val_loss = 1e10

        teacher_val_loss = 0

        of_metrics = {}
        depth_metrics_s = {}
        depth_metrics_t = {}

        for i, data in enumerate(val_loader, 0):
            batch_loss, metrics_s, metrics_t = val_step(
                data, model, optimizer, loss, teacher, args, it
            )

            val_loss += batch_loss

            util.accumulate_metrics(depth_metrics_s, metrics_s)
            if metrics_t is not None:
                util.accumulate_metrics(depth_metrics_t, metrics_t)

            if args.debug_training and i > DEBUG_DATA_LIMIT:
                break

        val_loss /= i
        losses = {"tr": train_loss, "val": val_loss}
        writer.add_scalars("loss", losses, epoch)

        # Log depth and optical flow metrics
        metrics = [{"Ep": epoch}, losses]

        if log_depth:
            for k, v in depth_metrics_s.items():
                depth_metrics_s[k] = np.mean(v)
            log_dict(writer, depth_metrics_s, "depth_metrics_s/epoch", it=epoch)

            if depth_metrics_t is not None:
                for k, v in depth_metrics_t.items():
                    depth_metrics_t[k] = np.mean(v)
                log_dict(writer, depth_metrics_t, "depth_metrics_t/epoch", it=epoch)

            metrics += [depth_metrics_s]
            metrics += [depth_metrics_t]

        if log_flow:
            for k, v in of_metrics.items():
                of_metrics[k] = np.mean(v)

            log_of_metrics(writer, of_metrics, epoch=epoch)
            metrics += [of_metrics]

        # Save checkpoint
        has_improved = val_loss < best_val_loss - 1e-6
        save_chkp = epoch % args.ckp_freq == 0

        if save_chkp or has_improved:
            save_checkpoint(
                model, optimizer, epoch, save_chkp, has_improved, teacher=teacher
            )

        elapsed_time = time.perf_counter() - start_training
        epoch_time = time.perf_counter() - start_epoch
        avg_epoch_time = elapsed_time // epoch
        remaining_time = (args.epochs - epoch) * avg_epoch_time
        expected_training_time = avg_epoch_time * args.epochs

        time_metrics = {
            "epoch": util.human_time(epoch_time),
            "elap": util.human_time(elapsed_time),
            "rem": util.human_time(remaining_time),
            "tot": util.human_time(expected_training_time),
        }
        metrics.append(time_metrics)

        print_metric_groups(metrics)

    writer.close()

    print("training time", time.perf_counter() - start_training)
