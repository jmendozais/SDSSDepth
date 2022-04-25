from .consistency_loss import *


# loss aggregator
def compute_loss(
    model,
    data,
    results,
    params,
    norm_op,
    log_misc,
    log_depth,
    log_flow,
    it,
    epoch,
    writer,
):

    rec_loss, res, err, pooled_err = representation_consistency(
        results=results, params=params, norm=norm_op, return_residuals=True
    )

    batch_loss = 0
    batch_loss += rec_loss

    device = data[("color", 0)].device
    ds = torch.tensor(0, dtype=torch.float, device=device)
    fs = torch.tensor(0, dtype=torch.float, device=device)
    ofc = torch.tensor(0, dtype=torch.float, device=device)

    for i in range(params.num_scales):
        b, s, c, h, w = data[("color", i)].size()
        imgs = data[("color", i)].view(-1, c, h, w)
        scale_weight = 1.0 / (2**i)

        if params.weight_ds > 0:
            loss_term = normalized_smoothness(
                results.disps_pyr[i], imgs, order=2, alpha=params.ds_alpha
            )
            ds += scale_weight * params.weight_ds * loss_term

        if params.weight_fs > 0:
            loss_term = flow_smoothness(
                results.flows_pyr[i], imgs, order=1, alpha=params.fs_alpha
            )
            fs += scale_weight * params.weight_fs * loss_term

        if params.weight_ofc > 0 and it > params.init_iters:
            loss_term, rigid_mask = flow_consistency(
                results.rigid_flows_pyr[i], results.flows_pyr[i]
            )
            ofc += scale_weight * params.weight_ofc * loss_term
            results.rigid_mask_pyr.append(rigid_mask)

    batch_loss += ofc
    batch_loss += ds
    batch_loss += fs

    rc = torch.zeros((1,), dtype=torch.float)
    if params.weight_rc > 0:
        rc = params.weight_rc * rotation_consistency(results)
        batch_loss += rc

    tc = torch.zeros((1,), dtype=torch.float)
    if params.weight_tc > 0:
        tc = params.weight_tc * translation_consistency(results)
        batch_loss += tc

    mr = torch.zeros((1,), dtype=torch.float)
    if params.weight_msm > 0 or params.weight_msp > 0:
        if params.weight_rigid == 1:
            raise Exception("Motion regularization not allowed for depth only mode")

        mr = motion_regularization(
            results.res_flows_pyr,
            results.total_flows_pyr,
            params.weight_msm,
            params.weight_msp,
        )
        batch_loss += mr

    ec_loss = torch.zeros((1,), dtype=torch.float)
    if params.weight_ec > 0:
        coords = [apply_flow.coords for apply_flow in model.ms_applyflow]
        ec_loss = epipolar_constraint(coords, results, params.ec_mode)
        ec_loss *= params.weight_ec
        batch_loss += ec_loss

    if log_depth or log_flow:
        log_results(
            writer,
            model.seq_len,
            results,
            res,
            err,
            pooled_err,
            norm_op,
            epoch=epoch,
            log_depth=log_depth,
            log_flow=log_flow,
            data=data,
        )

    if log_misc:
        writer.add_scalars(
            "loss/batch",
            {
                "rec": rec_loss.item(),
                "ds": ds.item(),
                "fsm": fs.item(),
                "ofc": ofc.item(),
                "ec": ec_loss.item(),
                "rc": rc.item(),
                "tc": tc.item(),
                "mr": mr.item(),
                "s1_all": batch_loss.item(),
            },
            it,
        )

        depth_means = dict()
        for i in range(len(results.tgt_depths_pyr)):
            depth_means["scale-1/{}".format(2**i)] = results.tgt_depths_pyr[i][
                0, 0
            ].mean()

        writer.add_scalars("depth/mean", depth_means, it)

        for j in range(len(results.K_pyr)):
            writer.add_scalars(
                "intrinsics/{}".format(j),
                {
                    "fx": results.K_pyr[j][0, 0, 0].item(),
                    "fy": results.K_pyr[j][0, 1, 1].item(),
                },
                it,
            )

        A = transforms3D.rotation_conversions.matrix_to_euler_angles(
            results.T.view(-1, 4, 4)[:, :3, :3], "XYZ"
        )
        writer.add_scalars(
            "pose",
            {"e1": A[0, 0].item(), "e2": A[0, 1].item(), "e3": A[0, 2].item()},
            it,
        )

    return batch_loss, err
