import torch
import torch.nn.functional as F


def normalize_coords(coords):
    _, _, h, w = coords.shape
    coords[:, 0, :, :] /= w - 1
    coords[:, 1, :, :] /= h - 1
    coords = (coords - 0.5) * 2
    return coords


def grid_sample(imgs, pix_coords, return_mask=False):
    """
    Args:
        pix_coords: pix coords normalized to [-1, 1]

    """
    _, _, h, w = imgs.size()
    pix_coords = pix_coords.permute(0, 2, 3, 1)

    if return_mask:
        mask = torch.logical_and(
            pix_coords[:, :, :, 0] > -1, pix_coords[:, :, :, 1] > -1
        )
        mask = torch.logical_and(mask, pix_coords[:, :, :, 0] < 1)
        mask = torch.logical_and(mask, pix_coords[:, :, :, 1] < 1)
        # Monodepth2 used the default of old pytorch: align corners = true
        return (
            F.grid_sample(imgs, pix_coords, padding_mode="border", align_corners=False),
            mask,
        )
    else:
        return F.grid_sample(
            imgs, pix_coords, padding_mode="border", align_corners=False
        )
