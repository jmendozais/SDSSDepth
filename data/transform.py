from data.dataset import Dataset
import torch
import torch.utils.data as data
import os
import time
import glob
import numpy as np
import numpy.random as rng
from skimage import io, transform
import torch.nn.functional as F
from torchvision.transforms import functional as VF
import math

from . import lanczos

from PIL import Image
import cv2
from turbojpeg import TurboJPEG as JPEG

#from memory_profiler import profile

'''
A dataset is compose of a set of clips. For each clip there is a directory that contains its frames.

Input: 
    A list with the clip names for training/validation/testing:
        We create a dataset instance for each clip of the validation/test set, and one instance for the training set.
'''

# TODO: do a last test with nocoloraug photometric error and remove if it is not benefitial

# Color augmentation. Parameters borrowed from monodepth2
weak_transform_params = {
    'color': {
        'brightness': (0.8, 1.2),
        'contrast': (0.8, 1.2),
        'saturation': (0.8, 1.2),
        'hue': (-0.1, 0.1),
    },
    'scale_crop': {
        'max_offset': 0.25
    },
    'flip': True,
}

wt_v0_1_nosc = {
    'color': {
        'brightness': (0.8, 1.2),
        'contrast': (0.8, 1.2),
        'saturation': (0.8, 1.2),
        'hue': (-0.1, 0.1),
    },
    'scale_crop': {
        'max_offset': 0.0
    },
    'flip': True,
}

wt_v0_1_nosc_rlc0_4 = {
    'color': {
        'brightness': (0.8, 1.2),
        'contrast': (0.8, 1.2),
        'saturation': (0.8, 1.2),
        'hue': (-0.1, 0.1),
    },
    'scale_crop': {
        'max_offset': 0.0
    },
    'flip': True,
    'random_local': {
        'max_area': 0.4
    }
}

flip_params = {
    'color': {
        'brightness': (1.0, 1.0),
        'contrast': (1.0, 1.0),
        'saturation': (1.0, 1.0),
        'hue': (-0.0, 0.0),
    },
    'scale_crop': {
        'max_offset': 0.0
    },
    'flip': True,
}

bright0_2_flip_params = {
    'color': {
        'brightness': (0.8, 1.2),
        'contrast': (1.0, 1.0),
        'saturation': (1.0, 1.0),
        'hue': (-0.0, 0.0),
    },
    'scale_crop': {
        'max_offset': 0.0
    },
    'flip': True,
}

bright0_4_flip_params = {
    'color': {
        'brightness': (0.6, 1.4),
        'contrast': (1.0, 1.0),
        'saturation': (1.0, 1.0),
        'hue': (-0.0, 0.0),
    },
    'scale_crop': {
        'max_offset': 0.0
    },
    'flip': True,
}

bright0_6_flip_params = {
    'color': {
        'brightness': (0.4, 1.6),
        'contrast': (1.0, 1.0),
        'saturation': (1.0, 1.0),
        'hue': (-0.0, 0.0),
    },
    'scale_crop': {
        'max_offset': 0.0
    },
    'flip': True,
}

bright0_5__2_flip_params = {
    'color': {
        'brightness': (0.5, 2),
        'contrast': (1.0, 1.0),
        'saturation': (1.0, 1.0),
        'hue': (-0.0, 0.0),
    },
    'scale_crop': {
        'max_offset': 0.0
    },
    'flip': True,
}

bright0_33__3_flip_params = {
    'color': {
        'brightness': (0.33, 3),
        'contrast': (1.0, 1.0),
        'saturation': (1.0, 1.0),
        'hue': (-0.0, 0.0),
    },
    'scale_crop': {
        'max_offset': 0.0
    },
    'flip': True,
}

bright0_25__4_flip_params = {
    'color': {
        'brightness': (0.25, 4),
        'contrast': (1.0, 1.0),
        'saturation': (1.0, 1.0),
        'hue': (-0.0, 0.0),
    },
    'scale_crop': {
        'max_offset': 0.0
    },
    'flip': True,
}

contrast0_2_flip_params = {
    'color': {
        'brightness': (0.8, 1.2),
        'contrast': (1.0, 1.0),
        'saturation': (1.0, 1.0),
        'hue': (-0.0, 0.0),
    },
    'scale_crop': {
        'max_offset': 0.0
    },
    'flip': True,
}
contrast0_4_flip_params = {
    'color': {
        'brightness': (1.0, 1.0),
        'contrast': (0.6, 1.4),
        'saturation': (1.0, 1.0),
        'hue': (-0.0, 0.0),
    },
    'scale_crop': {
        'max_offset': 0.0
    },
    'flip': True,
}

contrast0_6_flip_params = {
    'color': {
        'brightness': (1.0, 1.0),
        'contrast': (0.4, 1.6),
        'saturation': (1.0, 1.0),
        'hue': (-0.0, 0.0),
    },
    'scale_crop': {
        'max_offset': 0.0
    },
    'flip': True,
}

contrast0_5__2_flip_params = {
    'color': {
        'brightness': (1.0, 1.0),
        'contrast': (0.5, 2.0),
        'saturation': (1.0, 1.0),
        'hue': (-0.0, 0.0),
    },
    'scale_crop': {
        'max_offset': 0.0
    },
    'flip': True,
}

contrast0_33__4_flip_params = {
    'color': {
        'brightness': (1.0, 1.0),
        'contrast': (0.33, 3.0),
        'saturation': (1.0, 1.0),
        'hue': (-0.0, 0.0),
    },
    'scale_crop': {
        'max_offset': 0.0
    },
    'flip': True,
}

contrast0_25__4_flip_params = {
    'color': {
        'brightness': (1.0, 1.0),
        'contrast': (0.25, 4.0),
        'saturation': (1.0, 1.0),
        'hue': (-0.0, 0.0),
    },
    'scale_crop': {
        'max_offset': 0.0
    },
    'flip': True,
}

sat0_2_flip_params = {
    'color': {
        'brightness': (1.0, 1.0),
        'contrast': (1.0, 1.0),
        'saturation': (0.8, 1.2),
        'hue': (-0.0, 0.0),
    },
    'scale_crop': {
        'max_offset': 0.0
    },
    'flip': True,
}

sat0_4_flip_params = {
    'color': {
        'brightness': (1.0, 1.0),
        'contrast': (1.0, 1.0),
        'saturation': (0.6, 1.4),
        'hue': (-0.0, 0.0),
    },
    'scale_crop': {
        'max_offset': 0.0
    },
    'flip': True,
}
sat0_6_flip_params = {
    'color': {
        'brightness': (1.0, 1.0),
        'contrast': (1.0, 1.0),
        'saturation': (0.4, 1.6),
        'hue': (-0.0, 0.0),
    },
    'scale_crop': {
        'max_offset': 0.0
    },
    'flip': True,
}

sat0_5__2_flip_params = {
    'color': {
        'brightness': (1.0, 1.0),
        'contrast': (1.0, 1.0),
        'saturation': (0.5, 2.0),
        'hue': (-0.0, 0.0),
    },
    'scale_crop': {
        'max_offset': 0.0
    },
    'flip': True,
}

sat0_33__3_flip_params = {
    'color': {
        'brightness': (1.0, 1.0),
        'contrast': (1.0, 1.0),
        'saturation': (0.33, 3.0),
        'hue': (-0.0, 0.0),
    },
    'scale_crop': {
        'max_offset': 0.0
    },
    'flip': True,
}

sat0_25__4_flip_params = {
    'color': {
        'brightness': (1.0, 1.0),
        'contrast': (1.0, 1.0),
        'saturation': (0.25, 4.0),
        'hue': (-0.0, 0.0),
    },
    'scale_crop': {
        'max_offset': 0.0
    },
    'flip': True,
}

hue_0_1_flip_params = {
    'color': {
        'brightness': (1.0, 1.0),
        'contrast': (1.0, 1.0),
        'saturation': (1.0, 1.0),
        'hue': (-0.1, 0.1),
    },
    'scale_crop': {
        'max_offset': 0.0
    },
    'flip': True,
}

hue_0_2_flip_params = {
    'color': {
        'brightness': (1.0, 1.0),
        'contrast': (1.0, 1.0),
        'saturation': (1.0, 1.0),
        'hue': (-0.2, 0.2),
    },
    'scale_crop': {
        'max_offset': 0.0
    },
    'flip': True,
}

hue_0_3_flip_params = {
    'color': {
        'brightness': (1.0, 1.0),
        'contrast': (1.0, 1.0),
        'saturation': (1.0, 1.0),
        'hue': (-0.3, 0.3),
    },
    'scale_crop': {
        'max_offset': 0.0
    },
    'flip': True,
}

hue_0_4_flip_params = {
    'color': {
        'brightness': (1.0, 1.0),
        'contrast': (1.0, 1.0),
        'saturation': (1.0, 1.0),
        'hue': (-0.4, 0.4),
    },
    'scale_crop': {
        'max_offset': 0.0
    },
    'flip': True,
}

wt_v0_2_params = {
    'color': {
        'brightness': (0.33, 3),
        'contrast': (0.4, 1.6),
        'saturation': (1.0, 1.0),
        'hue': (-0.0, 0.0),
    },
    'scale_crop': {
        'max_offset': 0.0
    },
    'flip': True,
}

wt_v0_2_re = {
    'color': {
        'brightness': (0.33, 3),
        'contrast': (0.4, 1.6),
        'saturation': (1.0, 1.0),
        'hue': (-0.0, 0.0),
    },
    'scale_crop': {
        'max_offset': 0.0
    },
    'flip': True,
    're': {
        'max_area': 0.4
    }
}

wt_v0_2_re0_2 = {
    'color': {
        'brightness': (0.33, 3),
        'contrast': (0.4, 1.6),
        'saturation': (1.0, 1.0),
        'hue': (-0.0, 0.0),
    },
    'scale_crop': {
        'max_offset': 0.0
    },
    'flip': True,
    're': {
        'max_area': 0.2
    }
}

wt_v0_2_re0_1 = {
    'color': {
        'brightness': (0.33, 3),
        'contrast': (0.4, 1.6),
        'saturation': (1.0, 1.0),
        'hue': (-0.0, 0.0),
    },
    'scale_crop': {
        'max_offset': 0.0
    },
    'flip': True,
    're': {
        'max_area': 0.1
    }
}

wt_v0_2_re0_05 = {
    'color': {
        'brightness': (0.33, 3),
        'contrast': (0.4, 1.6),
        'saturation': (1.0, 1.0),
        'hue': (-0.0, 0.0),
    },
    'scale_crop': {
        'max_offset': 0.0
    },
    'flip': True,
    're': {
        'max_area': 0.05
    }
}

wt_v0_2_hue0_1_params = {
    'color': {
        'brightness': (0.33, 3),
        'contrast': (0.4, 1.6),
        'saturation': (1.0, 1.0),
        'hue': (-0.1, 0.1),
    },
    'scale_crop': {
        'max_offset': 0.0
    },
    'flip': True,
}

wt_v0_2_hue0_2_params = {
    'color': {
        'brightness': (0.33, 3),
        'contrast': (0.4, 1.6),
        'saturation': (1.0, 1.0),
        'hue': (-0.2, 0.2),
    },
    'scale_crop': {
        'max_offset': 0.0
    },
    'flip': True,
}

wt_v0_2_hue0_3_params = {
    'color': {
        'brightness': (0.33, 3),
        'contrast': (0.4, 1.6),
        'saturation': (1.0, 1.0),
        'hue': (-0.3, 0.3),
    },
    'scale_crop': {
        'max_offset': 0.0
    },
    'flip': True,
}

wt_v0_2_sat0_6_params = {
    'color': {
        'brightness': (0.33, 3),
        'contrast': (0.4, 1.6),
        'saturation': (0.4, 1.6),
        'hue': (-0, 0),
    },
    'scale_crop': {
        'max_offset': 0.0
    },
    'flip': True,
}

wt_v0_2_sat0_5_2_params = {
    'color': {
        'brightness': (0.33, 3),
        'contrast': (0.4, 1.6),
        'saturation': (0.5, 2.0),
        'hue': (-0, 0),
    },
    'scale_crop': {
        'max_offset': 0.0
    },
    'flip': True,
}

wt_v0_2_sat0_33_3_params = {
    'color': {
        'brightness': (0.33, 3),
        'contrast': (0.4, 1.6),
        'saturation': (0.33, 3.0),
        'hue': (-0, 0),
    },
    'scale_crop': {
        'max_offset': 0.0
    },
    'flip': True,
}

wt_v0_2_eq0_1_params = {
    'color': {
        'brightness': (0.33, 3),
        'contrast': (0.4, 1.6),
        'saturation': (1.0, 1.0),
        'hue': (-0.0, 0.0),
        'equalize': 0.1,

    },
    'scale_crop': {
        'max_offset': 0.0
    },
    'flip': True,
}

eq0_5_params = {
    'color': {
        'brightness': (1.0, 1.0),
        'contrast': (1.0, 1.0),
        'saturation': (1.0, 1.0),
        'hue': (-0.0, 0.0),
        'equalize': 0.5,

    },
    'scale_crop': {
        'max_offset': 0.0
    },
    'flip': True,
}

wt_v0_2_eq0_2_params = {
    'color': {
        'brightness': (0.33, 3),
        'contrast': (0.4, 1.6),
        'saturation': (1.0, 1.0),
        'hue': (-0.0, 0.0),
        'equalize': 0.2,

    },
    'scale_crop': {
        'max_offset': 0.0
    },
    'flip': True,
}

wt_v0_2_eq0_4_params = {
    'color': {
        'brightness': (0.33, 3),
        'contrast': (0.4, 1.6),
        'saturation': (1.0, 1.0),
        'hue': (-0.0, 0.0),
        'equalize': 0.4,

    },
    'scale_crop': {
        'max_offset': 0.0
    },
    'flip': True,
}

ac0_5_params = {
    'color': {
        'brightness': (1.0, 1.0),
        'contrast': (1.0, 1.0),
        'saturation': (1.0, 1.0),
        'hue': (-0.0, 0.0),
        'autocontrast': 0.5,

    },
    'scale_crop': {
        'max_offset': 0.0
    },
    'flip': True,
}

wt_v0_2_ac0_1_params = {
    'color': {
        'brightness': (0.33, 3),
        'contrast': (0.4, 1.6),
        'saturation': (1.0, 1.0),
        'hue': (-0.0, 0.0),
        'autocontrast': 0.1,

    },
    'scale_crop': {
        'max_offset': 0.0
    },
    'flip': True,
}

wt_v0_2_ac0_2_params = {
    'color': {
        'brightness': (0.33, 3),
        'contrast': (0.4, 1.6),
        'saturation': (1.0, 1.0),
        'hue': (-0.0, 0.0),
        'autocontrast': 0.2,

    },
    'scale_crop': {
        'max_offset': 0.0
    },
    'flip': True,
}

wt_v0_2_ac0_4_params = {
    'color': {
        'brightness': (0.33, 3),
        'contrast': (0.4, 1.6),
        'saturation': (1.0, 1.0),
        'hue': (-0.0, 0.0),
        'autocontrast': 0.4,

    },
    'scale_crop': {
        'max_offset': 0.0
    },
    'flip': True,
}

po0_5__1 = {
    'color': {
        'brightness': (1, 1),
        'contrast': (1, 1),
        'saturation': (1.0, 1.0),
        'hue': (-0.0, 0.0),
        'posterize': (0.5, 1)
    },
    'scale_crop': {
        'max_offset': 0.0
    },
    'flip': True,
}

po0_5__2 = {
    'color': {
        'brightness': (1, 1),
        'contrast': (1, 1),
        'saturation': (1.0, 1.0),
        'hue': (-0.0, 0.0),
        'posterize': (0.5, 2)
    },
    'scale_crop': {
        'max_offset': 0.0
    },
    'flip': True,
}

po0_5__3 = {
    'color': {
        'brightness': (1, 1),
        'contrast': (1, 1),
        'saturation': (1.0, 1.0),
        'hue': (-0.0, 0.0),
        'posterize': (0.5, 3)
    },
    'scale_crop': {
        'max_offset': 0.0
    },
    'flip': True,
}

po0_5__4 = {
    'color': {
        'brightness': (1, 1),
        'contrast': (1, 1),
        'saturation': (1.0, 1.0),
        'hue': (-0.0, 0.0),
        'posterize': (0.5, 4)
    },
    'scale_crop': {
        'max_offset': 0.0
    },
    'flip': True,
}

wt_v0_2_po0_1__4 = {
    'color': {
        'brightness': (0.33, 3),
        'contrast': (0.4, 1.6),
        'saturation': (1.0, 1.0),
        'hue': (-0.0, 0.0),
        'posterize': (0.1, 4)
    },
    'scale_crop': {
        'max_offset': 0.0
    },
    'flip': True,
}

wt_v0_2_po0_2__4 = {
    'color': {
        'brightness': (0.33, 3),
        'contrast': (0.4, 1.6),
        'saturation': (1.0, 1.0),
        'hue': (-0.0, 0.0),
        'posterize': (0.2, 4)
    },
    'scale_crop': {
        'max_offset': 0.0
    },
    'flip': True,
}

wt_v0_2_po0_4__4 = {
    'color': {
        'brightness': (0.33, 3),
        'contrast': (0.4, 1.6),
        'saturation': (1.0, 1.0),
        'hue': (-0.0, 0.0),
        'posterize': (0.4, 4)
    },
    'scale_crop': {
        'max_offset': 0.0
    },
    'flip': True,
}


sharp0_1_flip = {
    'color': {
        'brightness': (1, 1),
        'contrast': (1, 1),
        'saturation': (1, 1),
        'hue': (-0.0, 0.0),
        'sharpness': (1.0, 1.1)
    },
    'scale_crop': {
        'max_offset': 0.0
    },
    'flip': True,
}


sharp0_2_flip = {
    'color': {
        'brightness': (1, 1),
        'contrast': (1, 1),
        'saturation': (1, 1),
        'hue': (-0.0, 0.0),
        'sharpness': (1.0, 1.2)
    },
    'scale_crop': {
        'max_offset': 0.0
    },
    'flip': True,
}


sharp0_4_flip = {
    'color': {
        'brightness': (1, 1),
        'contrast': (1, 1),
        'saturation': (1, 1),
        'hue': (-0.0, 0.0),
        'sharpness': (1.0, 1.4)
    },
    'scale_crop': {
        'max_offset': 0.0
    },
    'flip': True,
}

sharp0_6_flip = {
    'color': {
        'brightness': (1, 1),
        'contrast': (1, 1),
        'saturation': (1, 1),
        'hue': (-0.0, 0.0),
        'sharpness': (1.0, 1.6)
    },
    'scale_crop': {
        'max_offset': 0.0
    },
    'flip': True,
}

wt_v0_2_rlc0_4 = {
    'color': {
        'brightness': (0.33, 3),
        'contrast': (0.4, 1.6),
        'saturation': (1.0, 1.0),
        'hue': (-0.0, 0.0),
    },
    'scale_crop': {
        'max_offset': 0.0
    },
    'flip': True,
    'random_local': {
        'max_area': 0.4
    }
}

params_by_label = {
    'fliponly': flip_params,
    'bright0.2': bright0_2_flip_params,
    'bright0.4': bright0_4_flip_params,
    'bright0.6': bright0_6_flip_params,
    'bright0.5_2': bright0_5__2_flip_params,
    'bright0.33_3': bright0_33__3_flip_params,
    'bright0.25_4': bright0_25__4_flip_params,
    'contrast0.2': contrast0_2_flip_params,
    'contrast0.4': contrast0_4_flip_params,
    'contrast0.6': contrast0_6_flip_params,
    'contrast0.5_2': contrast0_5__2_flip_params,
    'contrast0.33_3': contrast0_33__4_flip_params,
    'contrast0.25_4': contrast0_25__4_flip_params,
    'sat0.2': sat0_2_flip_params,
    'sat0.4': sat0_4_flip_params,
    'sat0.6': sat0_6_flip_params,
    'sat0.5_2': sat0_5__2_flip_params,
    'sat0.33_3': sat0_33__3_flip_params,
    'sat0.25_4': sat0_25__4_flip_params,
    'wt_v0.1_nosc': wt_v0_1_nosc,
    'hue0.1': hue_0_1_flip_params,
    'hue0.2': hue_0_2_flip_params,
    'hue0.3': hue_0_3_flip_params,
    'hue0.4': hue_0_4_flip_params,
    'sharp0.1': sharp0_1_flip,
    'sharp0.2': sharp0_2_flip,
    'sharp0.4': sharp0_4_flip,
    'sharp0.6': sharp0_6_flip,
    'wt_v0.2_hue0.1': wt_v0_2_hue0_1_params,
    'wt_v0.2_hue0.2': wt_v0_2_hue0_2_params,
    'wt_v0.2_hue0.3': wt_v0_2_hue0_3_params,
    'wt_v0.2_sat0.6': wt_v0_2_sat0_6_params,
    'wt_v0.2_sat0.5_2': wt_v0_2_sat0_5_2_params,
    'wt_v0.2_sat0.33_3': wt_v0_2_sat0_33_3_params,
    'wt_v0.2_eq0.1': wt_v0_2_eq0_1_params,
    'wt_v0.2_eq0.2': wt_v0_2_eq0_2_params,
    'wt_v0.2_eq0.4': wt_v0_2_eq0_4_params,
    'wt_v0.2_ac0.1': wt_v0_2_ac0_1_params,
    'wt_v0.2_ac0.2': wt_v0_2_ac0_2_params,
    'wt_v0.2_ac0.4': wt_v0_2_ac0_4_params,
    'ac0.5': ac0_5_params,
    'eq0.5': eq0_5_params,
    'po0.5_1': po0_5__1,
    'po0.5_2': po0_5__2,
    'po0.5_3': po0_5__3,
    'po0.5_4': po0_5__4,
    'wt_v0.2_po0.1_4': wt_v0_2_po0_1__4,
    'wt_v0.2_po0.2_4': wt_v0_2_po0_2__4,
    'wt_v0.2_po0.4_4': wt_v0_2_po0_4__4,
    'wt_v0.2_re': wt_v0_2_re,
    'wt_v0.2_re0.2': wt_v0_2_re0_2,
    'wt_v0.2_re0.1': wt_v0_2_re0_1,
    'wt_v0.2_re0.05': wt_v0_2_re0_05,
    'wt_v0.2_rlc0.4': wt_v0_2_rlc0_4,
    'wt_v0.1_nosc_rlc0.4': wt_v0_1_nosc_rlc0_4,
    'wt_v0.1': weak_transform_params,

    'wt_v0.1_nosc_rl0.2': {
        'color': {
            'brightness': (0.8, 1.2),
            'contrast': (0.8, 1.2),
            'saturation': (0.8, 1.2),
            'hue': (-0.1, 0.1),
        },
        'scale_crop': {
            'max_offset': 0.0
        },
        'flip': True,
        'random_local': {
            'max_area': 0.2
        }
    },

    'wt_v0.1_nosc_rl0.3': {
        'color': {
            'brightness': (0.8, 1.2),
            'contrast': (0.8, 1.2),
            'saturation': (0.8, 1.2),
            'hue': (-0.1, 0.1),
        },
        'scale_crop': {
            'max_offset': 0.0
        },
        'flip': True,
        'random_local': {
            'max_area': 0.3
        }
    },

    'wt_v0.1_nosc_rl0.4': {
        'color': {
            'brightness': (0.8, 1.2),
            'contrast': (0.8, 1.2),
            'saturation': (0.8, 1.2),
            'hue': (-0.1, 0.1),
        },
        'scale_crop': {
            'max_offset': 0.0
        },
        'flip': True,
        'random_local': {
            'max_area': 0.4
        }
    },

    'wt_v0.2': {
        'non_geometric': {
            'brightness': [['uniform_multiplier', 0.33, 3]],
            'contrast': [['uniform_multiplier', 0.4, 1.6]],
            'saturation': [['uniform_multiplier', 1.0, 1.0]],
            'hue': [['uniform', -0.0, 0.0]],
        },

        'geometric': {
            'flip': [['bernoulli', 0.5]]
        }
    },

    'wt_v0.2_sc1.15': {
        'non_geometric': {
            'brightness': [['uniform_multiplier', 0.33, 3]],
            'contrast': [['uniform_multiplier', 0.4, 1.6]],
            'saturation': [['uniform_multiplier', 1.0, 1.0]],
            'hue': [['uniform', -0.0, 0.0]],
        },

        'geometric': {
            'scale_crop': [['uniform', 1.0, 1.15],
                           ['uniform', 1.0, 1.15],
                           ['uniform', 0.0, 1.0],
                           ['uniform', 0.0, 1.0]],
            'flip': [['bernoulli', 0.5]],
        },
    },

    'wt_v0.2_sc1.2': {
        'non_geometric': {
            'brightness': [['uniform_multiplier', 0.33, 3]],
            'contrast': [['uniform_multiplier', 0.4, 1.6]],
            'saturation': [['uniform_multiplier', 1.0, 1.0]],
            'hue': [['uniform', -0.0, 0.0]],
        },

        'geometric': {
            'scale_crop': [['uniform', 1.0, 1.2],
                           ['uniform', 1.0, 1.2],
                           ['uniform', 0.0, 1.0],
                           ['uniform', 0.0, 1.0]],
            'flip': [['bernoulli', 0.5]],
        },
    },

    'wt_v0.2_sc1.4': {
        'non_geometric': {
            'brightness': [['uniform_multiplier', 0.33, 3]],
            'contrast': [['uniform_multiplier', 0.4, 1.6]],
            'saturation': [['uniform_multiplier', 1.0, 1.0]],
            'hue': [['uniform', -0.0, 0.0]],
        },

        'geometric': {
            'scale_crop': [['uniform', 1.0, 1.4],
                           ['uniform', 1.0, 1.4],
                           ['uniform', 0.0, 1.0],
                           ['uniform', 0.0, 1.0]],
            'flip': [['bernoulli', 0.5]],
        },
    },

    'wt_v0.2_sc1.5': {
        'non_geometric': {
            'brightness': [['uniform_multiplier', 0.33, 3]],
            'contrast': [['uniform_multiplier', 0.4, 1.6]],
            'saturation': [['uniform_multiplier', 1.0, 1.0]],
            'hue': [['uniform', -0.0, 0.0]],
        },

        'geometric': {
            'scale_crop': [['uniform', 1.0, 1.5],
                           ['uniform', 1.0, 1.5],
                           ['uniform', 0.0, 1.0],
                           ['uniform', 0.0, 1.0]],
            'flip': [['bernoulli', 0.5]],
        },
    },

    'wt_v0.2_sc1.1': {
        'non_geometric': {
            'brightness': [['uniform_multiplier', 0.33, 3]],
            'contrast': [['uniform_multiplier', 0.4, 1.6]],
            'saturation': [['uniform_multiplier', 1.0, 1.0]],
            'hue': [['uniform', -0.0, 0.0]],
        },

        'geometric': {
            'scale_crop': [['uniform', 1.0, 1.1],
                           ['uniform', 1.0, 1.1],
                           ['uniform', 0.0, 1.0],
                           ['uniform', 0.0, 1.0]],
            'flip': [['bernoulli', 0.5]],
        },
    },

    'wt_v0.2_sc1.01': {
        'non_geometric': {
            'brightness': [['uniform_multiplier', 0.33, 3]],
            'contrast': [['uniform_multiplier', 0.4, 1.6]],
            'saturation': [['uniform_multiplier', 1.0, 1.0]],
            'hue': [['uniform', -0.0, 0.0]],
        },

        'geometric': {
            'scale_crop': [['uniform', 1.0, 1.01],
                           ['uniform', 1.0, 1.01],
                           ['uniform', 0.0, 1.0],
                           ['uniform', 0.0, 1.0]],
            'flip': [['bernoulli', 0.5]],
        },
    },

    'wt_v0.2_sc1.0': {
        'non_geometric': {
            'brightness': [['uniform_multiplier', 0.33, 3]],
            'contrast': [['uniform_multiplier', 0.4, 1.6]],
            'saturation': [['uniform_multiplier', 1.0, 1.0]],
            'hue': [['uniform', -0.0, 0.0]],
        },

        'geometric': {
            'scale_crop': [['uniform', 1.0, 1.0],
                           ['uniform', 1.0, 1.0],
                           ['uniform', 0.0, 1.0],
                           ['uniform', 0.0, 1.0]],
            'flip': [['bernoulli', 0.5]],
        },
    },

    'st_smooth1': {
        'non_geometric': {
            'brightness': ['uniform_multiplier', 0.33, 3],
            'contrast': ['uniform_multiplier', 0.4, 1.6],
            'saturation': ['uniform_multiplier', 1.0, 1.0],
            'hue': ['uniform', -0.0, 0.0],
            'smooth': ['uniform', 0.0, 1.0],
        },

        'geometric': {
            'scale_crop': ['uniform', 0.0, 0.0],
            'flip': ['bernoulli', 0.5],
        }
    },

    'st_smooth2': {
        'non_geometric': {
            'brightness': ['uniform_multiplier', 0.33, 3],
            'contrast': ['uniform_multiplier', 0.4, 1.6],
            'saturation': ['uniform_multiplier', 1.0, 1.0],
            'hue': ['uniform', -0.0, 0.0],
            'smooth': ['uniform', 0.0, 2.0]
        },

        'geometric': {
            'scale_crop': ['uniform', 0.0, 0.0],
            'flip': ['bernoulli', 0.5]
        }
    },

    'st_noise_patch0.1': {
        'non_geometric': {
            'brightness': ['uniform_multiplier', 0.33, 3],
            'contrast': ['uniform_multiplier', 0.4, 1.6],
            'saturation': ['uniform_multiplier', 1.0, 1.0],
            'hue': ['uniform', -0.0, 0.0],
            'noise_patch': ['uniform', 0.0, 0.1]
        },

        'geometric': {
            'scale_crop': ['uniform', 0.0, 0.0],
            'flip': ['bernoulli', 0.5]
        }
    },

    'st_noise_patch0.2': {
        'non_geometric': {
            'brightness': ['uniform_multiplier', 0.33, 3],
            'contrast': ['uniform_multiplier', 0.4, 1.6],
            'saturation': ['uniform_multiplier', 1.0, 1.0],
            'hue': ['uniform', -0.0, 0.0],
            'noise_patch': ['uniform', 0.0, 0.2]
        },

        'geometric': {
            'scale_crop': ['uniform', 0.0, 0.0],
            'flip': ['bernoulli', 0.5]
        }
    },

    'st_wbc0.4': {
        'non_geometric': {
            'brightness': ['uniform_multiplier', 0.33, 3],
            'contrast': ['uniform_multiplier', 0.4, 1.6],
            'saturation': ['uniform_multiplier', 1.0, 1.0],
            'hue': ['uniform', -0.0, 0.0],
            'wbc': ['uniform', 0.0, 0.4]
        },

        'geometric': {
            'scale_crop': ['uniform', 0.0, 0.0],
            'flip': ['bernoulli', 0.5]
        }
    },

    'st_wbc0.3': {
        'non_geometric': {
            'brightness': ['uniform_multiplier', 0.33, 3],
            'contrast': ['uniform_multiplier', 0.4, 1.6],
            'saturation': ['uniform_multiplier', 1.0, 1.0],
            'hue': ['uniform', -0.0, 0.0],
            'wbc': ['uniform', 0.0, 0.3]
        },

        'geometric': {
            'scale_crop': ['uniform', 0.0, 0.0],
            'flip': ['bernoulli', 0.5]
        }
    },

    'st_anoise0.05': {
        'non_geometric': {
            'brightness': [['uniform_multiplier', 0.33, 3]],
            'contrast': [['uniform_multiplier', 0.4, 1.6]],
            'saturation': [['uniform_multiplier', 1.0, 1.0]],
            'hue': [['uniform', -0.0, 0.0]],
            'additive_noise': [['uniform', 0.0, 0.05]],
        },

        'geometric': {
            'flip': [['bernoulli', 0.5]]
        }
    },

    'st_anoise0.01': {
        'non_geometric': {
            'brightness': [['uniform_multiplier', 0.33, 3]],
            'contrast': [['uniform_multiplier', 0.4, 1.6]],
            'saturation': [['uniform_multiplier', 1.0, 1.0]],
            'hue': [['uniform', -0.0, 0.0]],
            'additive_noise': [['uniform', 0.0, 0.01]],
        },

        'geometric': {
            'flip': [['bernoulli', 0.5]]
        }
    },

    'st_anoise0.02': {
        'non_geometric': {
            'brightness': [['uniform_multiplier', 0.33, 3]],
            'contrast': [['uniform_multiplier', 0.4, 1.6]],
            'saturation': [['uniform_multiplier', 1.0, 1.0]],
            'hue': [['uniform', -0.0, 0.0]],
            'additive_noise': [['uniform', 0.0, 0.02]],
        },

        'geometric': {
            'flip': [['bernoulli', 0.5]]
        }
    },

    'st_anoise0.04': {
        'non_geometric': {
            'brightness': [['uniform_multiplier', 0.33, 3]],
            'contrast': [['uniform_multiplier', 0.4, 1.6]],
            'saturation': [['uniform_multiplier', 1.0, 1.0]],
            'hue': [['uniform', -0.0, 0.0]],
            'additive_noise': [['uniform', 0.0, 0.04]],
        },

        'geometric': {
            'flip': [['bernoulli', 0.5]]
        }
    },

    'st_anoise0.08': {
        'non_geometric': {
            'brightness': [['uniform_multiplier', 0.33, 3]],
            'contrast': [['uniform_multiplier', 0.4, 1.6]],
            'saturation': [['uniform_multiplier', 1.0, 1.0]],
            'hue': [['uniform', -0.0, 0.0]],
            'additive_noise': [['uniform', 0.0, 0.08]],
        },

        'geometric': {
            'flip': [['bernoulli', 0.5]]
        }
    },



}


def uniform_multiplier(lo, hi, size):
    assert lo <= 1.0 and hi >= 1.0

    lo = 1/lo - 1
    hi = hi - 1
    rnd = rng.uniform(-lo, hi, size=size)
    return np.where(rnd <= 0, 1/(1 - rnd), 1 + rnd)


def sample_param(distribution, size, params):
    assert len(params) <= 2, "Assertion failed len({}) <= 2 , {}, {}".format(
        params, distribution, size)

    if distribution == 'uniform_multiplier':
        lo, hi = params
        return uniform_multiplier(lo, hi, size)

    elif distribution == 'bernoulli':
        prob = params[0]
        return rng.uniform(0.0, 1.0, size) > (1 - prob)  # 0.0 is not possible

    elif distribution == 'uniform':
        lo, hi = params
        return rng.uniform(lo, hi, size)

    else:
        raise NotImplementedError(
            "Distribution " + distribution + " not implemented.")


def sample_transform(distribution_params, batch_size):
    instance = {}
    for group_name, group_params in distribution_params.items():
        instance_group_p = {}
        for single_transform, params in group_params.items():
            all_params = []
            for i in range(len(params)):
                dist_name = params[i][0]
                all_params.append(sample_param(
                    dist_name, (batch_size,), params=params[i][1:]))

            instance_group_p[single_transform] = np.stack(all_params, axis=1)
        instance[group_name] = instance_group_p

    return instance


def sample_transform_by_label(label, batch_size):
    transform_distribution_p = params_by_label[label]

    return sample_transform(transform_distribution_p, batch_size)


def invert_scale_crop(params):
    params = params.copy()
    bs, num_params = params.shape

    params[:, 0] = 1.0 / params[:, 0]
    params[:, 1] = 1.0 / params[:, 1]

    return params


invert_transform_map = {
    'scale_crop': invert_scale_crop
}


def invert_transform(transform_params):
    '''
    Invert geometric transformations only.
    '''

    # invert flip is just apply the transformation a second time, i.e., the same transform.

    inverted_params = transform_params.copy()

    for group_name, group_params in transform_params.items():
        instance_group_p = {}
        for single_transform, params in group_params.items():
            if single_transform in invert_transform_map.keys():
                inverted_params[group_name][single_transform] = invert_transform_map[single_transform](
                    params)

    return transform_params


# Geometric transforms

# Flip transforms

def flip_intrinsics(data, params):

    idx = params[:, 0]
    b, _, _, _, w = data['color'].shape
    assert params.shape[0] == b

    data['K'][idx, 0, 2] = w - data['K'][idx, 0, 2]


def flip_color(data, params):

    idx = params[:, 0]
    b, s, c, h, w = data['color'].shape
    assert params.shape[0] == b

    tmp = data['color'][idx].view(-1, c, h, w)
    data['color'][idx] = VF.hflip(tmp).view(-1, s, c, h, w)


def flip_depth(data, params):

    idx = params[:, 0]
    # hflip assumes inputs with shape [..., h, w]
    data['pred_depth_snp'][idx] = VF.hflip(data['pred_depth_snp'][idx])


def flip_error(data, params):

    idx = params[:, 0]
    # hflip assumes inputs with shape [..., h, w]
    data['error_snp'][idx] = VF.hflip(data['error_snp'][idx])


def flip_mask(data, params):

    idx = params[:, 0]
    data['mask_snp'][idx] = VF.hflip(data['mask_snp'][idx])


def decode_scale_crop_params(params, h, w):

    scale_w = params[:, 0]
    scale_h = params[:, 1]
    offset_w = params[:, 2]
    offset_h = params[:, 3]
    rnd_w = (scale_w * w).astype(np.int)
    rnd_h = (scale_h * h).astype(np.int)

    # torch.rand((bs,), device=device)).int()
    rnd_offset_w = ((rnd_w - w + 1) * offset_w).astype(np.int)
    # torch.rand((bs,), device=device)).int()
    rnd_offset_h = ((rnd_h - h + 1) * offset_h).astype(np.int)

    return rnd_h, rnd_w, rnd_offset_h, rnd_offset_w


def scale_crop_color(data, params):

    assert len(data['color'].size()) == 5
    bs, _, _, h, w = data['color'].size()
    assert len(params) == bs

    rnd_h, rnd_w, rnd_offset_h, rnd_offset_w = decode_scale_crop_params(
        params, h, w)

    for i in range(bs):
        # Implement resize lanczos for pytorch
        tmp = VF.resize(data['color'][i], (rnd_h[i], rnd_w[i]),
                        VF.InterpolationMode.BILINEAR)
        # tmp = lanczos.resize_image(data['color'][i], (rnd_height[i], rnd_width[i]), kernel='lanczos3') # resize_lanczos(img, size)
        if rnd_h[i] >= h:
            data['color'][i] = VF.crop(
                tmp, rnd_offset_h[i].item(), rnd_offset_w[i].item(), h, w)
        else:
            raise NotImplementedError(
                "Scale crop transform is not implemented for downscaling")


def scale_crop_data(data, key, params):

    assert len(data[key].size()) == 5 or len(data[key].size()) == 4
    bs, h, w = data[key].size(0), data[key].size(-2), data[key].size(-1)
    assert len(params) == bs

    rnd_h, rnd_w, rnd_offset_h, rnd_offset_w = decode_scale_crop_params(
        params, h, w)

    for i in range(bs):
        # Implement resize lanczos for pytorch
        tmp = VF.resize(data[key][i], (rnd_h[i], rnd_w[i]),
                        VF.InterpolationMode.BILINEAR)
        # tmp = lanczos.resize_image(data['color'][i], (rnd_height[i], rnd_width[i]), kernel='lanczos3') # resize_lanczos(img, size)
        if rnd_h[i] >= h:
            data[key][i] = VF.crop(
                tmp, rnd_offset_h[i].item(), rnd_offset_w[i].item(), h, w)

        else:
            device = data[key].device
            shift_h = -1 * rnd_offset_h[i]
            shift_w = -1 * rnd_offset_w[i]
            data[key][i] = torch.zeros(size=data[key][i].size(), device=device)
            if len(data[key].size()) == 5:
                data[key][i, :, :, shift_h:shift_h + rnd_h[i],
                          shift_w:shift_w + rnd_w[i]] = tmp
            else:
                data[key][i, :, shift_h:shift_h + rnd_h[i],
                          shift_w:shift_w + rnd_w[i]] = tmp


def scale_crop_depth(data, params):

    return scale_crop_data(data, 'pred_depth_snp', params)


def scale_crop_mask(data, params):

    return scale_crop_data(data, 'mask_snp', params)


def scale_crop_error(data, params):

    return scale_crop_data(data, 'error_snp', params)


def scale_crop_intrinsics(data, params):

    assert len(data['color'].size()) == 5
    bs, _, _, h, w = data['color'].size()
    assert len(params) == bs

    device = data['color'].device
    scale_w = torch.from_numpy(params[:, 0]).float().to(device)
    scale_h = torch.from_numpy(params[:, 1]).float().to(device)

    _, _, rnd_offset_h, rnd_offset_w = decode_scale_crop_params(params, h, w)
    rnd_offset_h = torch.from_numpy(rnd_offset_h).float().to(device)
    rnd_offset_w = torch.from_numpy(rnd_offset_w).float().to(device)

    if 'K' in data.keys():
        assert len(data['K']) == bs

        data['K'][:, 0, 0] *= scale_w
        data['K'][:, 1, 1] *= scale_h
        # rnd_offset_x is non-negative when scale_x >= 1 and negative when s < 1
        data['K'][:, 0, 2] = data['K'][:, 0, 2] * scale_w - rnd_offset_w
        data['K'][:, 1, 2] = data['K'][:, 1, 2] * scale_h - rnd_offset_h

# non geometric transforms


def adjust_brightness(data, params):
    batch_size, _, _, _, _ = data['color'].shape
    assert batch_size == len(params)
    value = params[:, 0]

    for i in range(batch_size):
        data['color'][i] = VF.adjust_brightness(data['color'][i], value[i])


def adjust_contrast(data, params):
    batch_size, _, _, _, _ = data['color'].shape
    assert batch_size == len(params)

    value = params[:, 0]
    for i in range(batch_size):
        data['color'][i] = VF.adjust_contrast(data['color'][i], value[i])


def adjust_saturation(data, params):
    batch_size, _, _, _, _ = data['color'].shape
    assert batch_size == len(params)

    value = params[:, 0]
    for i in range(batch_size):
        data['color'][i] = VF.adjust_saturation(data['color'][i], value[i])


def adjust_hue(data, params):
    batch_size, _, _, _, _ = data['color'].shape
    assert batch_size == len(params)

    value = params[:, 0]
    for i in range(batch_size):
        data['color'][i] = VF.adjust_hue(data['color'][i], value[i])


def smooth(data, params):
    batch_size, _, _, _, _ = data['color'].shape
    assert batch_size == len(params)

    sigma = params[:, 0]
    for i in range(batch_size):
        k = math.ceil(6.6 * sigma[i] - 2.3)
        k += 1 - (k % 2)

        if k >= 3:
            data['color'][i] = VF.gaussian_blur(
                data['color'][i], kernel_size=(k, k), sigma=(sigma[i], sigma[i]))

# additive white noise


def additive_noise_color(data, params):
    batch_size, _, _, _, _ = data['color'].shape
    assert batch_size == len(params)

    device = data['color'].device
    upper_bound = params[:, 0].reshape((-1, 1, 1, 1, 1))
    upper_bound = torch.from_numpy(upper_bound).float().to(device)
    noise = (2 * torch.rand(data['color'].shape,
             device=device) - 1) * upper_bound
    #m, s = data['color'].mean(), data['color'].std()
    #print("wo noise", m, s)
    data['color'] += noise
    #m, s = data['color'].mean(), data['color'].std()
    #print("w noise", m, s)


# local non-geometric transforms

def noise_patch(data, params):
    batch_size, _, _, _, _ = data['color'].shape
    assert batch_size == len(params)

    areas = params[:, 0]
    for i in range(batch_size):
        n, c, h, w = data['color'][i].shape
        #SL, SH = 0.02, max_area
        area = areas[i] * min(h, w)**2

        RL = 0.3
        RH = 1.0 / RL
        ratio = uniform_multiplier(RL, RH, (1,))[0]

        ch = min(int((area * ratio) ** 0.5), h)
        cw = min(int((area / ratio) ** 0.5), w)

        rnd = rng.uniform(size=(2,))
        offset_h = int((h - ch) * rnd[0])
        offset_w = int((w - cw) * rnd[1])

        noise = torch.rand(n, c, ch, cw)
        data['color'][i][:, :, offset_h:offset_h +
                         ch, offset_w:offset_w + cw] = noise


def white_blended_circles(data, params):
    batch_size, _, _, _, _ = data['color'].shape
    assert batch_size == len(params)

    MAX_BLEND = 0.7
    MAX_CIRCLES = 4
    MIN_AREA = 0.1
    areas = params[:, 0]

    for i in range(batch_size):
        n, c, h, w = data['color'][i].shape

        if areas[i] >= MIN_AREA:
            continue

        rnd = rng.uniform(size=(4 * MAX_CIRCLES,))

        # circle
        x = torch.arange(0, h)
        y = torch.arange(0, w)
        xx, yy = torch.meshgrid(x, y)

        RL = 0.3
        RH = 1.0/RL

        area_t = areas[i] * min(h, w) ** 2
        area = rnd[:MAX_CIRCLES]
        area = area_t * area / area.sum()
        circle_params = rnd[MAX_CIRCLES:]

        for j in range(MAX_CIRCLES):
            data['color'][i] = single_circle(
                data['color'][i], area[j], circle_params[3*j:3*j + 3], MAX_BLEND, xx, yy)

# Network perturbations
# ladder networks


transforms_map = {
    'brightness': {
        'color': adjust_brightness,
    },

    'contrast': {
        'color': adjust_contrast,
    },

    'saturation': {
        'color': adjust_saturation,
    },

    'hue': {
        'color': adjust_hue,
    },

    'smooth': {
        'color': smooth,
    },

    'noise_patch': {
        'color': noise_patch,
    },

    'wbc': {
        'color': white_blended_circles,
    },

    'flip': {
        'color': flip_color,
        'pred_depth_snp': flip_depth,
        'mask_snp': flip_mask,
        'error_snp': flip_error,
        'K': flip_intrinsics
    },

    'scale_crop': {
        'color': scale_crop_color,
        'pred_depth_snp': scale_crop_depth,
        'mask_snp': scale_crop_mask,
        'error_snp': scale_crop_error,
        'K': scale_crop_intrinsics,
    },

    'additive_noise': {
        'color': additive_noise_color,
    }
}


def apply_transform(data, transform_params):
    transformed_data = dict()
    for k in data.keys():
        transformed_data[k] = data[k].detach().clone()

    for _, group_params in transform_params.items():
        for transform, params in group_params.items():
            for input_name, _ in data.items():
                assert params is not None, "Transform {} does not have params".format(
                    transform)
                k = list(transformed_data.keys())[0]
                b = transformed_data[k].size(0)
                assert b == params.shape[0]

                if input_name in transforms_map[transform].keys():
                    #print("apply transform", transform, input_name, params)
                    transforms_map[transform][input_name](
                        transformed_data,
                        params)

    return transformed_data


def color_jitter(snippet, brightness, contrast, saturation, hue, sharpness=(1.0, 1.0), equalize=0, autocontrast=0, posterize=(0.0, 1)):
    assert len(snippet.shape) == 5
    batch_size, _, _, _, _ = snippet.shape

    # means blend(img, zero, ratio) for ration < 1, img = ratio * img for ratio >= 1, lb 0.2, ub = 2, 3, 4 o 5
    # alternative rnd_brightness = rng.uniform(brightness[0], brightness[1], batch_size)
    rnd_brightness = uniform_multiplier(
        brightness[0], brightness[1], batch_size)

    # contrast is implemented as blend img = img * ratio - mean(img) * (ratio - 1), mean is invariant, but std is scaled: ratio * std
    # alternative rnd_contrast = rng.uniform(contrast[0], contrast[1], batch_size)
    rnd_contrast = uniform_multiplier(contrast[0], contrast[1], batch_size)

    # saturation is implemented as blend img * ratio - gray * (ratio - 1), between gray(ratio = 0), and more colorfull (assumption)
    # alternative rnd_saturation = rng.uniform(saturation[0], saturation[1], batch_size)
    rnd_saturation = uniform_multiplier(
        saturation[0], saturation[1], batch_size)

    rnd_hue = rng.uniform(hue[0], hue[1], batch_size)
    rnd_sharpness = rng.uniform(sharpness[0], sharpness[1], batch_size)

    # normalized transformations
    rnd_equalize = rng.uniform(0, 1, batch_size) < equalize
    rnd_autocontrast = rng.uniform(0, 1, batch_size) < autocontrast

    assert posterize[1] <= 7
    rnd_posterize_on = rng.uniform(0, 1, batch_size) < posterize[0]
    rnd_posterize_bits = 8 - \
        rng.randint(1, posterize[1] + 1, batch_size)  # bits to keep

    for i in range(batch_size):

        if rnd_autocontrast[i]:
            snippet[i] = VF.autocontrast(snippet[i])

        int8_needed = rnd_equalize[i] or rnd_posterize_on[i]
        if int8_needed:
            snippet_uint8 = (snippet[i]*255).byte()

        if rnd_equalize[i]:
            snippet_uint8 = VF.equalize(snippet_uint8)

        if rnd_posterize_on[i]:
            snippet_uint8 = VF.posterize(snippet_uint8, rnd_posterize_bits[i])

        if int8_needed:
            snippet[i] = snippet_uint8.float()/255.0

        snippet[i] = VF.adjust_brightness(snippet[i], rnd_brightness[i])
        snippet[i] = VF.adjust_saturation(snippet[i], rnd_saturation[i])
        snippet[i] = VF.adjust_contrast(snippet[i], rnd_contrast[i])
        snippet[i] = VF.adjust_hue(snippet[i], rnd_hue[i])

        if sharpness[0] < sharpness[1]:
            snippet[i] = VF.adjust_sharpness(snippet[i], rnd_sharpness[i])


def scale_crop_transform(data, max_offset=0.15):
    '''
    Args
        snippet: A list of images [c, h, w]
        depth: A numpy array containing the depth map of the target image [h, w]
        K: A numpy array containing the intrinsic [4, 4]
    '''
    assert len(data['color'].shape) == 5

    if max_offset > 0:

        batch_size, _, _, height, width = data['color'].size()
        device = data['color'].device

        sf_width = 1.0 + max_offset * torch.rand((batch_size,), device=device)
        sf_height = 1.0 + max_offset * torch.rand((batch_size,), device=device)
        rnd_width = (sf_width * width).int()
        rnd_height = (sf_height * height).int()
        rnd_offsetw = ((rnd_width - width + 1) *
                       torch.rand((batch_size,), device=device)).int()
        rnd_offseth = ((rnd_height - height + 1) *
                       torch.rand((batch_size,), device=device)).int()

        for i in range(batch_size):
            # resize_lanczos(img, size)
            tmp = lanczos.resize_image(
                data['color'][i], (rnd_height[i], rnd_width[i]), kernel='lanczos3')
            data['color'][i] = VF.crop(
                tmp, rnd_offseth[i].item(), rnd_offsetw[i].item(), height, width)

        if 'K' in data.keys():
            assert len(data['K']) == batch_size

            data['K'][:, 0, 0] *= sf_width
            data['K'][:, 1, 1] *= sf_height

            data['K'][:, 0, 2] = data['K'][:, 0, 2] * sf_width - rnd_offsetw
            data['K'][:, 1, 2] = data['K'][:, 1, 2] * sf_height - rnd_offseth

        if 'pred_depth' in data.keys():
            raise NotImplementedError(
                "TODO: implement for self-training approach if needed")


def flip_transform(data):
    assert len(data['color'].size()) == 5
    batch_size, s, c, h, w = data['color'].size()

    p = rng.uniform(0, 1, batch_size)
    idx = p > 0.5
    np.expand_dims(p, 1)

    if 'K' in data.keys():
        data['K'][idx, 0, 2] = w - data['K'][idx, 0, 2]

    tmp = data['color'][idx].view(-1, c, h, w)
    data['color'][idx] = VF.hflip(tmp).view(-1, s, c, h, w)

    if 'pred_depth_snp' in data.keys():
        # hflip assumes inputs with shape [..., h, w]
        data['pred_depth_snp'][idx] = VF.hflip(data['pred_depth_snp'][idx])

    if 'mask_snp' in data.keys():
        data['mask_snp'][idx] = VF.hflip(data['mask_snp'][idx])


def normalize(data):
    if 'color' in data.keys():
        assert len(data['color'].size()) == 5
        b, n, c, h, w = data['color'].size()

        data['color'] = data['color'].view(-1, c, h, w)
        data['color'] = VF.normalize(
            data['color'], mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        data['color'] = data['color'].view(b, n, c, h, w)

    return data


def denormalize(data):
    assert len(data['color'].size()) == 5
    b, n, c, h, w = data['color'].size()

    data['color'] = data['color'].view(-1, c, h, w)
    device = data['color'].device
    mean = torch.tensor([[[[0.485]], [[0.456]], [[0.406]]]],
                        dtype=torch.float32, device=device)
    std = torch.tensor([[[[0.229]], [[0.224]], [[0.225]]]],
                       dtype=torch.float32, device=device)
    data['color'] = data['color'] * std + mean
    data['color'] = data['color'].view(b, n, c, h, w)

    return data


def composite_transform(data, params):
    assert len(data['color'].size()) == 5
    b, _, _, _, _ = data['color'].size()

    transformed_data = dict()
    for k in data.keys():
        transformed_data[k] = data[k].detach().clone()

    color_jitter(transformed_data['color'], **params['color'])
    scale_crop_transform(transformed_data, **params['scale_crop'])

    if params['flip']:
        flip_transform(transformed_data)

    if 're' in params.keys() and params['re']:
        rnd = rng.uniform(size=(b,))
        for i in range(b):
            transformed_data['color'][i] = random_erasing_f(
                transformed_data['color'][i], rnd[i], **params['re'])

    if 'random_local' in params.keys() and params['random_local']:
        rnd = rng.uniform(size=(b,))
        for i in range(b):
            transformed_data['color'][i] = random_local_transform(
                transformed_data['color'][i], rnd[i], **params['random_local'])

    return transformed_data


# rand augment
def autocontrast_f(snippet, prob):
    if prob < 0.5:
        snippet = VF.autocontrast(snippet)

    return snippet


def equalize_f(snippet, prob):
    if prob < 0.5:
        snippet_uint8 = (snippet*255).byte()
        snippet = VF.equalize(snippet_uint8).float()/255.0

    return snippet


def posterize_f(snippet, bits):
    assert bits < 5

    snippet_uint8 = (snippet*255).byte()
    return VF.posterize(snippet_uint8, int(bits)).float()/255.0


def random_erasing_f(snippet, prob, max_area=0.4):
    if prob < 0.5:
        rnd = rng.uniform(size=(4,))
        n, c, h, w = snippet.size()

        SL, SH = 0.02, max_area
        RL = 0.3
        RH = 1.0/RL

        area = (SL + (SH - SL) * rnd[0]) * min(h, w)**2
        ratio = uniform_multiplier(RL, RH, (1,))[0]

        ch = min(int((area * ratio) ** 0.5), h)
        cw = min(int((area / ratio) ** 0.5), w)
        offset_h = int((h - ch) * rnd[2])
        offset_w = int((w - cw) * rnd[3])

        noise = torch.rand(n, c, ch, cw)

        snippet[:, :, offset_h:offset_h + ch, offset_w:offset_w + cw] = noise

    return snippet


def single_circle(snippet, area, rnd, max_blend, xx, yy):
    _, _, h, w = snippet.shape

    cx = int(h * rnd[0])
    cy = int(w * rnd[1])
    sq_dist_map = (xx - cx) ** 2 + (yy - cy) ** 2

    sq_rad = area / math.pi

    perturbed = snippet.clone()
    perturbed[:, :, sq_dist_map < sq_rad] = snippet.max()

    rnd_blend = rnd[2] * max_blend
    snippet = snippet * (1 - rnd_blend) + perturbed * rnd_blend

    return snippet


def random_local_transform(snippet, prob, max_area=0.4, max_blend=0.7, num_patches=4):
    if prob < 0.5:
        rnd = rng.uniform(size=(1 + 4*num_patches,))

        n, c, h, w = snippet.size()

        # circle
        x = torch.arange(0, h)
        y = torch.arange(0, w)
        xx, yy = torch.meshgrid(x, y)

        SL, SH = 0.1, max_area
        RL = 0.3
        RH = 1.0/RL

        area_t = (SL + (SH - SL) * rnd[0]) * min(h, w)**2
        area = rnd[1:1 + num_patches]
        area = area_t * area / area.sum()
        params = rnd[1 + num_patches:]

        for i in range(num_patches):
            snippet = single_circle(
                snippet, area[i], params[3*i:3*i + 3], max_blend, xx, yy)

    return snippet


# Sharpness did not work. Consider if invert, rotate, solarize solarize add gamma work.
transforms = [
    (VF.adjust_brightness, 0.33, 3),
    (VF.adjust_contrast, 0.4, 1.6),
    (equalize_f, 0, 1),
    (autocontrast_f, 0, 1),
    (lambda imgs, _: imgs, 0, 1),  # rav0.2
    (VF.adjust_saturation, 0.8, 1.2),
    (VF.adjust_hue, -0.1, 0.1),
    (posterize_f, 1, 2),  # [1, 2>
]

def composite_randaugment(data, num_ops=2, magnitude=1.0, total_ops=-1, random_mode=True):
    assert random_mode  # TODO: fixed
    assert magnitude <= 1.0 and magnitude >= 0
    assert total_ops <= len(transforms)

    if total_ops == -1:
        total_ops = len(transforms)

    bs, n, c, h, w = data['color'].size()
    assert len(data['color'].size()) == 5

    transformed_data = dict()
    for k in data.keys():
        transformed_data[k] = data[k].detach().clone()

    snippet = transformed_data['color']

    ops_sample_idx = rng.choice(total_ops, size=num_ops, replace=False)
    for i in range(num_ops):
        transform_f, minv, maxv = transforms[ops_sample_idx[i]]
        if random_mode:
            cur_mag = minv + rng.uniform(0, magnitude * (maxv - minv), bs)
        else:
            cur_mag = minv + np.ones(size=(bs,)) * magnitude * (maxv - minv)

        for j in range(bs):
            snippet[j] = transform_f(snippet[j], cur_mag[j])

    transformed_data['color'] = snippet.view(bs, n, c, h, w)

    flip_transform(transformed_data)

    return transformed_data


def composite_transform_by_label(data, transform_label):
    assert transform_label in params_by_label.keys()

    return composite_transform(data, params_by_label[transform_label])


def compute_image_pyramid(data, num_scales=4, downsample='bilinear', full_res=False):
    has_pred = 'pred_depth_snp' in data.keys()
    has_mask = 'mask_snp' in data.keys()
    has_error = 'error_snp' in data.keys()

    b, s, c, h, w = data['color'].size()

    if downsample == 'bilinear':
        def downsample_f(img, size): return VF.resize(
            img, size, VF.InterpolationMode.BILINEAR)
    if downsample == 'bicubic':
        def downsample_f(img, size): return VF.resize(
            img, size, VF.InterpolationMode.BICUBIC)
    if downsample == 'lanczos':
        def downsample_f(img, size): return lanczos.resize_image(
            img, size, kernel='lanczos3')  # resize_lanczos(img, size)

    data[('color', 0)] = data['color'].view(-1, c, h, w)
    if has_pred:
        data[('pred_depth_snp', 0)] = data['pred_depth_snp'].view(-1, 1, h, w)
    if has_mask:
        data[('mask_snp', 0)] = data['mask_snp'].view(-1, 1, h, w)
    if has_error:
        data[('error_snp', 0)] = data['error_snp'].view(-1, 1, h, w)

    for i in range(1, num_scales):
        size = (h//(2**i), w//(2**i))
        data[('color', i)] = downsample_f(
            data[('color', 0)], size).view(b, s, c, size[0], size[1])
        if has_pred:
            data[('pred_depth_snp', i)] = downsample_f(
                data[('pred_depth_snp', 0)], size).view(b, s, 1, size[0], size[1])
        if has_mask:
            data[('mask_snp', i)] = downsample_f(
                data[('mask_snp', 0)], size).view(b, s, 1, size[0], size[1])
        if has_error:
            data[('error_snp', i)] = downsample_f(
                data[('error_snp', 0)], size).view(b, s, 1, size[0], size[1])

    data[('color', 0)] = data[('color', 0)].view(b, s, c, h, w)
    if has_pred:
        data[('pred_depth_snp', 0)] = data[(
            'pred_depth_snp', 0)].view(b, s, 1, h, w)
    if has_mask:
        data[('mask_snp', 0)] = data[('mask_snp', 0)].view(b, s, 1, h, w)
    if has_error:
        data[('error_snp', 0)] = data[('error_snp', 0)].view(b, s, 1, h, w)

    return data

