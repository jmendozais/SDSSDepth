import torch
import numpy as np
from PIL import Image
from turbojpeg import TurboJPEG as JPEG

reader = JPEG()
filename='/home/Datasets/tartan-lr/hospital/hospital/Easy/P031/image_left/000060_left.jpg'
with open(filename, 'rb') as f:
    t = reader.decode(f.read(), pixel_format=0)

print(t.size)
t  = Image.fromarray(t)
n = np.asarray(t)
print(t.size)
print(n.shape)
t.resize((100, 100), resample=Image.BILINEAR)
flow = n[:,:,:2]
f = Image.fromarray(flow)
f = f.resize((100, 100), resample=Image.BILINEAR)
print(np.asarray(f).shape)
