# Getting Started

## Requirements

SDSSDepth requires the following libraries installed in the host operating system:

* libturbojpeg
* libjpeg-dev
* libsm6
* libxrender1

## Install
Clone the repository:

<code>
git clone https://github.com/jmendozais/SDSSDepth
</code>

Install dependencies:

<code>
pip install -r requirements.txt
</code>

[Optional] Install Pillow SIMD for an additional speed-up (requires SSE4):

<code>
pip uninstall pillow; CC="cc -mavx2" pip install -U --force-reinstall pillow-simd
</code>

## Running a pre-trained model
Download a pretrained model [here](https://drive.google.com/file/d/1MhVFZdyptfRsOwJdfqtDL8PArihKWjs7/view?usp=sharing). 

Move the model file into the misc/ directory.

The following code runs the pre-trained model on the test images located in the directory misc/test_data:

<code>
CUDA_VISIBLE_DEVICES=0 python inference.py --checkpoint misc/depth-crfinal-s3-seq3-ema3-swa3-lrcoswr1e-4-to-1e-5-crw1e5-pixth0.2-pixema0.99.tar --input-dir misc/test_data/ --output-dir misc/test_data_out/
</code>

