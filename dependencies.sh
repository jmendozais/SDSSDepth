apt-get update 
apt-get install libturbojpeg
apt-get install zlip-dev
apt-get install libjpeg-dev
apt-get install libsm6
apt-get install libxrender1


#pip uninstall pillow; CC="cc -mavx2" pip install -U --force-reinstall pillow-simd #TODO test SSE4

#PYCLS=../pycls
#git clone https://github.com/facebookresearch/pycls $PYCLS
#pip install -r $PYCLS/requirements.txt

#current=$(pwd)
#cd $PYCLS && python setup.py develop --user
#cd $current

# Prepare dataset
#python3 ytwalking_download.py --input_file ytwalking_urls.csv
