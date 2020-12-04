# AVX2 could be slightly faster than SSE4
# If AVX2 (AVX is not enough, check grep flags /proc/cpuinfo)
#pip uninstall pillow; CC="cc -mavx2" pip install -U --force-reinstall pillow-simd #TODO test SSE4
pip uninstall pillow; pip install --no-cache-dir -U --force-reinstall pillow-simd #TODO test SSE4


