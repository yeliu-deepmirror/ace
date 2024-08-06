#!/bin/sh

ONNX_PATH="models/model_nano_lines.onnx"
FP32_MODEL="models/model_nano_lines.dlc"
INT8_MODEL="models/model_nano_lines_quantized.dlc"
IMAGE_LIST="data/quantize.txt"

# conda activate py36
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/yeliu/anaconda3/envs/py36/lib
export ANDROID_NDK_ROOT=/opt/android-sdk/ndk
SNPE_PATH=/home/yeliu/Downloads/dep/snpe-1.61.0.3358
source ${SNPE_PATH}/bin/envsetup.sh -p /home/yeliu/anaconda3/envs/py36/lib/python3.6/site-packages/torch

echo "========================================"

snpe-onnx-to-dlc --input_network ${ONNX_PATH} \
                 --output_path ${FP32_MODEL}
# snpe-dlc-info -i models/model_nano_lines_quantized.dlc

echo "========================================"

snpe-dlc-quantize \
  --input_dlc $FP32_MODEL \
  --input_list $IMAGE_LIST \
  --output_dlc $INT8_MODEL \
  --enable_htp \
  --htp_socs=sm8350 \
  --axis_quant \
  --use_enhanced_quantizer \
  --verbose

cp ${ONNX_PATH} /home/yeliu/Development/LidarMapping/data/map/model_nano_lines.onnx
cp ${FP32_MODEL} /home/yeliu/Development/LidarMapping/data/map/model_nano_lines.dlc
cp ${INT8_MODEL} /home/yeliu/Development/LidarMapping/data/map/model_nano_lines_quantized.dlc
