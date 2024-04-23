#!/bin/sh
ONNX_PATH="../quantize_jixiu/models/yolov8n_opt.onnx"
FP32_MODEL="../quantize_jixiu/models/yolov8n_opt.dlc"
INT8_MODEL="../quantize_jixiu/models/yolov8n_onnx_16_quantized.dlc"
IMAGE_LIST="../data_jixiu/image_file_list.txt"
snpe-onnx-to-dlc -i $ONNX_PATH \
  -d x.3 1,3,640,640 \
  --out_name bboxes \
  --out_name scores \
  -o $FP32_MODEL \

snpe-dlc-quantize \
  --input_dlc $FP32_MODEL \
  --input_list $IMAGE_LIST \
  --output_dlc $INT8_MODEL \
  --enable_htp \
  --htp_socs=sm8350 \
  --axis_quant \
  --use_enhanced_quantizer \
  --verbose