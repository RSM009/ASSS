#!/bin/bash

# ========= Basic Configuration =========

LANGUAGE="ponss_k_32"
MODEL_NAME="meta-llama/Llama-3.2-3B-Instruct"
MAX_SEQ_LENGTH=2048

export CUDA_VISIBLE_DEVICES=1
echo "Using GPU: $CUDA_VISIBLE_DEVICES"

OUTPUT_DIR="./output/${LANGUAGE}_${MODEL_NAME//\//_}"
TRAIN_PATH="/home/om/train_data/${LANGUAGE}_train_data.json"
EVAL_PATH="/home/om/val_data/${LANGUAGE}_val_data.json"

# ========= Launch Training =========

python3.10 lora3.2_3b.py \
  --model_name "${MODEL_NAME}" \
  --train_dataset_path "${TRAIN_PATH}" \
  --eval_dataset_path "${EVAL_PATH}" \
  --output_dir "${OUTPUT_DIR}" \
  \
  --lora_r 16 \
  --lora_alpha 32 \
  \
  --max_seq_length ${MAX_SEQ_LENGTH} \

echo "Training finished for '${LANGUAGE}'. Output saved to '${OUTPUT_DIR}'"
