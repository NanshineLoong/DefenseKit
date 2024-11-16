#!/bin/bash

DEVICE=${1:-0}
MODEL_NAME=${2:-"llava-hf/llava-1.5-7b-hf"}
DATASET_PATH=${3:-"./datasets/mmsafetybench/train.json"}
TASK=${4:-"generation"}
OUTPUT_DIR=${5:-"./outputs"}

OUTPUT_PATH="${OUTPUT_DIR}/intention-responsible.jsonl"
if [ -f "$OUTPUT_PATH" ]; then
  echo "Skip $OUTPUT_PATH"
  exit
fi

python main.py \
  --model_name $MODEL_NAME \
  --device $DEVICE \
  --dataset_path $DATASET_PATH \
  --input_defense_strategy "intention,responsible" \
  --decoding_strategy "" \
  --output_defense_strategy "" \
  --task $TASK \
  --batch_size 5 \
  --judger "KeywordJudger" \
  --logging_path "./logs" \
  --output_path $OUTPUT_PATH
