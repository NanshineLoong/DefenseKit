#!/bin/bash

DEVICE=${1:-0}
MODEL_NAME=${2:-"llava-hf/llava-1.5-7b-hf"}
DATASET_PATH=${3:-"./datasets/mmsafetybench/train.json"}
TASK=${4:-"generation"}
OUTPUT_DIR=${5:-"./outputs"}
ADAPTER_DIR=${6:-"./resources/adapter/llava-1.5-7b"}

OUTPUT_PATH="${OUTPUT_DIR}/safedecoding-responsible.jsonl"
if [ -f "$OUTPUT_PATH" ]; then
  echo "Skip $OUTPUT_PATH"
  exit
fi

python main.py \
  --model_name $MODEL_NAME \
  --use_adapter \
  --adapter_dir "${ADAPTER_DIR}lora/sft/spavl_llava-rlhf" \
  --device $DEVICE \
  --dataset_path $DATASET_PATH \
  --input_defense_strategy "responsible" \
  --decoding_strategy "safedecoding" \
  --output_defense_strategy "" \
  --task $TASK \
  --batch_size 5 \
  --judger "KeywordJudger" \
  --logging_path "./logs" \
  --output_path $OUTPUT_PATH
