#!/bin/bash
MODEL='llava-1.5-7b'
DATASET='mm-vet'
TASK="generation"
OUTPUT_DIR="./outputs/${DATASET}/${MODEL}"

MODEL_NAME="llava-hf/${MODEL}-h"
DATASET_PATH="./datasets/${DATASET}/test.json"
ADAPTER_DIR="./resources/adapter/${MODEL}/"


# 可用设备ID列表
DEVICES=(0 1 2 3 4 5 6 7)

# 当前正在执行的脚本的设备映射
declare -A DEVICE_MAP

# 脚本目录
SCRIPTS_DIR="./scripts"

# 分配设备并执行脚本
execute_script() {
  local script=$1
  for device in "${DEVICES[@]}"; do
    if [ -z "${DEVICE_MAP[$device]}" ]; then
      echo "Execute $script on device $device"
      bash "$script" "$device" "$MODEL_NAME" "$DATASET_PATH" "$TASK" "$OUTPUT_DIR" "$ADAPTER_DIR" &  # 异步执行脚本
      DEVICE_MAP[$device]=$!  # 存储脚本的进程ID
      return
    fi
  done
}

# 监控并处理脚本执行
monitor_scripts() {
  local script_list=($(find "$SCRIPTS_DIR" -type f -name "*.sh"))
  for script in "${script_list[@]}"; do
    while true; do
      for device in "${DEVICES[@]}"; do
        if ! ps -p ${DEVICE_MAP[$device]} &>/dev/null; then
          unset DEVICE_MAP[$device]  # 释放设备
          execute_script "$script"   # 分配设备并执行下一个脚本
          break 2
        fi
      done
      sleep 1
    done
  done

  # 等待所有脚本执行完毕
  wait
}

# 运行脚本监控
monitor_scripts
