#!/bin/bash

MODEL_PATH=""
DATA_PATH=""

GPU_COUNT=$(nvidia-smi -L | wc -l)
echo "检测到当前机器共有 $GPU_COUNT 张显卡。"

if [ "$GPU_COUNT" -eq 0 ]; then
    echo "错误：未检测到可用 GPU。"
    exit 1
fi

echo "正在启动 $GPU_COUNT 个分片并行运行 (tp=1)..."

for (( i=0; i<$GPU_COUNT; i++ ))
do
    echo "正在后台启动 Shard $i (使用 GPU $i)..."
    CUDA_VISIBLE_DEVICES=$i python step1_generate_solutions.py \
        --shard $i \
        --total_shards $GPU_COUNT \
        --tp 1 > "log_shard$i.txt" 2>&1 &
done


# 等待所有后台进程结束
wait

echo "所有分片计算完成！正在执行合并脚本..."
python merge_shards.py

