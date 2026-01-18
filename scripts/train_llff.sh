#!/bin/bash

# 1. 修正实验名称
exp_name='llff3'
# 2. 修正数据集路径：确保指向你现在的 dataset/llff
dataset_path='dataset/llff'
scenes=("fern" "flower" "fortress" "horns" "leaves" "orchids" "room" "trex")
n_views=3

for scene in "${scenes[@]}"
do
  echo "正在处理场景: $scene..."
  
  # 关键修复：确保视图文件夹存在（防止 FileNotFoundError）
  mkdir -p $dataset_path/$scene/${n_views}_views/dense/

  # 3. 训练命令：注意每行末尾必须有反斜杠 \
  python train.py -s $dataset_path/$scene/ \
    -m output/$exp_name/$scene \
    --eval -r 8 \
    --n_views $n_views \
    --pre_densify
  
  echo "正在渲染 $scene..."
  python render.py -m output/$exp_name/$scene -r 8
done

# 计算所有场景的平均指标
python metric.py --path output/$exp_name