#!/bin/bash

# 实验名称前缀
exp_base="llff_pre_densify"
# 八个标准场景
scenes=("fern" "flower" "fortress" "horns" "leaves" "orchids" "room" "trex")
# 修正后的数据集路径（根据你之前的 tree 命令结果）
dataset_path='dataset/llff'
# 视图列表：对应论文中的 3-view, 6-view, 9-view
view_list=(3 6 9)

for n_views in "${view_list[@]}"
do
  exp_name="${exp_base}_${n_views}view"
  echo "===================================================="
  echo "开始执行 ${n_views} 视图实验，输出目录: output/$exp_name"
  echo "===================================================="

  for scene in "${scenes[@]}"
  do
    echo "------------------------------------------------"
    echo "正在处理场景: $scene (视图数: $n_views)"
    echo "------------------------------------------------"

    # 关键修复：手动创建点云转换所需的目录，防止报错
    mkdir -p $dataset_path/$scene/${n_views}_views/dense/

    # 训练阶段：加入你的 --pre_densify 改进逻辑
    # 按照论文设置：迭代 10,000 次，下采样率为 8 [cite: 242, 247]
    python train.py -s $dataset_path/$scene/ \
      -m output/$exp_name/$scene \
      --eval -r 8 \
      --n_views $n_views \
      --pre_densify \
      --port 6007

    # 渲染阶段
    echo "正在渲染 $scene..."
    python render.py -m output/$exp_name/$scene -r 8
  done

  # 每跑完一种视图（如 3 视图下的所有 8 个场景），自动计算一次平均指标
  echo "正在计算 ${n_views} 视图下的 LLFF 平均指标..."
  python metric.py --path output/$exp_name
done

echo "所有实验已完成！"