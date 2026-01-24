#!/bin/bash

# ==============================================================================
# aGPD-Lite++ 6-View Experiment (Cleaned & Strict Doc Version)
# 策略: 使用 6 张训练视图。
#       严格对齐文档 Algorithm 2，移除 Fallback 和 Perturb 参数。
# ==============================================================================

# 1. 实验名称
exp_name="llff_sagpd_6view_final"

# 2. 场景列表 (LLFF 数据集)
scenes=("fern" "flower" "fortress" "horns" "leaves" "orchids" "room" "trex")

# 3. 循环训练
for scene_name in "${scenes[@]}"
do
    echo ""
    echo "============================================================"
    echo " [Start] 正在优化场景: $scene_name (6-View aGPD-Lite++)"
    echo "============================================================"

    python train.py \
      -s dataset/llff/$scene_name \
      -m output/$exp_name/$scene_name \
      --eval -r 8 --n_views 6 \
      --pre_densify \
      --pre_knn_neighbors 16 \
      --pre_sparsity_threshold 0.7 \
      --pre_opacity_scale 0.3 \
      --pre_size_shrink 1.5 \
      --pre_min_consistency_views 2

    # 执行渲染评估
    echo ">> [Render] 正在生成评估图像..."
    python render.py \
      -m output/$exp_name/$scene_name \
      -r 8 \
      --quiet

    echo ">> [Done] 场景 $scene_name 完成。"
done

# 4. 汇总所有场景指标
echo ""
echo "============================================================"
echo " 正在汇总所有场景指标..."
echo "============================================================"
python metric.py --path output/$exp_name