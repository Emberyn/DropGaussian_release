#!/bin/bash

# ==============================================================================
# SaGPD 6-View Experiment (Precision Strategy)
#
# 策略: 相比 3 视图，这里全面收紧了约束。
#       利用更丰富的数据(6图)来追求更高的几何精度 (High Precision)。
# ==============================================================================

# 1. 实验名称 (建议标记为 refined 或 precision)
exp_name="llff_sagpd_6view_refined"

# 场景列表
scenes=("fern" "flower" "fortress" "horns" "leaves" "orchids" "room" "trex")

for scene_name in "${scenes[@]}"
do
    echo "============================================================"
    echo " 正在优化场景: $scene_name (6-View 精准策略)"
    echo "============================================================"

    # 执行训练
    python train.py \
      -s dataset/llff/$scene_name \
      -m output/$exp_name/$scene_name \
      --eval -r 8 --n_views 6 \
      --pre_densify \
      --pre_sparsity_threshold 0.5 \
      --pre_size_shrink 1.5 \
      --pre_perturb_strength 0.1 \
      \
      --pre_knn_neighbors 8 \
      --pre_opacity_scale 0.4 \
      --pre_min_consistency_views 3 \
      --pre_depth_error_limit 0.03

    # 注释说明 (Changes vs 3-View):
    # KNN: 6 -> 8 (数据密了，扩大邻域统计更稳健)
    # Opacity: 0.5 -> 0.3 (不需要那么实了，0.3融合更好)
    # MinViews: 2 -> 3 (6张图里至少3张一致，提升几何可信度)
    # Depth: 0.03 -> 0.02 (收紧误差，宁缺毋滥)

    # 执行离线渲染
    echo ">> 正在进行最终渲染评估..."
    python render.py \
      -m output/$exp_name/$scene_name \
      -r 8 \
      --quiet
done

# 汇总 6 视图总表
python metric.py --path output/$exp_name