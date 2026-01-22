#!/bin/bash

# ==============================================================================
# SaGPD A++ Unified Experiment Script (Auto-Adaptive)
#
# 策略说明:
# 本脚本不再手动指定几何参数 (KNN/Depth/Opacity/MinViews)。
# 所有参数由 Python 代码根据 n_views (3/6/9/12/24) 自动推导，
# 实现了从"激进补洞"到"精准稳健"的自动切换。
# ==============================================================================

# --- 全局实验配置 ---
BASE_EXP_NAME="llff_sagpd_final_adaptive"
SCENES=("fern" "flower" "fortress" "horns" "leaves" "orchids" "room" "trex")
VIEW_COUNTS=(3 6)

# --- 唯一需要手动调节的参数 ---
# 稀疏度阈值依然由人工决定显存预算 (0.7 = 填充最稀疏的 30% 区域)
SPARSITY_THRESHOLD=0.7

# ===================== 开始循环 =====================

for n_views in "${VIEW_COUNTS[@]}"
do
    echo "############################################################"
    echo ">>> 启动实验: $n_views 视图 (参数由 Python 自动适配)"
    echo "############################################################"

    CURRENT_EXP_DIR="output/${BASE_EXP_NAME}/views_${n_views}"

    for scene_name in "${SCENES[@]}"
    do
        echo "------------------------------------------------------------"
        echo "Processing: $scene_name | Views: $n_views"
        echo "------------------------------------------------------------"

        # 1. 执行训练
        # 注意：这里的 --pre_knn_neighbors 等参数只是为了满足 argparse 接口
        # 实际数值会被 GaussianModel 内部的自适应逻辑覆盖
        python train.py \
          -s dataset/llff/$scene_name \
          -m $CURRENT_EXP_DIR/$scene_name \
          --eval -r 8 --n_views $n_views \
          --pre_densify \
          --pre_sparsity_threshold $SPARSITY_THRESHOLD \
          --pre_knn_neighbors 8 \
          --pre_opacity_scale 0.3 \
          --pre_size_shrink 1.5 \
          --pre_perturb_strength 0.1 \
          --pre_min_consistency_views 2 \
          --pre_depth_error_limit 0.02

        # 2. 渲染
        python render.py -m $CURRENT_EXP_DIR/$scene_name -r 8 --quiet
    done

    # 3. 汇总
    python metric.py --path $CURRENT_EXP_DIR

done

echo "============================================================"
echo "所有测试完成。请检查训练日志中的 [SaGPD Auto-Tuning] 输出以确认实际参数。"
echo "============================================================"