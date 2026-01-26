#!/bin/bash

# ==============================================================================
# aGPD-Lite++ 6-View Experiment: "High Sparsity + Aggressive Fix"
# ==============================================================================

exp_name="llff_sagpd_6view_final_strategy"
scenes=("fern" "flower" "fortress" "horns" "leaves" "orchids" "room" "trex")

for scene_name in "${scenes[@]}"
do
    echo ""
    echo "============================================================"
    echo " [Start] Optimizing: $scene_name"
    echo "============================================================"

    python train.py \
      -s dataset/llff/$scene_name \
      -m output/$exp_name/$scene_name \
      --eval -r 8 --n_views 6 \
      \
      --pre_densify \
      \
      # --- [基础参数] ---
      # KNN邻居数 (默认: 16)
      --pre_knn_neighbors 16 \
      # 最小一致性视角 M (默认: 2)
      --pre_min_consistency_views 2 \
      # 尺寸收缩系数 (默认: 1.5)
      --pre_size_shrink 1.5 \
      # 不透明度系数 (默认: 0.3)
      --pre_opacity_scale 0.5 \
      # 稀疏度阈值 (默认: 0.7)
      --pre_sparsity_threshold 0.85 \
      \
      # --- [高级参数] ---
      # 长边判定分位数 dt (默认: 0.6)
      --pre_dt_quantile 0.4 \
      # 边长倍数 (默认: 2.0)
      --pre_len_threshold_mult 2.0 \
      # 动态误差阈值 eta_o (默认: 0.9)
      --pre_eta_o_quantile 0.7 \
      # DPT对齐下分位 (默认: 0.1)
      --pre_align_ql 0.1 \
      # DPT对齐上分位 (默认: 0.9)
      --pre_align_qh 0.9 \
      # 几何纠偏下限 (默认: 0.5)
      --pre_ratio_clamp_min 0.5 \
      # 几何纠偏上限 (默认: 2.0)
      --pre_ratio_clamp_max 2.0 \
      # 对齐最少可见点数 (默认: 50)
      --pre_visible_count_threshold 50

    # 执行渲染评估
    echo ">> [Render] Rendering..."
    python render.py \
      -m output/$exp_name/$scene_name \
      -r 8 \
      --quiet

    echo ">> [Done] Scene $scene_name finished."
done

# 汇总
echo ""
echo "============================================================"
echo " Summarizing Metrics..."
echo "============================================================"
python metric.py --path output/$exp_name