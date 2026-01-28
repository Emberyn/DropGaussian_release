#!/bin/bash

# ==============================================================================
# aGPD-Lite++ 6-View Experiment: "High Sparsity + Aggressive Fix"
# ==============================================================================

exp_name="llff_sagpd_6view_final_strategy"
scenes=("fern" "flower" "fortress" "horns" "leaves" "orchids" "room" "trex")
#scenes=("fern" "flower")

# 安装缺失的库 (自动修复)
pip install scikit-learn > /dev/null 2>&1

for scene_name in "${scenes[@]}"
do
    echo ""
    echo "============================================================"
    echo " [Start] Optimizing: $scene_name"
    echo "============================================================"

    # 注意：反斜杠 \ 后不能有任何字符（包括空格和注释）
    python train.py \
      -s dataset/llff/$scene_name \
      -m output/$exp_name/$scene_name \
      --eval -r 8 --n_views 6 \
      --pre_densify \
      --pre_knn_neighbors 16 \
      --pre_min_consistency_views 2 \
      --pre_size_shrink 1.5 \
      --pre_opacity_scale 0.5 \
      --pre_sparsity_threshold 0.7 \
      --pre_dt_quantile 0.4 \
      --pre_len_threshold_mult 2.0 \
      --pre_eta_o_quantile 0.7 \
      --pre_align_ql 0.1 \
      --pre_align_qh 0.9 \
      --pre_ratio_clamp_min 0.5 \
      --pre_ratio_clamp_max 2.0 \
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