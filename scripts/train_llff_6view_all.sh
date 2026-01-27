#!/bin/bash

# ==============================================================================
# aGPD-Lite++ 6-View Experiment: "High Sparsity + Aggressive Fix"
#
# 参数说明 (Configuration):
# [Standard Params]
#   knn_neighbors=16: 稳健感知
#   min_consistency_views=2: 宽松过滤保证召回
#   size_shrink=1.5: 标准大小
#   opacity_scale=0.5: [Modified] 提高可见度 (原0.3)
#   sparsity_threshold=0.85: [Modified] 极高阈值避开Room墙面 (原0.7)
# [Advanced Params]
#   dt_quantile=0.4: [Modified] 激进生成 (原0.6)
#   len_threshold_mult=2.0: 标准长度判定
#   eta_o_quantile=0.7: [Modified] 收紧误差容忍 (原0.9)
#   ratio_clamp=[0.5, 2.0]: 宽松纠偏
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
      --pre_dt_quantile 0.6 \
      --pre_len_threshold_mult 2.0 \
      --pre_eta_o_quantile 0.9 \
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