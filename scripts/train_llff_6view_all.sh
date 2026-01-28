#!/bin/bash
# ==============================================================================
# aGPD-Lite++ 6-View Experiment: "High Sparsity + Aggressive Fix"
# ==============================================================================

exp_name="llff_sagpd_6view_final_strategy"
scenes=("fern" "flower" "fortress" "horns" "leaves" "orchids" "room" "trex")

# 安装缺失的库 (自动修复)
pip install scikit-learn > /dev/null 2>&1

for scene_name in "${scenes[@]}"
do
    echo ""
    echo "============================================================"
    echo " [Start] Optimizing: $scene_name"
    echo "============================================================"

    # 注意：反斜杠 \ 后不能有任何字符（包括空格和注释），注释必须写在同一行或另起一行
    # 下面的注释为了方便阅读写在行尾，复制到终端执行时请确保环境支持，或删除注释
    python train.py \
      -s dataset/llff/$scene_name \
      -m output/$exp_name/$scene_name \
      --eval -r 8 --n_views 6 \
      --pre_densify \
      --pre_knn_neighbors 16 \
      --pre_min_consistency_views 2 \
      --pre_size_shrink 1.5 \
      --pre_opacity_scale 0.5 \
      --pre_sparsity_threshold 0.4 \
      --pre_dt_quantile 0.6 \
      --pre_len_threshold_mult 2.0 \
      --pre_eta_o_quantile 0.9 \
      --pre_align_ql 0.1 \
      --pre_align_qh 0.9 \
      --pre_ratio_clamp_min 0.5 \
      --pre_ratio_clamp_max 2.0 \
      --pre_visible_count_threshold 50


# --pre_densify        | 核心开关：启用SaGPD预致密化算法 | 无（开关型参数，启用即生效）
# --pre_knn_neighbors  | KNN邻居数（计算局部密度）| 16
# --pre_sparsity_threshold | 稀疏度触发阈值（仅对最稀疏区域补点） | 0.7
# --pre_min_consistency_views | 新点需通过的最小验证视角数 | 2
# --pre_size_shrink    | 新点尺寸收缩系数 | 1.5
# --pre_opacity_scale  | 新点不透明度缩放系数 | 0.5
# --pre_dt_quantile    | 全局距离基准分位数 | 0.6
# --pre_len_threshold_mult | 候选点最大跨度倍数 | 2.0
# --pre_eta_o_quantile | 深度误差容忍度分位数 | 0.9
# --pre_align_ql       | DPT对齐下分位数 | 0.1
# --pre_align_qh       | DPT对齐上分位数 | 0.9
# --pre_visible_count_threshold | DPT对齐最小可见点数 | 50
# --pre_ratio_clamp_min | 几何纠偏最小缩放比例 | 0.5
# --pre_ratio_clamp_max | 几何纠偏最大缩放比例 | 2.0






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