#!/bin/bash
# ==============================================================================
# aGPD-Lite++ Ultimate Run: "Floodgate Open"
# ==============================================================================

exp_name="llff_sagpd_ultimate"
scenes=("fern" "flower" "fortress" "horns" "leaves" "orchids" "room" "trex")

pip install scikit-learn > /dev/null 2>&1

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
      --pre_densify \
      --pre_knn_neighbors 16 \
      --pre_min_consistency_views 2 \
      --pre_size_shrink 1.0 \
      --pre_opacity_scale 0.95 \
      --pre_sparsity_threshold 0.40 \
      --pre_dt_quantile 0.2 \
      --pre_len_threshold_mult 0.8 \
      --pre_eta_o_quantile 0.95 \
      --pre_align_ql 0.05 \
      --pre_align_qh 0.95 \
      --pre_ratio_clamp_min 0.2 \
      --pre_ratio_clamp_max 3.0 \
      --pre_visible_count_threshold 30

    python render.py -m output/$exp_name/$scene_name -r 8 --quiet
    echo ">> [Done] Scene $scene_name finished."
done

python metric.py --path output/$exp_name