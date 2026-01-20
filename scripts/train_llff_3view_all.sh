# 定义场景列表
scenes=("fern" "flower" "fortress" "horns" "leaves" "orchids" "room" "trex")

# 循环运行每个场景
for scene_name in "${scenes[@]}"
do
    echo "------------------------------------------------------------"
    echo "正在测试场景: $scene_name (3-View SaGPD)"
    echo "------------------------------------------------------------"

    python train.py \
      -s dataset/llff/$scene_name \
      -m output/llff_sagpd_3view/$scene_name \
      --eval \
      -r 8 \
      --n_views 3 \
      --pre_densify \
      --pre_knn_neighbors 8 \
      --pre_sparsity_threshold 0.7 \
      --pre_opacity_scale 0.3 \
      --pre_size_shrink 1.5 \
      --pre_perturb_strength 0.1 \
      --pre_min_consistency_views 2 \
      --pre_depth_error_limit 0.01
done