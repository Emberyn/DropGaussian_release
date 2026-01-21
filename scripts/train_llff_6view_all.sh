# 1. 修改实验名称，防止覆盖 3 视图数据
exp_name="llff_sagpd_6view"

# 定义场景列表
scenes=("fern" "flower" "fortress" "horns" "leaves" "orchids" "room" "trex")

for scene_name in "${scenes[@]}"
do
    echo "============================================================"
    echo " 正在优化场景: $scene_name (6-View模式)"
    echo "============================================================"

    # 执行训练 - 确保 n_views 是 6
    python train.py \
      -s dataset/llff/$scene_name \
      -m output/$exp_name/$scene_name \
      --eval -r 8 --n_views 6 \
      --pre_densify \
      --pre_knn_neighbors 6 \
      --pre_sparsity_threshold 0.5 \
      --pre_opacity_scale 0.3 \
      --pre_size_shrink 1.5 \
      --pre_perturb_strength 0.05 \
      --pre_min_consistency_views 2 \
      --pre_depth_error_limit 0.05

    # 执行离线渲染
    echo ">> 正在进行最终渲染评估..."
    python render.py \
      -m output/$exp_name/$scene_name \
      -r 8 \
      --quiet
done

# 汇总 6 视图总表
python metric.py --path output/$exp_name