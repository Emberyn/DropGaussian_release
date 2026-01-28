# 1. 定义实验名称
exp_name="llff_sagpd_9view"

# 2. 定义场景列表
scenes=("fern" "flower" "fortress" "horns" "leaves" "orchids" "room" "trex")

# 循环运行每个场景
for scene_name in "${scenes[@]}"
do
    echo "============================================================"
    echo " 正在优化场景: $scene_name (9-View 模式)"
    echo "============================================================"

    # 3. 执行训练 (注意: n_views 设置为 9)
    python train.py \
      -s dataset/llff/$scene_name \
      -m output/$exp_name/$scene_name \
      --eval -r 8 --n_views 9 \
      --pre_densify \
      --pre_knn_neighbors 6 \
      --pre_sparsity_threshold 0.5 \
      --pre_opacity_scale 0.3 \
      --pre_size_shrink 1.5 \
      --pre_perturb_strength 0.05 \
      --pre_min_consistency_views 2 \
      --pre_depth_error_limit 0.05

    # 4. 执行渲染评估
    echo ">> 正在进行最终渲染评估..."
    python render.py \
      -m output/$exp_name/$scene_name \
      -r 8 \
      --quiet
done

# 5. 汇总 9 视图最终总表
echo "============================================================"
echo " 9 视图所有场景运行完毕，正在生成汇总看板..."
echo "============================================================"
python metric.py --path output/$exp_name