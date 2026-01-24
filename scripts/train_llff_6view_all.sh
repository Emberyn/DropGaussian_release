#!/bin/bash

# ==============================================================================
# aGPD-Lite++ 6-View Experiment
#
# 策略: 使用 6 张训练视图。
#       相比 3 视图，数据更丰富，aGPD 的几何纠偏(Refine)将更加精准。
# ==============================================================================

# 1. 实验名称
exp_name="llff_sagpd_6view_final"

# 2. 场景列表 (LLFF 数据集)
scenes=("fern" "flower" "fortress" "horns" "leaves" "orchids" "room" "trex")

# 3. 循环训练
for scene_name in "${scenes[@]}"
do
    echo ""
    echo "============================================================"
    echo " [Start] 正在优化场景: $scene_name (6-View aGPD-Lite++)"
    echo "============================================================"

    # 执行训练
    # 关键参数修改: --n_views 6
    #
    # 参数微调说明:
    # --pre_min_consistency_views 2:
    #   虽然有6张图，但为了防止在边缘区域过度过滤，我们保持 M=2。
    #   (如果您希望获得极高精度的核心几何，可以将其改为 3)
    #
    # --pre_knn_neighbors 16: 保持 K=16 以获得稳定的稀疏度估计。

    python train.py \
      -s dataset/llff/$scene_name \
      -m output/$exp_name/$scene_name \
      --eval -r 8 --n_views 6 \
      --pre_densify \
      --pre_knn_neighbors 16 \
      --pre_sparsity_threshold 0.7 \
      --pre_opacity_scale 0.3 \
      --pre_size_shrink 2.0 \
      --pre_perturb_strength 0.0 \
      --pre_min_consistency_views 2 \
      --pre_depth_error_limit 0.03

    # 执行渲染评估
    echo ">> [Render] 正在生成评估图像..."
    python render.py \
      -m output/$exp_name/$scene_name \
      -r 8 \
      --quiet

    echo ">> [Done] 场景 $scene_name 完成。"
done

# 4. 汇总所有场景指标
echo ""
echo "============================================================"
echo " 正在汇总所有场景指标..."
echo "============================================================"
python metric.py --path output/$exp_name