#!/bin/bash

# ==============================================================================
# SaGPD A++ Final Experiment Script
# 目标: 在 3-view 和 6-view 下全面反超 DropGaussian (PSNR/SSIM/LPIPS)
# 策略: 混合策略 (Hybrid Strategy)
#       1. 几何: KNN=6, Depth=0.05 (放宽约束，确保 Horns 结构完整)
#       2. 探索: Perturb=0.1 (大扰动，增加稀疏点命中真实表面的概率)
#       3. 优化: Opacity=0.5 (高透明度，强力清洗背景杂点，提升 PSNR)
# ==============================================================================

# 定义实验基础名称
base_exp_name="llff_sagpd_final_repro"

# 定义场景列表
scenes=("fern" "flower" "fortress" "horns" "leaves" "orchids" "room" "trex")

# 定义要运行的视图组 (3视图 和 6视图)
view_counts=(3 6)

# ===================== 开始循环 =====================

for n_views in "${view_counts[@]}"
do
    echo "############################################################"
    echo ">>> 开始执行: $n_views 视图组实验"
    echo "############################################################"

    # 定义当前视图组的输出路径
    current_exp_dir="output/${base_exp_name}/views_${n_views}"

    for scene_name in "${scenes[@]}"
    do
        echo "------------------------------------------------------------"
        echo "正在处理: Scene: $scene_name | Views: $n_views"
        echo "------------------------------------------------------------"

        # 1. 执行训练 (Train)
        # 参数说明:
        # --pre_knn_neighbors 6           : [保持] 较小的K值，对细微结构(Horns)更敏感
        # --pre_sparsity_threshold 0.7    : [回调] 恢复到0.7，保证足够的稀疏区覆盖率
        # --pre_opacity_scale 0.5         : [核心] 提至0.5，让新增点更"实"，加速收敛和噪点剔除
        # --pre_size_shrink 1.5           : [标准] 标准缩放衰减
        # --pre_perturb_strength 0.1      : [回调] 恢复到0.1，利用大扰动增加命中率
        # --pre_min_consistency_views 2   : [标准] 最少2视角的几何校验
        # --pre_depth_error_limit 0.05    : [保持] 宽松的深度误差，容忍SfM的不确定性

        python train.py \
          -s dataset/llff/$scene_name \
          -m $current_exp_dir/$scene_name \
          --eval -r 8 --n_views $n_views \
          --pre_densify \
          --pre_knn_neighbors 6 \
          --pre_sparsity_threshold 0.7 \
          --pre_opacity_scale 0.5 \
          --pre_size_shrink 1.5 \
          --pre_perturb_strength 0.1 \
          --pre_min_consistency_views 2 \
          --pre_depth_error_limit 0.05

        # 2. 执行渲染与评估 (Render)
        echo ">> 正在进行场景 $scene_name 的最终渲染评估..."
        python render.py \
          -m $current_exp_dir/$scene_name \
          -r 8 \
          --quiet
    done

    # 3. 汇总当前视图组指标 (Metrics)
    echo ">>> 正在汇总 $n_views 视图组的平均指标..."
    python metric.py --path $current_exp_dir

done

echo "============================================================"
echo "所有基准测试已完成！"
echo "3 视图结果: output/${base_exp_name}/views_3"
echo "6 视图结果: output/${base_exp_name}/views_6"
echo "============================================================"