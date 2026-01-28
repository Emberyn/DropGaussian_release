#!/bin/bash

# =================================================================
# 3DGS + SaGPD 自动化基准测试脚本 (3/6/9 视图)
# =================================================================

# 1. 定义场景列表 (LLFF 数据集)
SCENES=("fern" "flower" "fortress" "horns" "leaves" "orchids" "room" "trex")

# 2. 定义稀疏视图设置
VIEW_COUNTS=(3 6 9)

# 3. 设置输出根目录
OUTPUT_ROOT="output/llff_comprehensive_test"

# 4. 开始三层循环测试
for num_views in "${VIEW_COUNTS[@]}"
do
    echo "============================================================"
    echo ">>> 开始执行: ${num_views} 视图下的所有场景测试"
    echo "============================================================"

    for scene in "${SCENES[@]}"
    do
        echo "------------------------------------------------------------"
        echo "正在训练场景: $scene | 视图数: $num_views"
        echo "------------------------------------------------------------"

        # 运行训练脚本
        python train.py \
          -s dataset/llff/$scene \
          -m $OUTPUT_ROOT/views_${num_views}/$scene \
          --eval \
          -r 8 \
          --n_views $num_views \
          --pre_densify \
          --pre_knn_neighbors 8 \
          --pre_sparsity_threshold 0.7 \
          --pre_opacity_scale 0.3 \
          --pre_size_shrink 1.5 \
          --pre_perturb_strength 0.1 \
          --pre_min_consistency_views 2 \
          --pre_depth_error_limit 0.01

        echo "场景 $scene (${num_views} views) 训练及初步评估完成。"
    done

    # 每跑完一个视图组，执行一次汇总指标计算
    echo ">>> 正在汇总 ${num_views} 视图组的最终指标..."
    python metric.py -m $OUTPUT_ROOT/views_${num_views}
done

echo "============================================================"
echo "所有测试已完成！最终结果请查看 $OUTPUT_ROOT 目录下的各组指标总结。"
echo "============================================================"