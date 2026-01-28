#!/bin/bash

# =================================================================
# 3DGS + SaGPD 自动化基准测试脚本 (3/6/9 视图)
# 整合最优参数：depth_error_limit=0.05, knn=6
# =================================================================

# 1. 定义场景列表 (LLFF 数据集)
SCENES=("fern" "flower" "fortress" "horns" "leaves" "orchids" "room" "trex")

# 2. 定义稀疏视图设置
VIEW_COUNTS=(3 6 9)

# 3. 设置输出根目录 (建议区分文件夹，避免覆盖)
OUTPUT_ROOT="output/llff_benchmark_final"

# 检查并创建根目录
mkdir -p $OUTPUT_ROOT

# 开始视图数循环
for num_views in "${VIEW_COUNTS[@]}"
do
    echo "============================================================"
    echo ">>> 开始执行: ${num_views} 视图组实验"
    echo "============================================================"

    # 循环运行每个场景
    for scene in "${SCENES[@]}"
    do
        echo "------------------------------------------------------------"
        echo "正在处理: Scene: $scene | Views: $num_views"
        echo "------------------------------------------------------------"

        # 定义当前实验的模型保存路径
        MODEL_PATH="$OUTPUT_ROOT/views_${num_views}/$scene"

        # 第一步：执行训练 (使用你验证过的 Exp1 最优参数)
        python train.py \
          -s dataset/llff/$scene \
          -m $MODEL_PATH \
          --eval -r 8 \
          --n_views $num_views \
          --pre_densify \
          --pre_knn_neighbors 6 \
          --pre_sparsity_threshold 0.5 \
          --pre_opacity_scale 0.3 \
          --pre_size_shrink 1.5 \
          --pre_perturb_strength 0.05 \
          --pre_min_consistency_views 2 \
          --pre_depth_error_limit 0.05

        # 第二步：执行离线渲染与最终指标计算
        # 这一步非常重要，它会生成最终用于汇总的 metrics_10000.txt
        echo ">> 正在进行场景 $scene 的最终渲染评估..."
        python render.py \
          -m $MODEL_PATH \
          -r 8 \
          --quiet

    done

    # 第三步：每跑完一个视图组，立即汇总生成该组的 Mean 表格
    echo ">>> 正在汇总 ${num_views} 视图组的平均指标..."
    python metric.py --path $OUTPUT_ROOT/views_${num_views}
done

echo "============================================================"
echo "所有基准测试已完成！"
echo "结果分布："
echo "3 视图结果: $OUTPUT_ROOT/views_3"
echo "6 视图结果: $OUTPUT_ROOT/views_6"
echo "9 视图结果: $OUTPUT_ROOT/views_9"
echo "============================================================"