#!/bin/bash

# =================================================================
# 3DGS Baseline Benchmark Script (No Pre-densification)
# 覆盖 3/6/9 视图，仅使用原生 3DGS 训练逻辑
# =================================================================

# 1. 定义场景列表 (LLFF 数据集)
SCENES=("fern" "flower" "fortress" "horns" "leaves" "orchids" "room" "trex")

# 2. 定义稀疏视图设置
VIEW_COUNTS=(3 6 9)

# 3. 设置输出根目录 (与 SaGPD 实验区分开)
OUTPUT_ROOT="output/llff_no_densify_benchmark"

# 检查并创建根目录
mkdir -p $OUTPUT_ROOT

# 开始视图数循环
for num_views in "${VIEW_COUNTS[@]}"
do
    echo "============================================================"
    echo ">>> 开始执行: ${num_views} 视图组实验 (纯净版 3DGS)"
    echo "============================================================"

    # 循环运行每个场景
    for scene in "${SCENES[@]}"
    do
        echo "------------------------------------------------------------"
        echo "正在处理: Scene: $scene | Views: $num_views"
        echo "------------------------------------------------------------"

        # 定义当前实验的模型保存路径
        MODEL_PATH="$OUTPUT_ROOT/views_${num_views}/$scene"

        # 第一步：执行训练
        # 注意：这里移除了所有 --pre_densify 相关的参数
        python train.py \
          -s dataset/llff/$scene \
          -m $MODEL_PATH \
          --eval -r 8 \
          --n_views $num_views

        # 第二步：执行离线渲染与最终指标计算
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
echo "所有 Baseline (无预密集) 实验已完成！"
echo "结果分布："
echo "3 视图结果: $OUTPUT_ROOT/views_3"
echo "6 视图结果: $OUTPUT_ROOT/views_6"
echo "9 视图结果: $OUTPUT_ROOT/views_9"
echo "============================================================"