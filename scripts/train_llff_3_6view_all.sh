#!/bin/bash

# ==============================================================================
# aGPD-Lite++ Highly Customizable Experiment Script
#
# [使用说明]:
# 所有核心参数均已提取至脚本顶部的 "配置区"。
# 您可以分别针对 3-View 和 6-View 调整一套完全不同的参数组合。
# ==============================================================================

# 定义实验名称
BASE_EXP_NAME="llff_sagpd_custom_exposed"

# 定义场景列表
SCENES=("fern" "flower" "fortress" "horns" "leaves" "orchids" "room" "trex")

# 定义视图数
VIEW_COUNTS=(3 6)

# ==============================================================================
# [配置区] - 请在此处修改参数
# ==============================================================================

# --- 3-View 配置 (极稀疏场景) ---
# 建议策略: 激进 (Aggressive) 或 微雕 (Micro-Splat)
V3_KNN=16                # 邻域搜索数: 越小越敏锐 (推荐 8 或 16)
V3_SPARSITY=0.5         # 稀疏阈值: 越小越勤快 (推荐 0.5)
V3_OPACITY=0.3          # 初始不透明度: 越大越容易存活 (推荐 0.4 - 0.6)
V3_SHRINK=1.5           # 尺度收缩: 1.5(默认/平滑) vs 3.0(微雕/细节)
V3_PERTURB=0.0          # 随机扰动: 0.0(Refine模式推荐) vs 0.2(混沌模式)
V3_MIN_VIEWS=2          # 最小一致性视角: 固定为 2

# --- 6-View 配置 (稀疏场景) ---
# 建议策略: 平衡 (Balanced)
V6_KNN=16               # 依然保持敏锐，因为 LLFF 细节依然是痛点
V6_SPARSITY=0.5         # 稍微收敛，节省显存
V6_OPACITY=0.3          # 保持一致
V6_SHRINK=1.5           # 6视图下覆盖率较好，通常不需要微雕
V6_PERTURB=0.0          # 禁用扰动
V6_MIN_VIEWS=2          # 固定为 2

# ==============================================================================
# [逻辑区] - 以下代码通常无需修改
# ==============================================================================

for n_views in "${VIEW_COUNTS[@]}"
do
    echo ""
    echo "############################################################"
    echo ">>> 启动实验: $n_views 视图"
    echo "############################################################"

    CURRENT_EXP_DIR="output/${BASE_EXP_NAME}/views_${n_views}"

    # 根据视图数加载对应的配置变量
    if [ "$n_views" -eq 3 ]; then
        KNN_VAL=$V3_KNN
        SPARSITY_VAL=$V3_SPARSITY
        OPACITY_VAL=$V3_OPACITY
        SHRINK_VAL=$V3_SHRINK
        PERTURB_VAL=$V3_PERTURB
        MIN_VIEWS_VAL=$V3_MIN_VIEWS
    elif [ "$n_views" -eq 6 ]; then
        KNN_VAL=$V6_KNN
        SPARSITY_VAL=$V6_SPARSITY
        OPACITY_VAL=$V6_OPACITY
        SHRINK_VAL=$V6_SHRINK
        PERTURB_VAL=$V6_PERTURB
        MIN_VIEWS_VAL=$V6_MIN_VIEWS
    fi

    # 打印当前使用的参数表
    echo " >> [当前参数表]"
    echo "    |-- KNN Neighbors : $KNN_VAL"
    echo "    |-- Sparsity Thres: $SPARSITY_VAL"
    echo "    |-- Opacity Scale : $OPACITY_VAL"
    echo "    |-- Size Shrink   : $SHRINK_VAL"
    echo "    |-- Perturb Str   : $PERTURB_VAL"
    echo "    |-- Min Views     : $MIN_VIEWS_VAL"
    echo "    |-- Depth Error   : (代码内置 0.1*Z)"
    echo "------------------------------------------------------------"

    for scene_name in "${SCENES[@]}"
    do
        echo " 正在优化场景: $scene_name ..."

        # 1. 执行训练
        python train.py \
          -s dataset/llff/$scene_name \
          -m $CURRENT_EXP_DIR/$scene_name \
          --eval -r 8 --n_views $n_views \
          --pre_densify \
          --pre_knn_neighbors $KNN_VAL \
          --pre_sparsity_threshold $SPARSITY_VAL \
          --pre_opacity_scale $OPACITY_VAL \
          --pre_size_shrink $SHRINK_VAL \
          --pre_perturb_strength $PERTURB_VAL \
          --pre_min_consistency_views $MIN_VIEWS_VAL

        # 2. 执行渲染评估
        echo "    >> 生成评估图像..."
        python render.py \
          -m $CURRENT_EXP_DIR/$scene_name \
          -r 8 \
          --quiet
    done

    # 3. 汇总当前视图数的所有指标
    echo ""
    echo "============================================================"
    echo " [Summary] $n_views 视图实验结束，汇总指标..."
    echo "============================================================"
    python metric.py --path $CURRENT_EXP_DIR

done

echo "============================================================"
echo " 所有实验圆满完成！"
echo "============================================================"