#!/bin/bash

# ==============================================================================
# aGPD-Lite++ 3-View SOTA Strategy Script
#
# 核心策略: 场景分类处理 (Scene-Specific Optimization)
# 1. 结构组 (Fortress, Room, Trex):
#    -> 采用 [稳健参数]。利用 DPT 的平滑优势，压平墙面，减少噪点。
# 2. 细节组 (Fern, Flower, Horns, Leaves, Orchids):
#    -> 采用 [激进参数]。放宽 DPT 约束，降低稀疏门槛，强行修补细小缝隙。
# ==============================================================================

# 定义实验名称
exp_name="llff_sagpd_3view_sota"

# 定义场景列表
scenes=("fern" "flower" "fortress" "horns" "leaves" "orchids" "room" "trex")

# 定义结构主导型场景 (用于判断)
structured_scenes=("fortress" "room" "trex")

# 循环运行每个场景
for scene_name in "${scenes[@]}"
do
    echo ""
    echo "============================================================"
    echo " 正在优化场景: $scene_name"
    echo "============================================================"

    # --- [关键步骤] 参数自动判定逻辑 ---

    # 1. 默认设置为 [细节组/激进模式] (适合 Leaves, Horns 等)
    # 理由: 承认 DPT 在细节处不准，放宽标准，勤快补洞
    KNN_VAL=8               # 敏锐：感知小缝隙
    SPARSITY_VAL=0.5        # 勤快：修补前 50% 的稀疏区
    OPACITY_VAL=0.6         # 显眼：提高新点存活率
    DEPTH_ERR=0.05          # 宽容：允许 5%~20% 的误差 (对应代码逻辑)
    STRATEGY="Aggressive (Detail)"

    # 2. 判断是否为 [结构组]，如果是则切换为 [稳健模式]
    for s in "${structured_scenes[@]}"; do
        if [ "$scene_name" == "$s" ]; then
            KNN_VAL=16          # 宏观：关注整体平滑
            SPARSITY_VAL=0.7    # 懒惰：只修补大窟窿
            OPACITY_VAL=0.3     # 柔和：软着陆
            DEPTH_ERR=0.02      # 严苛：拒绝偏离平面的噪点
            STRATEGY="Conservative (Structured)"
            break
        fi
    done

    # 打印当前使用的策略参数
    echo " >> [策略判定] $STRATEGY"
    echo " >> 当前参数: KNN=$KNN_VAL | Sparsity=$SPARSITY_VAL | Opacity=$OPACITY_VAL | ErrLimit=$DEPTH_ERR"

    # 3. 执行训练
    python train.py \
      -s dataset/llff/$scene_name \
      -m output/$exp_name/$scene_name \
      --eval -r 8 --n_views 3 \
      --pre_densify \
      --pre_knn_neighbors $KNN_VAL \
      --pre_sparsity_threshold $SPARSITY_VAL \
      --pre_opacity_scale $OPACITY_VAL \
      --pre_size_shrink 1.5 \
      --pre_perturb_strength 0.0 \
      --pre_min_consistency_views 2 \
      --pre_depth_error_limit $DEPTH_ERR

    # 4. 执行渲染评估
    echo ">> [Render] 正在生成评估图像..."
    python render.py \
      -m output/$exp_name/$scene_name \
      -r 8 \
      --quiet
done

# 5. 汇总
echo ""
echo "============================================================"
echo " 所有场景运行完毕，正在汇总 SOTA 指标..."
echo "============================================================"
python metric.py --path output/$exp_name