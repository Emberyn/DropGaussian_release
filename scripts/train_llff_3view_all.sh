# 定义实验名称
exp_name="llff_sagpd_3view"

# 定义场景列表
scenes=("fern" "flower" "fortress" "horns" "leaves" "orchids" "room" "trex")

# 循环运行每个场景
for scene_name in "${scenes[@]}"
do
    echo "============================================================"
    echo " 正在优化场景: $scene_name"
    echo "============================================================"

    # 1. 执行训练
    python train.py \
      -s dataset/llff/$scene_name \
      -m output/$exp_name/$scene_name \
      --eval -r 8 --n_views 3 \
      --pre_densify \
      --pre_depth_error_limit 0.01

    # 2. 执行离线渲染与最终评估 (生成 metrics_10000.txt)
    echo ">> 正在进行最终渲染评估..."
    python render.py \
      -m output/$exp_name/$scene_name \
      -r 8 \
      --quiet
done

# 3. 循环结束后，汇总 8 个场景的总分
echo "============================================================"
echo " 所有场景运行完毕，正在汇总平均指标..."
echo "============================================================"
python metric.py --path output/$exp_name