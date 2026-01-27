#!/bin/bash

# ========================================================
# 快速验证 SaGPD 指标汇总表格 (Init Pts & Added Pts)
# ========================================================

BASE_DIR="test_output_mock"
ITERATION="10000"
PYTHON_SCRIPT="metrics_summary_test.py"

echo ">> [1/4] 正在清理旧的测试环境..."
rm -rf $BASE_DIR
rm -f $PYTHON_SCRIPT
mkdir -p $BASE_DIR

echo ">> [2/4] 生成模拟数据 (模拟 train.py 的输出)..."

# --- 场景 1: Fern (模拟标准格式) ---
mkdir -p "$BASE_DIR/fern"
cat <<EOF > "$BASE_DIR/fern/metrics_${ITERATION}.txt"
Initial_Points : 10000
Added_Points : 150
PSNR : 25.5000
SSIM : 0.8500
LPIPS : 0.1200
EOF

# --- 场景 2: Flower (模拟另一种标准格式) ---
mkdir -p "$BASE_DIR/flower"
cat <<EOF > "$BASE_DIR/flower/metrics_${ITERATION}.txt"
Initial_Points : 20000
Added_Points : 300
PSNR : 26.5000
SSIM : 0.8600
LPIPS : 0.1100
EOF

# --- 场景 3: Trex (模拟杂乱格式 - 测试正则鲁棒性) ---
# 注意：没有空格，或者顺序打乱
mkdir -p "$BASE_DIR/trex"
cat <<EOF > "$BASE_DIR/trex/metrics_${ITERATION}.txt"
PSNR: 23.5
SSIM:0.87
Initial_Points:35000
Added_Points : 220
LPIPS : 0.13
EOF

echo "   已生成 3 个场景的模拟 metrics 文件。"

echo ">> [3/4] 生成 Python 汇总脚本 (包含最新修改)..."

# 将最新的 Python 代码写入临时文件
cat <<'PY_EOF' > $PYTHON_SCRIPT
import os
from glob import glob
from argparse import ArgumentParser
import re

parser = ArgumentParser("测试汇总")
parser.add_argument("--path", "-s", required=True, type=str)
parser.add_argument("--iteration", "-i", required=False, type=str, default='10000')
args = parser.parse_args()

PSNR_total = 0
SSIM_total = 0
LPIPS_total = 0
Added_total = 0
Init_total = 0
results = []
valid_count = 0

print(f"正在扫描目录: {args.path} ...")
dir_lst = sorted(glob(args.path + '/*'))

for d in dir_lst:
    if not os.path.isdir(d): continue
    file_path = os.path.join(d, 'metrics_{}.txt'.format(args.iteration))
    if not os.path.exists(file_path): continue

    scene_name = os.path.basename(d)
    try:
        with open(file_path, 'r') as f:
            content = f.read()

            # 正则提取 - 能够处理有无空格、不同顺序的情况
            psnr_match = re.search(r"PSNR\s*[:]\s*([-+]?\d*\.\d+|\d+)", content)
            psnr = float(psnr_match.group(1)) if psnr_match else 0.0

            ssim_match = re.search(r"SSIM\s*[:]\s*([-+]?\d*\.\d+|\d+)", content)
            ssim = float(ssim_match.group(1)) if ssim_match else 0.0

            lpips_match = re.search(r"LPIPS\s*[:]\s*([-+]?\d*\.\d+|\d+)", content)
            lpips = float(lpips_match.group(1)) if lpips_match else 0.0

            # 关键：点数提取
            added_pts = 0
            added_match = re.search(r"Added_Points\s*[:]\s*(\d+)", content)
            if added_match: added_pts = int(added_match.group(1))

            init_pts = 0
            init_match = re.search(r"Initial_Points\s*[:]\s*(\d+)", content)
            if init_match: init_pts = int(init_match.group(1))

            results.append({
                "scene": scene_name, "psnr": psnr, "ssim": ssim, "lpips": lpips,
                "added": added_pts, "init": init_pts
            })

            PSNR_total += psnr; SSIM_total += ssim; LPIPS_total += lpips
            Added_total += added_pts; Init_total += init_pts
            valid_count += 1
    except Exception as e:
        print(f"Error reading {scene_name}: {e}")

if valid_count > 0:
    PSNR_avg = PSNR_total / valid_count
    SSIM_avg = SSIM_total / valid_count
    LPIPS_avg = LPIPS_total / valid_count
    Added_avg = Added_total / valid_count
    Init_avg = Init_total / valid_count

    # 打印表格
    header = f"{'场景 (Scene)':<15} | {'Init Pts':^10} | {'Added Pts':^10} | {'PSNR (↑)':^10} | {'SSIM (↑)':^10} | {'LPIPS (↓)':^10}"
    sep = "-" * len(header)
    print("\n" + "=" * len(header))
    print(header)
    print(sep)
    for r in results:
        print(f"{r['scene']:<15} | {r['init']:^10} | {r['added']:^10} | {r['psnr']:^10.4f} | {r['ssim']:^10.4f} | {r['lpips']:^10.4f}")
    print(sep)
    print(f"{'均值 (Mean)':<15} | {int(Init_avg):^10} | {int(Added_avg):^10} | {PSNR_avg:^10.4f} | {SSIM_avg:^10.4f} | {LPIPS_avg:^10.4f}")
    print("=" * len(header) + "\n")
else:
    print("未找到有效数据！")
PY_EOF

echo ">> [4/4] 运行测试..."
python $PYTHON_SCRIPT -s $BASE_DIR -i $ITERATION

# 清理
echo ">> 清理临时文件..."
rm -rf $BASE_DIR
rm -f $PYTHON_SCRIPT

echo ">> 测试完成！如果你能看到上面的表格包含 'Init Pts' 和 'Added Pts' 列，说明逻辑正确。"