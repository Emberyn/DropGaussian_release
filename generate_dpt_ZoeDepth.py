# generate_dpt_ZoeDepth.py (Offline FINAL)
import os
import torch
import cv2
import glob
import numpy as np
from tqdm import tqdm
from PIL import Image
import torch.nn.functional as F

# ==========================================
# 1. Monkey Patch (修复 Numpy 类型报错)
# ==========================================
_original_interpolate = F.interpolate


def _patched_interpolate(input, size=None, scale_factor=None, mode='nearest', align_corners=None,
                         recompute_scale_factor=None, antialias=False):
    if size is not None:
        if isinstance(size, (tuple, list)):
            size = tuple(int(x) for x in size)
        else:
            size = int(size)
    return _original_interpolate(input, size, scale_factor, mode, align_corners, recompute_scale_factor, antialias)


F.interpolate = _patched_interpolate
print("[System] Monkey Patch applied.")


# ==========================================

def generate_depths_zoe(dataset_root):
    print(f"\n[ZoeDepth] Scanning dataset root: {dataset_root}")

    # 查找场景
    search_path = os.path.join(dataset_root, "*", "images")
    scene_image_dirs = glob.glob(search_path)
    scene_image_dirs.sort()

    if not scene_image_dirs:
        print(f"[Error] No scenes found in {dataset_root}")
        return

    print(f"[Process] Found {len(scene_image_dirs)} scenes.")

    # ==========================================
    # 2. 强制离线加载模型
    # ==========================================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[Device] Using {device}")

    # [关键路径 1] 你的权重文件 (在当前目录)
    local_weight_path = "ZoeD_M12_NK.pt"

    # [关键路径 2] 你的代码缓存目录 (基于你之前的日志)
    # 既然之前下载过，这里一定有文件。我们强制读这里。
    local_code_path = "/root/.cache/torch/hub/isl-org_ZoeDepth_main"

    if not os.path.exists(local_weight_path):
        print(f"\n[Error] 权重文件没找到: {local_weight_path}")
        return

    if not os.path.exists(local_code_path):
        print(f"\n[Fatal Error] 代码缓存目录不存在: {local_code_path}")
        print("请检查路径是否被删除。")
        return

    try:
        print(f"Loading ZoeDepth from local cache: {local_code_path}")

        # [核心] source='local' 彻底断网，直接读硬盘
        model = torch.hub.load(local_code_path, "ZoeD_NK", source='local', pretrained=False)

        print(f"Loading weights from: {local_weight_path}")
        ckpt = torch.load(local_weight_path, map_location=device)

        if isinstance(ckpt, dict) and 'model' in ckpt:
            state_dict = ckpt['model']
        else:
            state_dict = ckpt

        # strict=False 忽略多余键值
        model.load_state_dict(state_dict, strict=False)
        model = model.to(device).eval()
        print("[Success] Model loaded successfully (Offline Mode)!")

    except Exception as e:
        print(f"\n[Error] Loading failed: {e}")
        # 如果是因为缺依赖包(如 timm)，这里会报错
        return

    # ==========================================
    # 3. 循环处理
    # ==========================================
    for img_dir in scene_image_dirs:
        scene_dir = os.path.dirname(img_dir)
        scene_name = os.path.basename(scene_dir)
        out_dir = os.path.join(scene_dir, "depths_dpt_ZoeDepth")
        os.makedirs(out_dir, exist_ok=True)

        img_files = glob.glob(os.path.join(img_dir, "*"))
        img_files = [f for f in img_files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.JPG'))]
        img_files.sort()

        print(f"\n>> Processing: [{scene_name}] ({len(img_files)} imgs)")

        for img_path in tqdm(img_files, desc=scene_name):
            basename = os.path.splitext(os.path.basename(img_path))[0]
            save_path = os.path.join(out_dir, basename + ".png")

            if os.path.exists(save_path):
                continue

            try:
                img_pil = Image.open(img_path).convert("RGB")
                with torch.no_grad():
                    depth_metric = model.infer_pil(img_pil)

                d_min, d_max = depth_metric.min(), depth_metric.max()
                if d_max - d_min > 1e-8:
                    depth_norm = (depth_metric - d_min) / (d_max - d_min)
                else:
                    depth_norm = torch.zeros_like(torch.from_numpy(depth_metric)).numpy()

                depth_uint16 = (depth_norm * 65535.0).astype(np.uint16)
                cv2.imwrite(save_path, depth_uint16)

            except Exception as e:
                print(f"[Error] {img_path}: {e}")
                continue

    print("\n[All Done] Processed all scenes.")


if __name__ == "__main__":
    DATASET_ROOT = "dataset/llff"
    if not os.path.exists(DATASET_ROOT):
        print(f"Error: Dataset root '{DATASET_ROOT}' does not exist.")
    else:
        generate_depths_zoe(DATASET_ROOT)