# 导入核心依赖库
import os               # 操作系统路径处理（创建目录、拼接路径、判断文件存在性）
import torch            # PyTorch核心库（模型加载、推理、张量运算）
import cv2              # OpenCV库（图像读取、格式转换、保存）
import glob             # 文件通配符匹配（批量查找图片/目录）
import numpy as np      # 数值计算库（数组转换、归一化、类型转换）
from tqdm import tqdm   # 进度条库（可视化批量处理进度）
import sys              # 系统参数（本代码未直接使用，预留扩展）


def generate_depths_offline(dataset_root):
    """
    离线生成DPT深度图的核心函数
    功能：遍历指定数据集目录，对每个场景的RGB图运行DPT-Large模型，生成16bit高精度深度图，
          保存到对应场景的depths_dpt目录下（适配3DGS的SaGPD流程）
    参数：
        dataset_root: str - 数据集根目录（如"dataset/llff"），需符合llff数据集目录结构
    """
    print("\n[DPT Offline] Starting generation (High Quality Mode)...")

    # 1. 配置计算设备：优先使用GPU（CUDA），无GPU则用CPU（CPU推理速度极慢，仅应急）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] Using {device} for inference")

    # ==========================================
    # 2. 路径配置（核心：适配本地文件结构，避免联网依赖）
    # ==========================================
    # 本地MiDaS代码库路径（需提前解压MiDaS-master.zip，包含DPT模型定义）
    local_repo_path = os.path.abspath("MiDaS-master")  # os.path.abspath：转换为绝对路径，避免相对路径歧义
    # DPT-Large权重文件路径（需提前下载到当前目录，禁止联网自动下载）
    weights_path = "dpt_large_384.pt"

    # 检查关键文件是否存在：避免后续加载模型时报错
    if not os.path.exists(local_repo_path):
        print(f"[Error] Code folder not found: {local_repo_path}")
        print("Please run: unzip MiDaS-master.zip")  # 给出明确的修复指令
        return  # 终止函数，避免执行后续错误代码

    if not os.path.exists(weights_path):
        print(f"[Error] Weights file not found: {weights_path}")
        print("Please download dpt_large_384.pt from https://github.com/isl-org/MiDaS")
        return

    print(f"Loading code from: {local_repo_path}")
    print(f"Loading weights from: {weights_path}")

    try:
        # ==========================================
        # 3. 加载DPT-Large模型（核心步骤，规避联网依赖）
        # ==========================================
        # torch.hub.load：加载PyTorch Hub中的模型，这里强制使用本地代码
        # 参数说明：
        #   local_repo_path: 本地MiDaS代码库路径
        #   "DPT_Large": 要加载的模型名称（MiDaS中定义的DPT大模型）
        #   pretrained=False: 禁止自动下载预训练权重（避免联网）
        #   source='local': 强制从本地代码加载，而非PyTorch Hub仓库
        model = torch.hub.load(local_repo_path, "DPT_Large", pretrained=False, source='local')

        # 手动加载本地权重文件（替代联网下载）
        # torch.load: 加载权重文件，map_location=device避免GPU/CPU不匹配
        checkpoint = torch.load(weights_path, map_location=device)
        model.load_state_dict(checkpoint)  # 将权重加载到模型中
        model.to(device)                   # 将模型移到指定设备（GPU/CPU）
        model.eval()                       # 切换模型到评估模式（禁用Dropout/BatchNorm训练行为）

        # 加载DPT专用的图像变换（预处理）逻辑
        # DPT要求输入图像必须经过特定的归一化、尺寸调整，需加载官方变换函数
        midas_transforms = torch.hub.load(local_repo_path, "transforms", source='local')
        transform = midas_transforms.dpt_transform  # 选择DPT模型对应的变换（区别于MiDaS_small）

        print("[Success] Model loaded successfully!")

    except Exception as e:
        # 捕获模型加载的所有异常，给出针对性提示
        print(f"\n[Fatal Error] Loading failed: {e}")
        # 常见错误：缺少timm库（DPT依赖timm实现Vision Transformer）
        if "timm" in str(e):
            print(">> Please install timm: pip install timm")
        return  # 加载失败，终止函数

    # ==========================================
    # 4. 遍历数据集，批量处理图片生成深度图
    # ==========================================
    # 构造场景图片目录的搜索路径：dataset_root/*/images（适配llff数据集结构）
    # 例如：dataset/llff/fern/images、dataset/llff/flower/images
    search_path = os.path.join(dataset_root, "*", "images")
    # glob.glob：匹配所有符合路径规则的目录，返回列表
    scene_image_dirs = glob.glob(search_path)

    # 检查是否找到场景：避免后续空循环
    if not scene_image_dirs:
        print(f"[Warning] No scenes found in {dataset_root}")
        print(f">> Check if dataset path is correct, or if the dataset follows llff structure: {dataset_root}/<scene>/images")
        return

    print(f"\n[Process] Found {len(scene_image_dirs)} scenes.")

    # 遍历每个场景的图片目录
    for img_dir in scene_image_dirs:
        # 场景根目录：从images目录向上一级（如从fern/images到fern）
        scene_dir = os.path.dirname(img_dir)
        # 深度图输出目录：scene_dir/depths_dpt（适配3DGS的SaGPD流程的默认路径）
        out_dir = os.path.join(scene_dir, "depths_dpt")
        # 创建目录：exist_ok=True 避免目录已存在时报错
        os.makedirs(out_dir, exist_ok=True)

        # 查找当前场景下的所有图片文件
        img_files = glob.glob(os.path.join(img_dir, "*"))
        # 筛选有效图片格式：仅保留png/jpg/jpeg（排除txt/meta等无关文件）
        img_files = [f for f in img_files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.JPG'))]

        print(f"Processing {os.path.basename(scene_dir)} ({len(img_files)} images)...")

        # 遍历当前场景的所有图片，tqdm包裹实现进度条
        for img_path in tqdm(img_files):
            # 提取图片basename（不含后缀）：如"0001.jpg" → "0001"
            basename = os.path.splitext(os.path.basename(img_path))[0]
            # 深度图保存路径：depths_dpt/basename.png（16bit PNG格式）
            save_path = os.path.join(out_dir, basename + ".png")

            # 跳过已生成的深度图：避免重复计算，提升批量处理效率
            if os.path.exists(save_path):
                continue

            try:
                # -------------------------- 单张图片处理流程 --------------------------
                # 1. 读取图片：cv2.imread默认读取BGR格式（OpenCV标准）
                img = cv2.imread(img_path)
                if img is None:  # 图片损坏/路径错误时跳过
                    continue
                # 转换为RGB格式：DPT模型要求输入为RGB（与训练时一致）
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                # 2. 预处理：应用DPT专用变换（归一化、调整尺寸到384x384等）
                # transform是MiDaS提供的dpt_transform，返回的是适配模型输入的张量
                input_batch = transform(img).to(device)

                # 3. 模型推理：禁用梯度计算（eval模式+no_grad，节省显存+加速）
                with torch.no_grad():
                    # 前向传播：输入预处理后的张量，输出原始深度预测（低分辨率）
                    prediction = model(input_batch)

                    # 4. 插值回原图尺寸：DPT输出是384x384，需还原为输入图片的原始尺寸
                    # 语法说明：
                    #   unsqueeze(1)：扩展通道维度，从(1, H, W)→(1, 1, H, W)（适配interpolate输入要求）
                    #   size=img.shape[:2]：目标尺寸为原图的(H, W)（img.shape[:2] = (高度, 宽度)）
                    #   mode="bicubic"：双三次插值（比双线性更精细，适合深度图）
                    #   align_corners=False：像素中心对齐，避免边缘失真
                    #   squeeze()：删除维度为1的通道，还原为(H, W)的2D张量
                    prediction = torch.nn.functional.interpolate(
                        prediction.unsqueeze(1),
                        size=img.shape[:2],
                        mode="bicubic",
                        align_corners=False,
                    ).squeeze()

                # 4. 深度值归一化：转换为0~65535（适配16bit PNG存储）
                depth_min = prediction.min()  # 深度最小值（场景最近点）
                depth_max = prediction.max()  # 深度最大值（场景最远点）

                # 避免除零错误：如果深度值全为0（异常情况），直接设为0
                if depth_max - depth_min > 1e-8:
                    # 归一化到0~1：(x - min) / (max - min)
                    depth_norm = (prediction - depth_min) / (depth_max - depth_min)
                else:
                    depth_norm = torch.zeros_like(prediction)

                # 转换为16bit无符号整数：0~1 → 0~65535（16bit PNG的最大范围）
                # cpu().numpy()：从GPU张量转换为CPU的numpy数组（OpenCV仅支持numpy）
                # astype(np.uint16)：转换为16bit类型（区别于8bit的uint8，保留更高精度）
                depth_uint16 = (depth_norm * 65535.0).cpu().numpy().astype(np.uint16)

                # 5. 保存深度图：cv2.imwrite支持uint16格式的PNG保存
                cv2.imwrite(save_path, depth_uint16)

            except Exception as e:
                # 捕获单张图片处理的异常，不终止整个场景的处理
                print(f"\nError processing {img_path}: {e}")
                continue  # 跳过当前图片，处理下一张

    # 所有场景处理完成
    print("\n[Done] All depth maps generated. Now run your training script!")


# 主函数入口：仅当脚本直接运行时执行（import时不执行）
if __name__ == "__main__":
    # 指定数据集根目录：需替换为你的llff数据集路径
    DATASET_ROOT = "dataset/llff"
    # 调用深度图生成函数
    generate_depths_offline(DATASET_ROOT)