import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
from utils.general_utils import PILtoTorch
from utils.image_utils import psnr
from utils.loss_utils import ssim
from lpipsPyTorch import lpips


def render_set(model_path, name, iteration, views, gaussians, pipeline, background):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    PSNR = []
    SSIM = []
    LPIPS = []

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        gt = view.original_image[0:3, :, :].cuda()
        render_pkg = render(view, gaussians, pipeline, background)
        rendering = render_pkg["render"]
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
        PSNR.append(psnr(rendering.unsqueeze(0), gt.unsqueeze(0)))
        SSIM.append(ssim(rendering.unsqueeze(0), gt.unsqueeze(0)))
        LPIPS.append(lpips(rendering.unsqueeze(0), gt.unsqueeze(0), net_type='vgg'))

    psnr_mean = torch.tensor(PSNR).mean().item()
    ssim_mean = torch.tensor(SSIM).mean().item()
    lpips_mean = torch.tensor(LPIPS).mean().item()

    print('\n[Evaluation] {} Set Results:'.format(name))
    print('PSNR : {:>12.7f}'.format(psnr_mean))
    print('SSIM : {:>12.7f}'.format(ssim_mean))
    print('LPIPS : {:>12.7f}'.format(lpips_mean))

    # ==========================================================
    # 修复逻辑：读取并保留 train.py 写入的点数信息
    # ==========================================================
    metrics_file = os.path.join(model_path, 'metrics_{0}.txt'.format(iteration))
    point_info_lines = []

    # 如果文件已存在（train.py 生成的），提取包含 "Points" 的行
    if os.path.exists(metrics_file):
        with open(metrics_file, 'r') as f:
            lines = f.readlines()
            # 筛选出 Initial_Points 和 Added_Points
            point_info_lines = [l for l in lines if "Points" in l]

    # 重新写入：覆盖指标，但写回点数信息
    with open(metrics_file, 'w') as f:
        f.write('PSNR : {:>12.7f}\n'.format(psnr_mean))
        f.write('SSIM : {:>12.7f}\n'.format(ssim_mean))
        f.write('LPIPS : {:>12.7f}\n'.format(lpips_mean))
        # 将点数信息追加到末尾
        for line in point_info_lines:
            f.write(line)


def render_sets(dataset: ModelParams, iteration: int, pipeline: PipelineParams, skip_train: bool, skip_test: bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
            render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline,
                       background)

        if not skip_test:
            render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline,
                       background)


if __name__ == "__main__":
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    safe_state(args.quiet)
    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test)