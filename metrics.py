#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from pathlib import Path
import os
from PIL import Image
import torch
import torchvision.transforms.functional as tf
import sys

sys.path.append('./gaussian-splatting')
from utils.loss_utils import ssim
from lpipsPyTorch import lpips
import json
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser

def readImages(renders_dir, gt_dir):
    renders = []
    gts = []
    image_names = []
    for fname in sorted(os.listdir(renders_dir)):
        render = Image.open(renders_dir / fname)
        gt = Image.open(gt_dir / fname)
        renders.append(tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda())
        gts.append(tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda())
        image_names.append(fname)
    return renders, gts, image_names

def evaluate(output_dir):
    full_dict = {}
    per_view_dict = {}
    full_dict_polytopeonly = {}
    per_view_dict_polytopeonly = {}
    print("")

    scene_dir = os.path.basename(output_dir)
    print("Scene:", scene_dir)
    full_dict[scene_dir] = {}
    per_view_dict[scene_dir] = {}
    full_dict_polytopeonly[scene_dir] = {}
    per_view_dict_polytopeonly[scene_dir] = {}

    test_dir = Path(output_dir) / "update"

    gt_dir = test_dir / "gt_all"
    renders_dir = test_dir / "render_all"

    time_indices = sorted([int(d.name) for d in gt_dir.iterdir() if d.is_dir()])

    psnrs, ssims, lpipss = [], [], []
    image_names_all = []
    for time_idx in time_indices:
        if time_idx == 0: 
            continue

        renders, gts, image_names = readImages(renders_dir / str(time_idx), gt_dir / str(time_idx))

        for idx in tqdm(range(len(renders)), desc="Metric evaluation progress"):
            ssims.append(ssim(renders[idx], gts[idx]))
            psnrs.append(psnr(renders[idx], gts[idx]))
            lpipss.append(lpips(renders[idx], gts[idx], net_type='vgg'))

        image_names_all.extend([os.path.join(str(time_idx), name) for name in image_names])

    print("  SSIM : {:>12.7f}".format(torch.tensor(ssims).mean(), ".5"))
    print("  PSNR : {:>12.7f}".format(torch.tensor(psnrs).mean(), ".5"))
    print("  LPIPS: {:>12.7f}".format(torch.tensor(lpipss).mean(), ".5"))
    print("")

    per_view_dict[scene_dir].update({"SSIM": {name: ssim for ssim, name in zip(torch.tensor(ssims).tolist(), image_names_all)},
                                    "PSNR": {name: psnr for psnr, name in zip(torch.tensor(psnrs).tolist(), image_names_all)},
                                    "LPIPS": {name: lp for lp, name in zip(torch.tensor(lpipss).tolist(), image_names_all)}})

    full_dict[scene_dir].update({"SSIM": torch.tensor(ssims).mean().item(),
                                "PSNR": torch.tensor(psnrs).mean().item(),
                                "LPIPS": torch.tensor(lpipss).mean().item()})

    with open(output_dir + "/results.json", 'w') as fp:
        json.dump(full_dict[scene_dir], fp, indent=True)
    with open(output_dir + "/per_view.json", 'w') as fp:
        json.dump(per_view_dict[scene_dir], fp, indent=True)


if __name__ == "__main__":
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument('--output_dir', '-m', required=True, type=str, default='')
    args = parser.parse_args()
    evaluate(args.output_dir)
