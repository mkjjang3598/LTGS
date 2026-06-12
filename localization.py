import sys
import torch
import os
import torchvision
from argparse import ArgumentParser
from pathlib import Path
import shutil
import numpy as np

sys.path.append('./gaussian-splatting')
from scene import Scene
from gaussian_renderer import render
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False
from utils.camera_utils import cameraList_from_camInfos

from src.utils.localization_utils import hloc_localization, colmap_localization, readHlocCameras, save_hloc_results, hloc_results_from_colmap
from src.utils.visualization_utils import plot_rendering

def localize(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_localization : bool, known_intrinsics:bool, separate_sh: bool):
    with torch.no_grad():
        source_path = Path(dataset.source_path)
        scene_name = source_path.parent.stem if str(source_path).endswith("hloc") else source_path.stem
        output_dir = os.path.join("output", scene_name)
        hloc_result_dir = os.path.join(output_dir, "hloc")
        os.makedirs(hloc_result_dir, exist_ok=True)
        hloc_result_path =  os.path.join(hloc_result_dir, "hloc_results.json")

        if os.path.exists(os.path.join(dataset.model_path, "change")):
            shutil.rmtree(os.path.join(dataset.model_path, "change"))
        if not skip_localization:
            if str(source_path).endswith("hloc"):
                hloc_results = hloc_localization(dataset, known_intrinsics)
            else: # not used
                raise NotImplementedError("Use hloc for localization")
                # colmap_localization(dataset)
        else:
            dataset.single_timestep = False
            if str(source_path).endswith("hloc"):
                images_txt_path = source_path.parent / Path("images/changes.txt")
            else:
                images_txt_path = source_path / Path("images/changes.txt")
            with open(images_txt_path, 'r') as file:            
                images_path = file.read().strip().split()

        capture_path = os.path.join(output_dir, "change", "capture")
        render_path = os.path.join(output_dir, "change", "renders")
        os.makedirs(capture_path, exist_ok=True)
        os.makedirs(render_path, exist_ok=True)
        
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        if skip_localization:
            change_cameras = scene.getChangeCameras(images_path)  
        else:
            hloc_cameras = readHlocCameras(dataset, hloc_results, num_original_cameras=len(scene.train_cameras)+len(scene.test_cameras))
            change_cameras = cameraList_from_camInfos(hloc_cameras, 1.0, dataset, False, True)

        renderings, captures = [], []
        for idx, view in enumerate(change_cameras):
            rendering = render(view, gaussians, pipeline, background, use_trained_exp=dataset.train_test_exp, separate_sh=separate_sh)["render"]
            capture = view.original_image[0:3, :, :]
            if dataset.train_test_exp:
                rendering = rendering[..., rendering.shape[-1] // 2:]
                capture = view.original_image[0:3, :, :]

            renderings.append(rendering)
            captures.append(capture)
        
            torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
            torchvision.utils.save_image(capture, os.path.join(capture_path, '{0:05d}'.format(idx) + ".png"))

        renderings = torch.stack(renderings, dim=0).detach().cpu()
        captures = torch.stack(captures, dim=0).detach().cpu()

        plot_rendering(renderings, captures, output_dir, refined_renderings=None, iteration=0)

    if skip_localization:
        hloc_results = hloc_results_from_colmap(change_cameras)

    if str(source_path).endswith("hloc"):
        shutil.copy(source_path.parent / "images" / "changes.txt", os.path.join(hloc_result_dir, "changes.txt"))
    else:
        shutil.copy(source_path / "images" / "changes.txt", os.path.join(hloc_result_dir, "changes.txt"))
    
    save_hloc_results(hloc_results, hloc_result_path)


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_localization", action="store_true")
    parser.add_argument("--known_intrinsics", default=True, type=bool)
    
    args = get_combined_args(parser)
    print("HLOC Localization for " + args.model_path)
    
    localize(model.extract(args), args.iteration, pipeline.extract(args), args.skip_localization, args.known_intrinsics, SPARSE_ADAM_AVAILABLE)


