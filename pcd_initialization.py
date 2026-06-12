
import sys
sys.path.append('./gaussian-splatting')
sys.path.append('./submodules/mast3r')

import torch
import os
import matplotlib.pyplot as plt

from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from pathlib import Path
from utils.graphics_utils import fov2focal

from PIL import Image
import numpy as np
import cv2

from mast3r.cloud_opt.sparse_ga import proj3d
from dust3r.utils.geometry import inv, geotrf 
            
import pickle
import shutil
from src.utils.localization_utils import extract_sort_key, get_reconstructed_scene, load_hloc_results, readHlocCameras
from src.utils.splitting_utils import flashsplat
from src.utils.visualization_utils import visualize_geometry, ColorInfo

def object_pcds_from_mast3r(dataset: ModelParams, change_cameras: list, timestep:int, temporal_indices: list):
    source_path = Path(dataset.source_path)
    scene_name = source_path.parent.stem if str(source_path).endswith("hloc") else source_path.stem
    output_dir = os.path.join("output", scene_name)
    capture_dir = os.path.join(output_dir, "change", "capture")
    instance_dir = os.path.join(output_dir, "instances")
    num_cameras = len(temporal_indices)
    filenames = [filename for idx, filename in enumerate(sorted(os.listdir(capture_dir))) if idx in temporal_indices]
    captures = [np.array(Image.open(os.path.join(capture_dir, filename))) / 255.0 for filename in filenames]

    ## Load saved hloc cameras 
    change_cameras = [change_cameras[idx] for idx in temporal_indices]
    hloc_w2c = np.tile(np.eye(4), (len(temporal_indices), 1, 1)) # w2c : 3dgs
    for idx, change_cam in enumerate(change_cameras):   
        hloc_w2c[idx, :3, :3] = np.transpose(change_cam.R) # w2c : change to original COLMAP
        hloc_w2c[idx, :3, 3] = change_cam.T 
    hloc_c2w = np.linalg.inv(hloc_w2c) # c2w : original colmap 

    # Load object masks
    object_masks = []
    obj_num = 0
    for file in filenames:
        obj_mask = np.load(os.path.join(instance_dir, "capture", file.split('.')[0]+'.npy'))
        # Also load mask for removed regions (negative values)
        rm_obj_mask = np.load(os.path.join(instance_dir, "renders", file.split('.')[0]+'.npy'))

        if len(obj_mask) == 0:
            object_masks.append(-rm_obj_mask)
        elif len(rm_obj_mask) == 0:
            object_masks.append(obj_mask)
        else:
            # Remove overlapping regions
            rm_obj_mask[:, obj_mask.sum(0) > 0] = 0
            object_masks.append(np.concatenate((obj_mask, - rm_obj_mask), axis=0))  
        obj_num = obj_mask.shape[0] if obj_mask.shape[0] > obj_num else obj_num

    # Load raw MASt3R assets
    mast3r_recon_path = os.path.join(output_dir, "mast3r", f"change_recon_{timestep}.npz")
    mast3r_recon = np.load(mast3r_recon_path)
    pts3d_original, depthmaps, confs = mast3r_recon["pts3d"], mast3r_recon["depthmaps"], mast3r_recon["confs"] 
    imgs, mast3r_c2w = mast3r_recon["imgs"], mast3r_recon["mast3r_c2w"]
    mast3r_w2c = np.linalg.inv(mast3r_c2w)

    H, W = object_masks[0].shape[-2:]
    h, w = imgs.shape[1:3]

    # Interpolate and scale depthmaps
    depthmaps = torch.from_numpy(depthmaps).to('cuda')
    depthmaps = depthmaps.reshape(depthmaps.shape[0], h, w).unsqueeze(0)
    depthmaps = torch.nn.functional.interpolate(depthmaps, [H, W], mode='bilinear', align_corners=True).squeeze(0)
    z = depthmaps.reshape(len(filenames), -1)
    
    # Backproject to make dense pcds
    pixels = torch.from_numpy(np.mgrid[:W, :H].T.reshape(-1, 2)).float().to("cuda")
    focal = fov2focal(change_cameras[0].FoVy, H)
    K = torch.eye(3).to("cuda")
    K[0, 0] = focal
    K[1, 1] = focal
    K[0, 2] = W / 2
    K[1, 2] = H / 2
    K = torch.tile(K.unsqueeze(0), (num_cameras, 1, 1))

    invK = inv(K)
    all_pts3d = []
    cam2w = torch.from_numpy(hloc_c2w).to("cuda")
    for i in range(num_cameras):
        pts3d = proj3d(invK[i], pixels, z[i])
        pts3d = geotrf(cam2w[i], pts3d)
        all_pts3d.append(pts3d)
    all_pts3d = torch.stack(all_pts3d, dim=0)
    all_pts3d = all_pts3d.view(all_pts3d.shape[0], H, W, 3).detach().cpu().numpy()

    # Visualization
    camera_sets = {}
    camera_sets["hloc"] = hloc_w2c
    camera_sets["mast3r"] = mast3r_w2c
    
    # Apply object masks
    confs = np.array([cv2.resize(c, (W, H), interpolation=cv2.INTER_LINEAR) for c in confs])

    change_obj_pcds, change_obj_confs, change_obj_rgbs = {}, {}, {}
    pcd_after = {}

    for i, object_mask in enumerate(object_masks):
        for j in range(len(object_mask)):
            obj_label = int(object_mask[j].max().item()) if object_mask[j].max() > 0 else int(object_mask[j].min().item())
            if obj_label == 0:
                continue
            pcd_idx = np.where(object_mask[j]==obj_label)
            if not obj_label in change_obj_pcds.keys():
                pcd_after[obj_label] = {}
                pcd_after[obj_label]['pcd'] = all_pts3d[i][pcd_idx] 
                pcd_after[obj_label]['conf'] = confs[i][pcd_idx]
                pcd_after[obj_label]['rgb'] = captures[i][pcd_idx]
                
                change_obj_pcds[obj_label] = all_pts3d[i][pcd_idx]
                change_obj_confs[obj_label] = confs[i][pcd_idx]
                change_obj_rgbs[obj_label] = captures[i][pcd_idx]
            else:
                pcd_after[obj_label]['pcd'] = np.concatenate((change_obj_pcds[obj_label], all_pts3d[i][pcd_idx]), axis=0)
                pcd_after[obj_label]['conf'] = np.concatenate((change_obj_confs[obj_label], confs[i][pcd_idx]), axis=0)
                pcd_after[obj_label]['rgb'] = np.concatenate((change_obj_rgbs[obj_label], captures[i][pcd_idx]), axis=0)
            
                change_obj_pcds[obj_label] = np.concatenate((change_obj_pcds[obj_label], all_pts3d[i][pcd_idx]), axis=0)
                change_obj_confs[obj_label] = np.concatenate((change_obj_confs[obj_label], confs[i][pcd_idx]), axis=0)
                change_obj_rgbs[obj_label] = np.concatenate((change_obj_rgbs[obj_label], captures[i][pcd_idx]), axis=0)

    after_pcd_path = os.path.join(output_dir, "instances", f"pcd_after_{timestep}.pkl")
    with open(after_pcd_path, 'wb') as f:
        pickle.dump(pcd_after, f)
    
    # Visualization
    # points3d_xyz = all_pts3d.reshape(-1,3)
    # points3d_rgb = np.array(captures).reshape(-1,3) 

    # visualize_geometry(camera_sets, points3d_xyz=points3d_xyz, points3d_rgb=points3d_rgb)
    # breakpoint()
    # points3d_xyz = pts3d_original.reshape(-1,3)
    # points3d_rgb = imgs.reshape(-1,3)

    # pcd, pcd_color = [], []
    # color_infos = ColorInfo()
    # for i in change_obj_pcds.keys():
    #     if i < 0:
    #         continue
    #     pcd.append(change_obj_pcds[i])
    #     pcd_color.append(change_obj_rgbs[i])
    #     # pcd_color.append(np.array(color_infos.get_color(abs(i)-1))[None]*np.ones_like(change_obj_pcds[i])/255)

    # if len(pcd) > 0:

    #     pcd = np.concatenate(pcd, 0) if len(pcd) > 1 else np.array(pcd).reshape(-1,3)
    #     pcd_color = np.concatenate(pcd_color, 0) if len(pcd_color) > 1 else np.array(pcd_color).reshape(-1,3)
 
    #     visualize_geometry(camera_sets, points3d_xyz=pcd, points3d_rgb=pcd_color)   

    return change_obj_pcds, change_obj_confs


def initialize_pcds(dataset: ModelParams, iteration: int, pipeline: PipelineParams, slackness):
    source_path = Path(dataset.source_path) # ends with hloc
    scene_name = source_path.parent.stem if str(source_path).endswith("hloc") else source_path.stem
    output_dir = os.path.join("output", scene_name)
    
    capture_path = os.path.join(output_dir, "change", "capture")
    render_path = os.path.join(output_dir, "change", "renders")
    filenames = sorted(os.listdir(capture_path))

    query_path = Path('output') / scene_name / "hloc" / "changes.txt"
    with open(query_path, 'r') as file:            
        image_filenames = file.read().strip().split()
    query_list = sorted(image_filenames, key = extract_sort_key) if image_filenames[0].startswith("IMG_") else sorted(image_filenames)
    temporal_inputs = [path.split('/')[0] for path in query_list]
    temporal_indices = {}
    for timestep, temporal_input in enumerate(sorted(set(temporal_inputs))):
        ## 2D instance matching
        temporal_indices[timestep] = [idx for idx, name in enumerate(temporal_inputs) if name.startswith(temporal_input)]

    initial_obj_pcds, change_cameras = flashsplat(dataset, iteration, pipeline, slackness)

    for timestep in range(len(set(temporal_inputs))):
        change_obj_pcds, change_obj_confs = object_pcds_from_mast3r(dataset, change_cameras, timestep, temporal_indices[timestep])
    print("Generated object pcds from MASt3R.")
    # shutil.rmtree(os.path.join(output_dir, "mast3r"))


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--slackness", default=0.0, type=float)

    args = get_combined_args(parser)
    print("Detecting changes in " + args.model_path)
    
    initialize_pcds(model.extract(args), args.iteration, pipeline.extract(args), args.slackness)