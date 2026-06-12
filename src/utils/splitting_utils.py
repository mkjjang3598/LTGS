import os
import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path

from scene import Scene
from gaussian_renderer import GaussianModel
from gaussian_renderer import flashsplat_render
from utils.camera_utils import cameraList_from_camInfos
from arguments import ModelParams, PipelineParams

import pickle
import copy
from src.utils.localization_utils import load_hloc_results, readHlocCameras
from src.utils.visualization_utils import visualize_geometry, visualize_object_points

def mean_neighborhood(input_img, N):
    pad = (N - 1) // 2
    padded_img = torch.nn.functional.pad(input_img, (pad, pad, pad, pad), mode='constant', value=0)
    patches = padded_img.unfold(1, N, 1).unfold(2, N, 1)
    mean_patches = patches.mean(dim=-1).mean(dim=-1)
    return mean_patches

def multi_instance_opt(all_counts, slackness=0.):
    all_counts = torch.nn.functional.normalize(all_counts, dim=0, p=1) # default p = 2
    all_counts_sum = all_counts.sum(dim=0)

    all_obj_labels = torch.zeros_like(all_counts)
    obj_num = all_counts.size(0)
    for obj_idx, obj_counts in enumerate(tqdm(all_counts, desc="multi-view optimize")):
        if obj_counts.sum().item() == 0:
            continue        
        # other_idx = list(range(obj_idx)) + list(range(obj_idx + 1, obj_num))
        # other_counts = all_counts[other_idx, :].sum(dim=0)
        # obj_counts = torch.stack([other_counts, obj_counts], dim=0)
        # dynamic programming
        obj_counts = torch.stack([all_counts_sum - obj_counts, obj_counts], dim=0)
        if slackness != 0:
            obj_counts = torch.nn.functional.normalize(obj_counts, dim=0)
            obj_counts[0, :] += slackness
        obj_label = obj_counts.max(dim=0)[1]
        all_obj_labels[obj_idx] = obj_label
    return all_obj_labels

## 3D instance matching
def flashsplat(dataset : ModelParams, iteration : int, pipeline : PipelineParams, slackness : float):
    with torch.no_grad():
        source_path = Path(dataset.source_path)
        scene_name = source_path.parent.stem if str(source_path).endswith("hloc") else source_path.stem
        output_dir = os.path.join("output", scene_name)
        
        render_path = os.path.join(output_dir, "change", "renders")
        filenames = sorted(os.listdir(render_path))
        
        hloc_path =  os.path.join(output_dir, "hloc", "hloc_results.json")
        hloc_results = load_hloc_results(hloc_path)

        render_mask_dir = os.path.join(output_dir, "instances", "renders")
        render_masks = []
        obj_num = 0
        
        object_lists = []
        for file in sorted(os.listdir(render_mask_dir)):
            obj_mask = np.load(os.path.join(render_mask_dir, file))
            for mask in obj_mask:
                obj_id = mask.max()
                if obj_id > 0 and obj_id not in object_lists:
                    object_lists.append(obj_id)

            render_masks.append(torch.from_numpy(obj_mask).to("cuda").to(torch.float32))   
        obj_num = len(object_lists)

        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        hloc_cameras = readHlocCameras(dataset, hloc_results, num_original_cameras=len(scene.train_cameras)+len(scene.test_cameras))
        change_cameras = cameraList_from_camInfos(hloc_cameras, 1.0, dataset, False, True)
        H, W = change_cameras[0].image_height, change_cameras[0].image_width 

        view_num = len(change_cameras)
        views_used = change_cameras
        
        all_counts = None
        mapping = {}
        num_observation = {}

        if obj_num > 0:
            for idx, view in enumerate(views_used):
                render_mask = render_masks[idx].to(torch.float32)
                if render_mask.sum() == 0:
                    continue
                for row, mask in enumerate(render_mask):
                    obj_id = int(mask.max().item())
                    if obj_id not in mapping.keys():
                        mapping[obj_id] = 1 if len(mapping.keys())==0 else max(mapping.values())+1
                        num_observation[obj_id] = 1
                    else:
                        num_observation[obj_id] += 1

                if obj_num == 1:
                    render_mask_per_view = render_mask
                    obj_num_per_view = 1
                else:
                    render_mask_per_view = torch.zeros_like(render_mask)
                    for row, mask in enumerate(render_mask):
                        render_mask_per_view[row] = mask * (row+1) / mask.max()
                    obj_num_per_view = len(render_mask)
                    
                render_pkg = flashsplat_render(view, gaussians, pipeline, background, gt_mask=render_mask_per_view.sum(0), obj_num=obj_num_per_view)
                used_count = render_pkg["used_count"]
                                
                if all_counts is None:
                    all_counts = torch.zeros((obj_num+1, used_count.shape[-1]), device=used_count.device)
                
                all_counts[0] += used_count[0]
                for row, mask in enumerate(render_mask):
                    obj_id = int(mask.max().item())
                    all_counts[mapping[obj_id]] += used_count[row+1]

            inverse_mapping = {value:key for key, value in mapping.items()}

            # Compensate for different number of observations
            all_counts[0] /= view_num
            for idx, _ in enumerate(all_counts):
                if idx == 0:
                    continue
                else:
                    obj_id = inverse_mapping[idx]
                    all_counts[idx] /= num_observation[obj_id]

            # For binary seg,
            if obj_num == 1:
                all_counts = torch.nn.functional.normalize(all_counts, dim=0)
                all_counts[0, :] += slackness 
                obj_labels = all_counts.max(dim=0)[1]
            else:
                obj_labels = multi_instance_opt(all_counts, slackness)
            
            pcds = gaussians.get_xyz
            opacity = gaussians.get_opacity
            colors = gaussians.get_features_dc[:,0,:]
            object_pcds, pcd_before = {}, {}
            for idx in range(1, obj_num+1):
                obj_id = inverse_mapping[idx]
                if obj_num ==1 :
                    pcd_idx = torch.where(obj_labels==1)[0]
                else:
                    pcd_idx = torch.where(obj_labels[idx]==1)[0]

                # ## Saving object-level gaussians
                # object_gaussians = copy.deepcopy(gaussians)
                # object_filters = torch.zeros_like(opacity).to(opacity.device)
                # object_filters[pcd_idx] = 1.0

                # opacity = object_gaussians.get_opacity
                # object_gaussians._opacity = object_gaussians.inverse_opacity_activation(opacity * object_filters)

                # print("Saving object point cloud of obj index {}".format(idx))
                # object_gaussians.save_ply(os.path.join(dataset.model_path+"_update", "point_cloud", "assets", f"flashsplat_{idx}.ply"))

                # Refine detached pcds (Remove outliers)
                with torch.no_grad():
                    object_pcd = pcds[pcd_idx].detach()
                    counts = torch.zeros(object_pcd.shape[0], dtype=torch.int, device=object_pcd.device)
                    for view_idx, view in enumerate(views_used):
                        render_mask = render_masks[view_idx].to(torch.float32)
                        if render_mask.sum() == 0:
                            continue
                        if obj_id not in render_mask:
                            continue
                        mask = render_mask[torch.where(render_mask == obj_id)[0][0]] # fix

                        pcd_homo = torch.cat((object_pcd, torch.ones((object_pcd.shape[0],1), device=object_pcd.device, dtype=torch.float32)), dim=1)
                        proj = pcd_homo @ view.full_proj_transform
                        proj = proj[..., :3] / proj[..., 3:4]
                        proj_xy = proj[...,:2].cpu().numpy()
                        x, y = ((W-1) / 2 * (proj_xy[...,0]+1)).astype(int), ((H-1) / 2 * (proj_xy[...,1]+1)).astype(int)
                        valid_mask = (x >= 0) & (x < W) & (y >= 0) & (y < H)
                        y_valid, x_valid = y[valid_mask], x[valid_mask]
                        
                        if len(x_valid) == 0:
                            continue

                        # Select only valid 3D points
                        # import matplotlib.pyplot as plt
                        # plt.scatter(x_valid, y_valid, s=5)
                        # plt.imshow(mask.cpu().numpy(), cmap='gray', alpha=0.5)
                        # plt.xlim([0, W])
                        # plt.ylim([H, 0])
                        # plt.show()
                        # breakpoint()
                        
                        matches = (mask[y_valid, x_valid] == obj_id).detach().cpu().numpy()
                        indices = np.where(valid_mask)[0] 
                        counts[indices[matches]] += 1
                        
                        # print(len(np.where(matches)[0]))
                        
                    valid_idx = torch.where(counts > num_observation[obj_id]/2)[0]
                    pcd_idx = pcd_idx[valid_idx]

                object_pcds[obj_id] = pcds[pcd_idx].detach().cpu().numpy()
                pcd_before[obj_id] = {}
                pcd_before[obj_id]["pcd"] = pcds[pcd_idx].detach().cpu().numpy()
                pcd_before[obj_id]["rgb"] = colors[pcd_idx].detach().cpu().numpy()
                pcd_before[obj_id]["opacity"] = opacity[pcd_idx].detach().cpu().numpy()
                pcd_before[obj_id]["indices"] = pcd_idx.detach().cpu().numpy()
        else:
            object_pcds, pcd_before = {}, {}
        before_pcd_path = os.path.join(output_dir, "instances", "pcd_before.pkl")
        with open(before_pcd_path, 'wb') as f:
            pickle.dump(pcd_before, f)

        # # Visualization
        # if obj_num ==1:
        #     zero_idx = torch.where(obj_labels==0)[0]
        # else:
        #     zero_idx = torch.where(obj_labels[0]==1)[0]
        # # not selected indices
        # zero_idx = torch.cat([zero_idx, torch.where(all_counts.max(0).values == 0)[0]])

        # object_pcds[0] = pcds[zero_idx].detach().cpu().numpy()
        # visualize_object_points(object_pcds)
        # visualize_geometry(camera_sets={}, points3d_xyz=pcds.detach().cpu().numpy(), points3d_rgb=(gaussians.get_features_dc[:,0,:]).detach().cpu().numpy())
        # breakpoint()

    return object_pcds, change_cameras

