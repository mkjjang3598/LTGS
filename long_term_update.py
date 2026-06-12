import sys
import os
from tqdm import tqdm
import cv2 
from argparse import ArgumentParser, Namespace

sys.path.append('./gaussian-splatting')
from src.utils.descriptor_utils import extract_2d_descriptors, fuse_multiview_descriptors
from src.utils.visualization_utils import visualize_object_points, visualize_geometry, visualize_matches, compute_pca_image
from src.utils.localization_utils import save_hloc_results, load_hloc_results, readHlocCameras, find_test_cam_poses
from src.utils.registration_utils import find_3d_correspondences, run_teaserpp, chamfer_distance
from src.utils.update_utils import nearest_distances_ckdtree, exp_map_SO3xR3

import random
from random import randint
import uuid
from scene import Scene
from gaussian_renderer import render, dynamic_render, network_gui
from utils.image_utils import psnr
from utils.loss_utils import masked_l1_loss, l1_loss, ssim
from utils.camera_utils import cameraList_from_camInfos
from utils.general_utils import get_expon_lr_func, PILtoTorch
from scene.dataset_readers import CameraInfo, sceneLoadTypeCallbacks

from arguments import ModelParams, UpdateParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
    print("Sparse Adam Available")
except:
    SPARSE_ADAM_AVAILABLE = False
try:
    from fused_ssim import fused_ssim
    FUSED_SSIM_AVAILABLE = True
except:
    FUSED_SSIM_AVAILABLE = False
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False
import torch
import torchvision
import numpy as np
from pathlib import Path
from PIL import Image
import pickle
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity 
import glob

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def define_temporal_updates(temporal_pcds, object_tracks, object_indices, gaussians, opt, verbose=False, visualize=False):
    opacity_filters = {}
    updates = {}
    for time_idx in range(len(temporal_pcds)):
        update_indices = []
        selected_objects = []
        
        for obj_label in object_tracks.keys():
            if time_idx in object_tracks[obj_label]:
                indices = object_indices[obj_label]
                selected_objects.append(obj_label)
                update_indices.append(indices)
                # Visualization
                # points_rgb = (gaussians.get_features_dc).detach().cpu().numpy() #np.zeros((len(update), 3))
                # visualize_geometry({}, points3d_xyz=gaussians.get_xyz[indices].detach().cpu().numpy(), points3d_rgb=points_rgb[indices,0,:])
                # points_rgb = np.zeros((len(gaussians.get_xyz), 3))
                # points_rgb[indices, 0] = 1
                # visualize_geometry({}, points3d_xyz=gaussians.get_xyz.detach().cpu().numpy(), points3d_rgb=points_rgb)

        opacity_filter = torch.ones(gaussians.get_xyz.shape[0], 1).cuda() 
        for obj_label in object_indices.keys():
            if obj_label not in selected_objects:
                opacity_filter[object_indices[obj_label]] = 0
        opacity_filters[time_idx] = opacity_filter

        update_indices = np.concatenate(update_indices, axis=0) if len(update_indices) > 1 else np.array(update_indices)
        update = torch.zeros(gaussians.get_xyz.shape[0], dtype=torch.bool, device="cuda")
        update[update_indices] = True

        if opt.update_dist_thres > 0:
            with torch.no_grad():
                dist = nearest_distances_ckdtree(gaussians.get_xyz[update], gaussians.get_xyz)
                additional_update = (dist < opt.update_dist_thres) & ~update
                update = update | additional_update

        updates[time_idx] = update

        if verbose:
            print("Objects per timestep")
            print(time_idx, selected_objects)
        # ## Visualize regions to be updated

        if visualize:
            points_rgb = np.zeros((len(gaussians.get_xyz), 3)) # (gaussians.get_features_dc[:,0,:]).detach().cpu().numpy() #
            points_rgb[update_indices.astype(int), 0] = 1
            if opt.update_dist_thres > 0:
                points_rgb[torch.where(additional_update)[0].detach().cpu().numpy(), 1] = 1
            visualize_geometry({}, points3d_xyz=gaussians.get_xyz.detach().cpu().numpy(), points3d_rgb=points_rgb)

    return opacity_filters, updates
    
def render_gaussians(dataset, pipe, gaussians, cameras, background, object_indices, time_idx, target_timestep, opacity_filters, object_tracks, est_mats, render_path=None, gt_path=None, est_mats_errors=None):
    renderings, gts = [], []
    for idx, view in enumerate(cameras):
        if object_indices is None or time_idx is None or target_timestep is None or target_timestep is None or opacity_filters is None or object_tracks is None or est_mats is None:
            rendering = render(view, gaussians, pipe, background, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)["render"]
        else:
            rendering = dynamic_render(view, gaussians, pipe, background, object_indices, time_idx, target_timestep, opacity_filters, object_tracks, est_mats, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE, est_mats_errors=est_mats_errors)["render"]
        gt = view.original_image[0:3, :, :]
        # if dataset.train_test_exp:
        #     rendering = rendering[..., rendering.shape[-1] // 2:]
        #     gt = gt[..., gt.shape[-1] // 2:]

        renderings.append(rendering)
        gts.append(gt)

        if render_path is not None and gt_path is not None:
            torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
            torchvision.utils.save_image(gt, os.path.join(gt_path, '{0:05d}'.format(idx) + ".png"))

    return renderings, gts

def localize_render_gaussians(dataset, pipe, gaussians, scene, background, object_indices, time_idx, target_timestep, opacity_filters, object_tracks, est_mats, source_path, output_dir, render_all_path, gt_all_path, est_mats_errors=None):
    images_txt_path = source_path.parent / Path("images/changes.txt")
    test_hloc_path = os.path.join(output_dir, "update", "test_hloc_results.json")
    with open(images_txt_path, 'r') as file:            
        images_path = file.read().strip().split()
    
    change_idx = int(images_path[0].split('/')[0].split('_')[-1])+time_idx-1
    change_dir = f"IMG_{change_idx:04d}"
    scene_info = sceneLoadTypeCallbacks["Colmap"](str(source_path.parent), dataset.images, dataset.depths, True, dataset.train_test_exp, change_dir=None)
    
    if not os.path.exists(test_hloc_path):
        test_hloc_results = find_test_cam_poses(dataset, scene_info)
        save_hloc_results(test_hloc_results, test_hloc_path)
    else:
        test_hloc_results = load_hloc_results(test_hloc_path)
    
    # only select test_hloc_results starting from change_dir
    test_hloc_results = [image for image in test_hloc_results if image['name'].startswith(change_dir)]
    test_hloc_cameras = readHlocCameras(dataset, test_hloc_results, num_original_cameras=len(scene.train_cameras)+len(scene.test_cameras))
    test_cameras = cameraList_from_camInfos(test_hloc_cameras, 1.0, dataset, scene_info.is_nerf_synthetic, True)
    print("Rendering Test Sets")

    dataset.train_test_exp = False
    render_gaussians(dataset, pipe, gaussians, test_cameras, background, object_indices, time_idx, target_timestep, opacity_filters, object_tracks, est_mats, render_all_path, gt_all_path, est_mats_errors)


def fuse_descriptors(imagefiles, object_masks, change_cameras, src_pcds, tgt_pcds, src_indices, tgt_indices, output_dir):
    descriptor_path = os.path.join(output_dir, "descriptors")
    os.makedirs(descriptor_path, exist_ok=True)

    before_obj_indices, after_obj_indices = [], []
    for idx, object_mask in enumerate(object_masks):
        obj_indices = [int(obj.max()) for _, obj in enumerate(object_mask)]
        for obj_idx in obj_indices:
            if idx in src_indices and obj_idx not in before_obj_indices:
                before_obj_indices.append(obj_idx) 
            elif idx in tgt_indices and obj_idx not in after_obj_indices:
                after_obj_indices.append(obj_idx)
    # for idx in src_pcds.keys():
    #     if idx > 0:
    #         before_obj_indices.append(idx)
    # for idx in tgt_pcds.keys():
    #     if idx > 0:
    #         after_obj_indices.append(idx)

    # define remove, inserted, matching objects
    removed_obj_label = [idx for idx in before_obj_indices if idx not in after_obj_indices]
    inserted_obj_label = [idx for idx in after_obj_indices if idx not in before_obj_indices]
    matching_obj_label = [idx for idx in before_obj_indices if idx in after_obj_indices]
    all_obj_label = removed_obj_label + inserted_obj_label + matching_obj_label

    descriptors = extract_2d_descriptors(all_obj_label, object_masks, imagefiles, descriptor_path, input_mode="full")
    src_pcd = {idx: src_pcds[idx]['pcd'] for idx in src_pcds.keys()}
    tgt_pcd = {idx: tgt_pcds[idx]['pcd'] for idx in tgt_pcds.keys()}

    fused_outputs = fuse_multiview_descriptors(src_pcd, tgt_pcd, change_cameras, all_obj_label, descriptors, object_masks, src_indices, descriptor_path)

    return fused_outputs


def update_3dgs(dataset: ModelParams, opt: UpdateParams, pipe: PipelineParams, iteration: int, use_previous_viewpoints:bool, compensate_exposure: bool, skip_localization: bool):
    with torch.no_grad():
        source_path = Path(dataset.source_path)
        gaussians = GaussianModel(dataset.sh_degree)
        if skip_localization:
            dataset.single_timestep = False
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        scene_name = source_path.parent.stem if str(source_path).endswith("hloc") else source_path.stem
        target_timestep = 0

        output_dir = os.path.join("output", scene_name)
        capture_path = os.path.join(output_dir, "change", "capture")
        render_path = os.path.join(output_dir, "change", "renders")

        capture_mask_dir = os.path.join(output_dir, "instances", "capture")
        render_mask_dir = os.path.join(output_dir, "instances", "renders")
        
        # Load before and after pcds
        before_pcd_path = os.path.join(output_dir, "instances", "pcd_before.pkl")
        after_pcd_paths = sorted([os.path.join(output_dir, "instances", path) for path in os.listdir(os.path.join(output_dir, "instances")) if path.startswith("pcd_after")])

        descriptor_path = os.path.join(output_dir, "descriptors")
        hloc_path =  os.path.join(output_dir, "hloc", "hloc_results.json")
        hloc_results = load_hloc_results(hloc_path)
        hloc_cameras = readHlocCameras(dataset, hloc_results, num_original_cameras=len(scene.train_cameras)+len(scene.test_cameras))
        change_cameras = cameraList_from_camInfos(hloc_cameras, 1.0, dataset, False, True)
        
        temporal_inputs = [hloc['name'].split('/')[0] for hloc in hloc_results]
        temporal_indices = {}
        for timestep, temporal_input in enumerate(sorted(set(temporal_inputs))):
            temporal_indices[timestep+1] = [idx for idx, name in enumerate(temporal_inputs) if name.startswith(temporal_input)]
        
        filenames = sorted(os.listdir(capture_path))
        imagefiles, object_masks, image_batch = [], [], []
        for filename in filenames:                            
            imagefiles.append(os.path.join(render_path, filename))
            imagefiles.append(os.path.join(capture_path, filename))

            image_batch.append(np.array(Image.open(os.path.join(render_path, filename)).convert("RGB")))
            image_batch.append(np.array(Image.open(os.path.join(capture_path, filename)).convert("RGB")))
            
            object_masks.append(torch.from_numpy(np.load(os.path.join(render_mask_dir, filename.split('.')[0]+'.npy'))).to(torch.float32))   
            object_masks.append(torch.from_numpy(np.load(os.path.join(capture_mask_dir, filename.split('.')[0]+'.npy'))).to(torch.float32))   

        num_objects = 0
        for obj_mask in object_masks:
            if len(obj_mask) == 0:
                continue
            if obj_mask.max() > num_objects:
                num_objects = int(obj_mask.max().item())
        
        temporal_pcds = []
        with open(before_pcd_path, 'rb') as f:
            pcd_before = pickle.load(f)
            temporal_pcds.append(pcd_before)

        for after_pcd_path in after_pcd_paths:
            with open(after_pcd_path, 'rb') as f:
                pcd_after_loaded = pickle.load(f)
                temporal_pcds.append(pcd_after_loaded)

    # We estimate est_mats and obj_labels with the target_timestep as the canoncial 
    fused_descriptors = {}
    for time_idx, temporal_pcd in enumerate(temporal_pcds):
        src_pcds, tgt_pcds = temporal_pcds[0], temporal_pcds[time_idx]
        if time_idx == 0:
            continue
        input_change_cameras = [cam for idx, cam in enumerate(change_cameras) if idx in temporal_indices[time_idx]]
        input_imagefiles = [img for idx, img in enumerate(imagefiles) if idx // 2 in temporal_indices[time_idx]]
        input_object_masks = [mask for idx, mask in enumerate(object_masks) if idx // 2 in temporal_indices[time_idx]]

        src_indices = [idx for idx in range(len(input_imagefiles)) if idx % 2 == 0]
        tgt_indices = [idx for idx in range(len(input_imagefiles)) if idx % 2 != 0]

        if len(temporal_pcd) == 0:
            descriptors = {}
        else:
            descriptors = fuse_descriptors(input_imagefiles, input_object_masks, input_change_cameras, src_pcds, tgt_pcds, src_indices, tgt_indices, output_dir)
        fused_descriptors[time_idx] = descriptors

    est_mats, removed_obj_labels, inserted_obj_labels, matching_obj_labels = {}, {}, {}, {}
    object_tracks = {} # key: object_label, value: time_idx
    
    for temporal_pcd in temporal_pcds:
        for key in temporal_pcd.keys():
            if key not in object_tracks.keys():
                object_tracks[key] = []

    src_pcds = temporal_pcds[target_timestep].copy()
    # Do not modify the original pcds
    for time_idx, temporal_pcd in enumerate(temporal_pcds):
        tgt_pcds = temporal_pcds[time_idx]
        src_obj_indices = [idx for idx in src_pcds.keys() if idx > 0]
        tgt_obj_indices = [idx for idx in tgt_pcds.keys() if idx > 0]

        removed_obj_label = [idx for idx in src_obj_indices if idx not in tgt_obj_indices]
        inserted_obj_label = [idx for idx in tgt_obj_indices if idx not in src_obj_indices]
        matching_obj_label = [idx for idx in src_obj_indices if idx in tgt_obj_indices]

        # skip if time_idx is target_timestep
        if time_idx == target_timestep: # target_timestep = 0
            for obj_label in src_pcds.keys():
                object_tracks[obj_label].append(time_idx)
            fused_descriptors[time_idx] = {}
            continue
        
        temporal_descriptors = {}  
        for idx in matching_obj_label:
            temporal_descriptors[idx] = {}
            if idx not in temporal_pcds[target_timestep].keys():     
                # prev_time_idx = object_tracks[idx][-1]
                prev_time_idx = object_tracks[idx][0]
                temporal_descriptors[idx]['desc_3d_1'] = fused_descriptors[prev_time_idx][idx]['desc_3d_2']
                temporal_descriptors[idx]['valid_idx_1'] = fused_descriptors[prev_time_idx][idx]['valid_idx_2']
                temporal_descriptors[idx]['proj1_xy'] = fused_descriptors[prev_time_idx][idx]['proj2_xy']
            else:
                # Added for detached moving objects 
                no_descriptor = fused_descriptors[time_idx][idx]['desc_3d_1'] is None
                if no_descriptor:
                    for i in range(len(fused_descriptors)):
                        if idx in fused_descriptors[i].keys() and fused_descriptors[i][idx]['desc_3d_1'] is not None:
                            src_time_idx = i
                            no_descriptor = False
                            break
                else:
                    src_time_idx = time_idx
                temporal_descriptors[idx]['desc_3d_1'] = fused_descriptors[src_time_idx][idx]['desc_3d_1']
                temporal_descriptors[idx]['valid_idx_1'] = fused_descriptors[src_time_idx][idx]['valid_idx_1']
                temporal_descriptors[idx]['proj1_xy'] = fused_descriptors[src_time_idx][idx]['proj1_xy']

            temporal_descriptors[idx]['desc_3d_2'] = fused_descriptors[time_idx][idx]['desc_3d_2']
            temporal_descriptors[idx]['valid_idx_2'] = fused_descriptors[time_idx][idx]['valid_idx_2']
            temporal_descriptors[idx]['proj2_xy'] = fused_descriptors[time_idx][idx]['proj2_xy']

        for idx in src_pcds.keys():
            if int(idx) in matching_obj_label and idx not in temporal_descriptors.keys():
                prev_time_idx = object_tracks[idx][-1]
                temporal_descriptors[idx] = {}
                temporal_descriptors[idx]['desc_3d_1'] = fused_descriptors[prev_time_idx][int(idx)]['desc_3d_2']
                temporal_descriptors[idx]['valid_idx_1'] = fused_descriptors[prev_time_idx][int(idx)]['valid_idx_2']
                temporal_descriptors[idx]['proj1_xy'] = fused_descriptors[prev_time_idx][int(idx)]['proj2_xy']

                temporal_descriptors[idx]['desc_3d_2'] = fused_descriptors[time_idx][int(idx)]['desc_3d_2']
                temporal_descriptors[idx]['valid_idx_2'] = fused_descriptors[time_idx][int(idx)]['valid_idx_2']
                temporal_descriptors[idx]['proj2_xy'] = fused_descriptors[time_idx][int(idx)]['proj2_xy']

                tgt_pcds[idx] = tgt_pcds[int(idx)]
                matching_obj_label.append(idx)
                removed_obj_label.remove(idx)

        src_pcd = {idx: src_pcds[idx]['pcd'] for idx in src_pcds.keys()}
        tgt_pcd = {idx: tgt_pcds[idx]['pcd'] for idx in tgt_pcds.keys()}

        obj_kpts_1, obj_kpts_2 = find_3d_correspondences(src_pcd, tgt_pcd, temporal_descriptors, matching_obj_label, descriptor_path, opt.num_sample)
        est_mat = run_teaserpp(obj_kpts_1, obj_kpts_2, matching_obj_label)

        # If fail to register, change labels
        for obj_label in matching_obj_label.copy():
            transformed_obj_kpts = obj_kpts_1[obj_label] @ est_mat[obj_label][:3, :3].T + est_mat[obj_label][:3, 3:4].T 
            dist = chamfer_distance(transformed_obj_kpts[None], obj_kpts_2[obj_label][None])
            print(f"Chamfer Distance for {obj_label} : {dist}")
            if dist > opt.overlap_thres:
                # idx = np.where(np.array(matching_obj_label) == obj_label)[0].item() 
                # matching_obj_label.pop(idx)
                matching_obj_label.remove(obj_label)
                removed_obj_label.append(obj_label)
                inserted_obj_label.append(obj_label)
        
        # Remove overlaps
        for obj_label in src_pcds.keys():
            if isinstance(obj_label, int) or obj_label.is_integer():
                continue 
            if obj_label in inserted_obj_label and obj_label in removed_obj_label:
                inserted_obj_label.remove(obj_label)
                removed_obj_label.remove(obj_label) 
                tgt_pcds.pop(obj_label)
                
            if obj_label in matching_obj_label:
                inserted_obj_label.remove(int(obj_label))
                removed_obj_label.remove(int(obj_label)) 

        # Allocate new labels for unmatched instances
        for obj_label in inserted_obj_label.copy():
            if obj_label in removed_obj_label:
                new_obj_label = obj_label + 0.1
                while new_obj_label in src_pcds.keys():
                    new_obj_label += 0.1
                new_obj_label = round(new_obj_label, 1)

                inserted_obj_label.remove(obj_label)
                inserted_obj_label.append(new_obj_label)
                src_pcds[new_obj_label] = tgt_pcds[obj_label]

                object_tracks[new_obj_label] = []
                object_tracks[new_obj_label].append(time_idx)
            elif time_idx > 0:
                src_pcds[obj_label] = tgt_pcds[obj_label]
                object_tracks[obj_label].append(time_idx)

        if time_idx > 0:
            for obj_label in matching_obj_label:
                object_tracks[obj_label].append(time_idx)

            for obj_label in temporal_pcds[0]:
                if obj_label not in tgt_pcds.keys() and -obj_label not in tgt_pcds.keys():
                    object_tracks[obj_label].append(time_idx)
                    est_mat[obj_label] = np.eye(4)
                    src_pcds[obj_label] = temporal_pcds[0][obj_label]

            for obj_label in tgt_pcds.keys():
                if obj_label < 0:
                    object_tracks[obj_label].append(time_idx)

        # Remove matching_obj pcds, removed pcds for time_idx = 0
        # if time_idx == 0:
        #     for obj_label in matching_obj_label+inserted_obj_label:
        #         object_tracks[obj_label].append(time_idx)
        #         if obj_label in inserted_obj_label and -obj_label not in src_pcds.keys():
        #             est_mat[obj_label] = np.eye(4)
        #             src_pcds[obj_label] = tgt_pcds[obj_label]

        est_mats[time_idx] = est_mat
        removed_obj_labels[time_idx] = removed_obj_label
        matching_obj_labels[time_idx] = matching_obj_label
        inserted_obj_labels[time_idx] = inserted_obj_label 

        print("time_idx : ", time_idx)
        print("canonical_pcd keys : ", src_pcds.keys())
        print("object tracks : ", object_tracks)
        print("est_mat keys : ", est_mat.keys())

    # For background region
    for obj_label in object_tracks.keys():
        if obj_label < 0:
            xyz = [temporal_pcd[obj_label]['pcd'] for temporal_pcd in temporal_pcds if obj_label in temporal_pcd.keys()] 
            rgb = [temporal_pcd[obj_label]['rgb'] for temporal_pcd in temporal_pcds if obj_label in temporal_pcd.keys()] 
            conf = [temporal_pcd[obj_label]['conf'] for temporal_pcd in temporal_pcds if obj_label in temporal_pcd.keys()] 
            num = [len(temporal_pcd[obj_label]['pcd']) for temporal_pcd in temporal_pcds if obj_label in temporal_pcd.keys()]

            xyz = np.concatenate(xyz, axis=0)
            rgb = np.concatenate(rgb, axis=0)
            conf = np.concatenate(conf, axis=0)

            num = np.array(num).max()
            sample_indices = np.random.permutation(len(xyz))[:num]
            xyz, rgb, conf = xyz[sample_indices], rgb[sample_indices], conf[sample_indices]

            if obj_label in src_pcds.keys():
                src_pcds.pop(obj_label)
            
            src_pcds[obj_label] = {
                'pcd': xyz,
                'rgb': rgb,
                'conf': conf
            }

    print("Final canonical_pcd keys : ", src_pcds.keys())
        
    refine_optimize(dataset, opt, pipe, gaussians, scene, hloc_cameras, src_pcds, temporal_pcds, object_tracks, est_mats, target_timestep, use_previous_viewpoints, compensate_exposure, skip_localization)

def refine_optimize(dataset, opt, pipe, gaussians, scene, hloc_cameras, canonical_pcds, temporal_pcds, object_tracks, est_mats, target_timestep, use_previous_viewpoints, compensate_exposure, skip_localization):
    source_path = Path(dataset.source_path)
    scene_name = source_path.parent.stem if str(source_path).endswith("hloc") else source_path.stem
    scene.model_path = scene.model_path + "_update" 

    ### Important changes in configurations ###
    dataset.model_path = scene.model_path
    dataset.train_test_exp = True if compensate_exposure else False
    dataset.depths = ""

    opt.iterations = opt.refine_iterations
    opt.invalid_initialization = [int(label) for label in opt.invalid_initialization if label.isdigit()]

    # Set maximum SH degree to 1
    gaussians.optimizer_type = opt.optimizer_type
    gaussians.training_setup(opt, hloc_cameras) 
    gaussians.active_sh_degree = 3
    
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    output_dir = os.path.join("output", scene_name)
    render_path = os.path.join(output_dir, "update", "renders")
    capture_path = os.path.join(output_dir, "update", "gt")
    render_before_path = os.path.join(output_dir, "update", "before_all")
    render_initialized_path = os.path.join(output_dir, "update", "initialized_all")
    render_all_path = os.path.join(output_dir, "update", "render_all")
    capture_all_path = os.path.join(output_dir, "update", "gt_all")
    vis_diff_path = os.path.join(output_dir, "update", "vis_diff")

    os.makedirs(render_path, exist_ok=True)
    os.makedirs(capture_path, exist_ok=True)
    os.makedirs(render_before_path, exist_ok=True)
    os.makedirs(render_initialized_path, exist_ok=True)
    os.makedirs(render_all_path, exist_ok=True)
    os.makedirs(capture_all_path, exist_ok=True)
    os.makedirs(vis_diff_path, exist_ok=True)

    # # Rendering before gaussians
    # with torch.no_grad():
    #     if not skip_localization:
    #         localize_render_gaussians(dataset, pipe, gaussians, scene, background, None, None, None, None, None, None, source_path, output_dir, render_all_path=render_before_path, gt_all_path=gt_all_path)
    #     else:
    #         dataset.train_test_exp = False
    #         render_gaussians(dataset, pipe, gaussians, scene.test_cameras, background, None, None, None, None, None, None, render_all_path, gt_all_path)

    hloc_cameras_all = hloc_cameras.copy()
    if not use_previous_viewpoints:
        for idx, camera in enumerate(hloc_cameras):
            filename = f"{idx:05}.png"
            image_path = os.path.join(output_dir, "change", "renders", filename)
            image_name = scene.train_cameras[0].image_name.split('/')[0] + "/" + filename
            cam_info = CameraInfo(uid=hloc_cameras[-1].uid+1, R=camera.R, T=camera.T, FovY=camera.FovY, FovX=camera.FovX, depth_params=camera.depth_params,
                                image_path=image_path, image_name=image_name, depth_path=camera.depth_path,
                                width=camera.width, height=camera.height, is_test=True)
            hloc_cameras_all.append(cam_info)
    else:
        prev_cameras_all = []
        previous_rendered_path = sorted(glob.glob(os.path.join(dataset.model_path.replace("_update", ""), "train", "ours_30000", "renders", "*.png")))

        for idx, camera in enumerate(sorted(scene.train_cameras, key=lambda cam: cam.image_name)):
            if not camera.image_name.startswith("IMG_0000"):
                continue
            image_path = previous_rendered_path[idx]
            rendered_image = Image.open(image_path)
            resolution = (camera.original_image.shape[-1], camera.original_image.shape[-2])
            resized_image_rgb = PILtoTorch(rendered_image, resolution)
            gt_image = resized_image_rgb[:3, ...]
            camera.original_image = gt_image.clamp(0.0, 1.0).to(camera.data_device)
            prev_cameras_all.append(camera)
            
    change_cameras = cameraList_from_camInfos(hloc_cameras_all, 1.0, dataset, False, True, height=hloc_cameras[0].height, width=hloc_cameras[0].width)

    images_all = sorted([cam.image_name for cam in hloc_cameras_all])
    temporal_inputs = sorted(set([image.split('/')[0] for image in images_all]))
    if "IMG_0000" not in temporal_inputs:
        temporal_inputs.insert(0, "IMG_0000")

    ## Initialize MASt3R pcds (Refer to InstantSplat's method)
    object_indices = {}
    for obj_label, obj_pcd in canonical_pcds.items():        
        xyz, rgb = obj_pcd['pcd'], obj_pcd['rgb'] 
        if obj_label in temporal_pcds[0].keys() and len(object_tracks[obj_label]) > 0:
            print("Skipping for object label : ", obj_label)
            indices = obj_pcd['indices']
        else:
            print("Adding Gaussians for object label : ", obj_label)
            # conf thresholding
            conf = obj_pcd['conf']
            xyz, rgb = xyz[conf > opt.conf_thres], rgb[conf > opt.conf_thres]
            # downsampling
            xyz, rgb = xyz[::opt.downsample_factor], rgb[::opt.downsample_factor]
            # print("Number of points added : ", len(xyz))
            
            indices = gaussians.get_xyz.shape[0] + np.linspace(0, len(xyz)-1, len(xyz), dtype=np.int64)
            gaussians.initialize_from_mast3r(xyz, rgb, spatial_lr_scale=scene.cameras_extent)
        object_indices[obj_label] = indices

    opacity_filters, updates = define_temporal_updates(temporal_pcds, object_tracks, object_indices, gaussians, opt, verbose=True, visualize=False)

    ## Refine 3DGS with extra training (Training_code)   
    first_iter = 0 
    last_iter = first_iter + opt.refine_iterations
    tb_writer = prepare_output_and_logger(dataset)
    
    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    if len(opt.sparse_adam_temporal) == 0:
        sparse_adam_temporal = [idx for idx in range(len(temporal_inputs))]
    elif opt.sparse_adam_temporal == ['-', '1']:
        sparse_adam_temporal = []
    else:
        sparse_adam_temporal = [int(label) for label in opt.sparse_adam_temporal if label.isdigit()]

    use_sparse_adam = opt.optimizer_type == "sparse_adam" and SPARSE_ADAM_AVAILABLE 
    depth_l1_weight = get_expon_lr_func(opt.depth_l1_weight_init, opt.depth_l1_weight_final, max_steps=opt.refine_iterations)

    if all(v == {} for v in est_mats.values()):
        opt.optimize_obj_poses = False
        
    if opt.optimize_obj_poses:
        # Add additional 6DOF tuning parameters to each est_mats
        print("Jointly optimizing object poses")
        pose_params = []
        est_mats_errors = {}
        for time_idx, est_mat in est_mats.items():
            est_mat_err = {}
            for obj_label in est_mat.keys():
                pose_param = torch.nn.Parameter(torch.zeros(6, device="cuda"), requires_grad=True)
                est_mat_err[obj_label] = pose_param
                pose_params.append(pose_param)
            est_mats_errors[time_idx] = est_mat_err

        pose_optimizer = torch.optim.Adam(pose_params, lr=opt.obj_pose_lr)
    else:
        est_mats_errors = None

    viewpoint_stack = change_cameras.copy()
    if use_previous_viewpoints:
        random.shuffle(prev_cameras_all)
        prev_cameras_pos = 0
        num_to_include = len(hloc_cameras_all)

        # change image_name to rendered_images
        next_end = min(prev_cameras_pos + num_to_include, len(prev_cameras_all))
        viewpoint_stack += prev_cameras_all[prev_cameras_pos:next_end]
        prev_cameras_pos = next_end

    viewpoint_indices = list(range(len(viewpoint_stack)))
    ema_loss_for_log = 0.0
    ema_Ll1depth_for_log = 0.0

    progress_bar = tqdm(range(first_iter, last_iter), desc="Updating progress")
    first_iter += 1
    
    for iteration in range(first_iter, last_iter+1):
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = dynamic_render(custom_cam, gaussians, pipe, background, object_indices, time_idx, target_timestep, opacity_filters, object_tracks, est_mats, scaling_modifier=scaling_modifer, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE, est_mats_errors=est_mats_errors)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(last_iter)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 30000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = change_cameras.copy()
            if use_previous_viewpoints:
                if prev_cameras_pos >= len(prev_cameras_all):
                    prev_cameras_pos = 0
                next_end = min(prev_cameras_pos + num_to_include, len(prev_cameras_all))
                viewpoint_stack += prev_cameras_all[prev_cameras_pos:next_end]
                prev_cameras_pos = next_end
            viewpoint_indices = list(range(len(viewpoint_stack)))
        rand_idx = randint(0, len(viewpoint_indices) - 1)
        viewpoint_cam = viewpoint_stack.pop(rand_idx)
        vind = viewpoint_indices.pop(rand_idx)

        # Transform Gaussians
        time_idx = temporal_inputs.index(viewpoint_cam.image_name.split('/')[0])
        
        if time_idx == 0 and opt.initial_time_loss_weight == 0 and len(opt.invalid_initialization)==len(temporal_inputs)-1 and not use_previous_viewpoints:
            if iteration == last_iter:
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)
            continue
        # Render initialized gaussians
        # with torch.no_grad():
        #     localize_render_gaussians(dataset, pipe, gaussians, scene, background, object_indices, 4, target_timestep, opacity_filters, object_tracks, est_mats, source_path, output_dir, render_initialized_path, os.path.join(capture_all_path))
        # breakpoint()
        
        # Redefine update
        bg = torch.rand((3), device="cuda") if opt.random_background else background
        render_pkg = dynamic_render(viewpoint_cam, gaussians, pipe, bg, object_indices, time_idx, target_timestep, opacity_filters, object_tracks, est_mats, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE, est_mats_errors=est_mats_errors)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        if viewpoint_cam.alpha_mask is not None:
            alpha_mask = viewpoint_cam.alpha_mask.cuda()
            image *= alpha_mask

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)

        if FUSED_SSIM_AVAILABLE:
            ssim_value = fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
        else:
            ssim_value = ssim(image, gt_image)

        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)

        # Depth regularization
        Ll1depth_pure = 0.0
        if depth_l1_weight(iteration) > 0 and viewpoint_cam.depth_reliable:
            invDepth = render_pkg["depth"]
            mono_invdepth = viewpoint_cam.invdepthmap.cuda()
            depth_mask = viewpoint_cam.depth_mask.cuda()

            Ll1depth_pure = torch.abs((invDepth  - mono_invdepth) * depth_mask).mean()
            Ll1depth = depth_l1_weight(iteration) * Ll1depth_pure 
            loss += Ll1depth
            Ll1depth = Ll1depth.item()
        else:
            Ll1depth = 0

        if time_idx == 0 and not use_previous_viewpoints:
            image_idx = int(len(images_all)/2 + int(viewpoint_cam.image_name.split('/')[-1].split('.')[0]))
            if int(images_all[image_idx].split('/')[0].split('_')[1]) in opt.invalid_initialization:
                loss *= opt.initial_time_loss_weight
    
        loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_Ll1depth_for_log = 0.4 * Ll1depth + 0.6 * ema_Ll1depth_for_log

            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}", "Depth Loss": f"{ema_Ll1depth_for_log:.{7}f}"})
                progress_bar.update(10)

                # if time_idx > 0:
                #     torchvision.utils.save_image(image, '{0:05d}'.format(iteration) + ".png")

                # image_np = image.detach().cpu().numpy().transpose(1,2,0)
                # gt_image_np = gt_image.detach().cpu().numpy().transpose(1,2,0)
                # # Compute SSIM map
                # ssim_score, ssim_map = structural_similarity(image_np, gt_image_np, data_range=1, channel_axis=2, full=True)
                # # Compute absolute difference map
                # ssim_map = ssim_map.mean(-1)
                # diff_map = np.abs(image_np - gt_image_np)
                # diff_map = diff_map.mean(-1)
                # # ---------------------------
                # # Plot SSIM map and difference
                # # ---------------------------
                # fig, axes = plt.subplots(2, 2, figsize=(8, 6))

                # # GT
                # axes[0,0].imshow(gt_image_np)
                # axes[0,0].set_title('Capture Image')
                # axes[0,0].axis('off')

                # # Capture
                # axes[0,1].imshow(image_np)
                # axes[0,1].set_title('Render Image')
                # axes[0,1].axis('off')

                # # Plot SSIM map
                # axes[1,0].imshow(1-ssim_map)
                # axes[1,0].set_title(f'SSIM Map\nMean SSIM: {ssim_score:.4f}')
                # axes[1,0].axis('off')

                # # Plot difference image (clip to [0,1] for display if normalized)
                # axes[1,1].imshow(diff_map)
                # axes[1,1].set_title('Image Absolute Difference')
                # axes[1,1].axis('off')

                # plt.tight_layout()

                # plt.savefig(f"{vis_diff_path}/diff_{iteration}.png")  # Optional save
                # plt.show()

            if iteration == last_iter:
                progress_bar.close()

            # Log and save
            if iteration == last_iter:
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)
                ## Save temporal gaussians
                for time_idx in range(len(temporal_inputs)):
                    scene.save_temporal(iteration, object_indices, object_tracks, est_mats, opacity_filters, time_idx, separate_sh=SPARSE_ADAM_AVAILABLE)

            # Densification -> We don't use densification for updates
            # if iteration < opt.densify_until_iter:
            #     # Keep track of max radii in image-space for pruning
            #     gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
            #     gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

            #     if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
            #         size_threshold = 20 if iteration > opt.opacity_reset_interval else None
            #         gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold, radii)
                
            #     if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
            #         gaussians.reset_opacity()

            if iteration == 0 and opt.opacity_reset_interval > 0:
                gaussians.reset_opacity()

            if opt.prune_artifacts and iteration < opt.prune_until_iter:
                # Prune artifacts from detached region 
                if time_idx > 0:
                    gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.prune_from_iter and iteration % opt.prune_interval == 0: 
                    pruned_masks = gaussians.prune_artifacts(opt.prune_grad_thres, opt.prune_dist_thres, object_indices)
                    for obj_idx, indices in object_indices.copy().items():
                        # Step 1: Compute the mapping from original index → new index after pruning
                        mapping = np.full_like(pruned_masks, fill_value=-1, dtype=np.int32)
                        unpruned_indices = np.flatnonzero(~pruned_masks)
                        mapping[unpruned_indices] = np.arange(len(unpruned_indices))

                        # Step 2: Convert indices to new indices, but only for those that are not pruned
                        indices = np.array(indices)  
                        keep_mask = ~pruned_masks[indices]                          
                        mapped_indices = mapping[indices[keep_mask]]
                        object_indices[obj_idx] = mapped_indices
                    
                    radii = radii[~pruned_masks]
                    opacity_filters, updates = define_temporal_updates(temporal_pcds, object_tracks, object_indices, gaussians, opt, verbose=False, visualize=False)

            # Optimizer step
            if iteration < last_iter:
                gaussians.exposure_optimizer.step()
                gaussians.exposure_optimizer.zero_grad(set_to_none = True)

                if opt.optimize_obj_poses:
                    pose_optimizer.step()
                    pose_optimizer.zero_grad(set_to_none = True)

                if use_sparse_adam:
                    visible = radii > 0
                    if time_idx in sparse_adam_temporal:
                        visible = visible & updates[time_idx]
                    elif sparse_adam_temporal == []:
                        visible = visible & opacity_filters[time_idx][:,0].to(torch.bool)

                    gaussians.optimizer.step(visible, radii.shape[0])
                    gaussians.optimizer.zero_grad(set_to_none = True)
                else:
                    gaussians.optimizer.step()
                    gaussians.optimizer.zero_grad(set_to_none = True)
    
    with torch.no_grad():    
        ## Visualize optimized 3DGS
        # visualize_geometry({}, points3d_xyz=gaussians.get_xyz.detach().cpu().numpy(), points3d_rgb=points_rgb)
        
        ## Render test sets in timestep target_timestep
        for time_idx in range(len(temporal_inputs)):
            os.makedirs(os.path.join(render_all_path, str(time_idx)), exist_ok=True)
            os.makedirs(os.path.join(capture_all_path, str(time_idx)), exist_ok=True)
            if not skip_localization:
                localize_render_gaussians(dataset, pipe, gaussians, scene, background, object_indices, time_idx, target_timestep, opacity_filters, object_tracks, est_mats, source_path, output_dir, os.path.join(render_all_path, str(time_idx)), os.path.join(capture_all_path, str(time_idx)), est_mats_errors)
            else:
                dataset.train_test_exp = False
                test_cam = [cam for cam in scene.test_cameras if cam.image_name.split('/')[0] == temporal_inputs[time_idx]] # scene.test_cameras 
                render_gaussians(dataset, pipe, gaussians, test_cam, background, object_indices, time_idx, target_timestep, opacity_filters, object_tracks, est_mats, os.path.join(render_all_path, str(time_idx)),  os.path.join(capture_all_path, str(time_idx)), est_mats_errors)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    lp = ModelParams(parser, sentinel=True)
    op = UpdateParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--use_previous_viewpoints", action="store_true")
    parser.add_argument("--compensate_exposure", action="store_true")
    parser.add_argument("--skip_localization", action="store_true")
    
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    update_3dgs(lp.extract(args), op.extract(args), pp.extract(args), args.iteration, args.use_previous_viewpoints, args.compensate_exposure, args.skip_localization)