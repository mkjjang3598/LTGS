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
from src.utils.update_utils import quaternion_to_matrix, matrix_to_quaternion, sh_rotation, nearest_distances_ckdtree

from random import randint
import uuid
from scene import Scene
from gaussian_renderer import render, network_gui
from utils.image_utils import psnr
from utils.loss_utils import masked_l1_loss, l1_loss, ssim
from utils.camera_utils import cameraList_from_camInfos
from utils.general_utils import get_expon_lr_func
from scene.dataset_readers import CameraInfo, sceneLoadTypeCallbacks

from arguments import ModelParams, UpdateParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
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

def render_gaussians(dataset, pipe, gaussians, cameras, background, render_path=None, gt_path=None):
    renderings, gts = [], []
    for idx, view in enumerate(cameras):
        rendering = render(view, gaussians, pipe, background, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)["render"]
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

def localize_render_gaussians(dataset, pipe, gaussians, scene, background, source_path, output_dir, render_all_path, gt_all_path):
    images_txt_path = source_path.parent / Path("images/changes.txt")
    test_hloc_path = os.path.join(output_dir, "update", "test_hloc_results.json")
    with open(images_txt_path, 'r') as file:            
        images_path = file.read().strip().split()
    change_dir = images_path[0].split('/')[0]
    
    scene_info = sceneLoadTypeCallbacks["Colmap"](str(source_path.parent), dataset.images, dataset.depths, True, dataset.train_test_exp, change_dir=change_dir)
    if not os.path.exists(test_hloc_path):
        test_hloc_results = find_test_cam_poses(dataset, scene_info)
        save_hloc_results(test_hloc_results, test_hloc_path)
    else:
        test_hloc_results = load_hloc_results(test_hloc_path)

    test_hloc_cameras = readHlocCameras(dataset, test_hloc_results, num_original_cameras=len(scene.train_cameras)+len(scene.test_cameras))
    test_cameras = cameraList_from_camInfos(test_hloc_cameras, 1.0, dataset, scene_info.is_nerf_synthetic, True)
    print("Rendering Test Sets")

    dataset.train_test_exp = False
    render_gaussians(dataset, pipe, gaussians, test_cameras, background, render_all_path, gt_all_path)

def update_3dgs(dataset: ModelParams, opt: UpdateParams, pipe: PipelineParams, iteration: int, num_sample:int, load_mast3r_depth:bool, load_depth_anything:bool, dist_thres: float):
    with torch.no_grad():
        source_path = Path(dataset.source_path)
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        scene_name = source_path.parent.stem

        output_dir = os.path.join("output", scene_name)
        gt_path = os.path.join(output_dir, "change", "gt")
        render_path = os.path.join(output_dir, "change", "renders")

        gt_mask_dir = os.path.join(output_dir, "instances", "gt")
        render_mask_dir = os.path.join(output_dir, "instances", "renders")
        
        # Load before and after pcds
        before_pcd_path = os.path.join(output_dir, "instances", "pcd_before.pkl")
        after_pcd_path = os.path.join(output_dir, "instances", "pcd_after_0.pkl")

        hloc_path =  os.path.join(output_dir, "hloc", "hloc_results.json")
        hloc_results = load_hloc_results(hloc_path)
        hloc_cameras = readHlocCameras(dataset, hloc_results, num_original_cameras=len(scene.train_cameras)+len(scene.test_cameras), load_mast3r_depth=load_mast3r_depth, load_depth_anything=load_depth_anything)
        change_cameras = cameraList_from_camInfos(hloc_cameras, 1.0, dataset, False, True)
        filenames = sorted(os.listdir(gt_path))

        imagefiles, object_masks, image_batch = [], [], []
        for filename in filenames:                            
            imagefiles.append(os.path.join(render_path, filename))
            imagefiles.append(os.path.join(gt_path, filename))

            image_batch.append(np.array(Image.open(os.path.join(render_path, filename)).convert("RGB")))
            image_batch.append(np.array(Image.open(os.path.join(gt_path, filename)).convert("RGB")))
            
            object_masks.append(torch.from_numpy(np.load(os.path.join(render_mask_dir, filename.split('.')[0]+'.npy'))).to(torch.float32))   
            object_masks.append(torch.from_numpy(np.load(os.path.join(gt_mask_dir, filename.split('.')[0]+'.npy'))).to(torch.float32))   
        num_objects = [int(obj_mask.max().item()) for obj_mask in object_masks]

        with open(before_pcd_path, 'rb') as f:
            pcd_before_loaded = pickle.load(f)
        
        with open(after_pcd_path, 'rb') as f:
            pcd_after_loaded = pickle.load(f)

    est_mats, removed_obj_labels, matching_obj_labels = register_pcds(imagefiles, object_masks, change_cameras, pcd_before_loaded, pcd_after_loaded, output_dir, num_sample, dist_thres)
    refine_optimize(dataset, opt, pipe, gaussians, scene, hloc_cameras, change_cameras, pcd_before_loaded, pcd_after_loaded, est_mats, removed_obj_labels, matching_obj_labels, object_masks)
    

def register_pcds(imagefiles, object_masks, change_cameras, pcd_before_loaded, pcd_after_loaded, output_dir, num_sample = 1000,
    dist_thres=0.2):
    descriptor_path = os.path.join(output_dir, "descriptors")
    os.makedirs(descriptor_path, exist_ok=True)
    render_indices = [idx for idx in range(len(imagefiles)) if idx % 2 == 0]
    gt_indices = [idx for idx in range(len(imagefiles)) if idx % 2 != 0]

    before_obj_indices, after_obj_indices = [], []
    for idx, object_mask in enumerate(object_masks):
        obj_indices = [int(obj.max()) for _, obj in enumerate(object_mask)]
        for obj_idx in obj_indices:
            if idx in render_indices and obj_idx not in before_obj_indices:
                before_obj_indices.append(obj_idx) 
            elif idx in gt_indices and obj_idx not in after_obj_indices:
                after_obj_indices.append(obj_idx)
    
    # define remove, inserted, matching objects
    removed_obj_labels = [idx for idx in before_obj_indices if idx not in after_obj_indices]
    inserted_obj_labels = [idx for idx in after_obj_indices if idx not in before_obj_indices]
    matching_obj_labels = [idx for idx in before_obj_indices if idx in after_obj_indices]
    
    descriptors = extract_2d_descriptors(matching_obj_labels, object_masks, imagefiles, descriptor_path, input_mode="full")
    
    before_pcds = {idx: pcd_before_loaded[idx]['pcd'] for idx in pcd_before_loaded.keys()}
    after_pcds = {idx: pcd_after_loaded[idx]['pcd'] for idx in pcd_after_loaded.keys()}

    fused_outputs = fuse_multiview_descriptors(before_pcds, after_pcds, change_cameras, matching_obj_labels, descriptors, object_masks, render_indices, descriptor_path)
    
    for obj_label in matching_obj_labels:
        if len(fused_outputs[obj_label]['desc_3d_2']) < num_sample:
            idx = np.where(np.array(matching_obj_labels) == obj_label)[0].item()
            matching_obj_labels.pop(idx)
            removed_obj_labels.append(obj_label)
    
    obj_kpts_1, obj_kpts_2 = find_3d_correspondences(before_pcds, after_pcds, fused_outputs, matching_obj_labels, descriptor_path, num_sample)
    est_mats = run_teaserpp(obj_kpts_1, obj_kpts_2, matching_obj_labels)

    # If fail to register, change labels
    for obj_label in matching_obj_labels:
        est_mat = est_mats[obj_label]
        transformed_obj_kpts = obj_kpts_1[obj_label] @ est_mat[:3, :3].T + est_mat[:3, 3:4].T 
        dist = chamfer_distance(transformed_obj_kpts[None], obj_kpts_2[obj_label][None])
        print("Chamfer Distance between point cloud : ", dist)
        if dist > dist_thres:
            idx = np.where(np.array(matching_obj_labels) == obj_label)[0].item() 
            matching_obj_labels.pop(idx)
            removed_obj_labels.append(obj_label)

    return est_mats, removed_obj_labels, matching_obj_labels


def refine_optimize(dataset, opt, pipe, gaussians, scene, hloc_cameras, change_cameras, pcd_before_loaded, pcd_after_loaded, est_mats, removed_obj_labels, matching_obj_labels, object_masks):
    source_path = Path(dataset.source_path)
    scene_name = source_path.parent.stem
    scene.model_path = scene.model_path + "_update" 
    ### Important changes in configurations ###
    dataset.model_path = scene.model_path
    dataset.train_test_exp = True 
    dataset.depths = ""
    # Set maximum SH degree to 1
    # gaussians.active_sh_degree = 1

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    output_dir = os.path.join("output", scene_name)
    render_path = os.path.join(output_dir, "update", "renders")
    gt_path = os.path.join(output_dir, "update", "gt")
    render_before_path = os.path.join(output_dir, "update", "before_all")
    render_initialized_path = os.path.join(output_dir, "update", "initialized_all")
    render_all_path = os.path.join(output_dir, "update", "render_all")
    gt_all_path = os.path.join(output_dir, "update", "gt_all")
    
    os.makedirs(render_path, exist_ok=True)
    os.makedirs(gt_path, exist_ok=True)
    os.makedirs(render_before_path, exist_ok=True)
    os.makedirs(render_initialized_path, exist_ok=True)
    os.makedirs(render_all_path, exist_ok=True)
    os.makedirs(gt_all_path, exist_ok=True)

    # # Rendering before gaussians
    localize_render_gaussians(dataset, pipe, gaussians, scene, background, source_path, output_dir, render_all_path=render_before_path, gt_all_path=gt_all_path)

    ## Rotate and initialize rotated 3DGS (GaussReg)  
    with torch.no_grad():       
        update_indices = []
        # Initialize by registration
        for obj_label in matching_obj_labels:
            indices = pcd_before_loaded[obj_label]['indices']
            est_mat = torch.from_numpy(est_mats[obj_label]).to(torch.float32).cuda()

            # Rotate 3DGS
            xyz = gaussians.get_xyz[indices]
            rotation = quaternion_to_matrix(gaussians.get_rotation[indices])
            shs = gaussians.get_features[indices]
            h_dc, sh_rest = shs[:,0:1], shs[:,1:]

            rotated_xyz = (est_mat[:3, :3] @ xyz.T + est_mat[:3, 3:4]).T   
            rotated_rotation = matrix_to_quaternion(est_mat[:3, :3] @ rotation)
            rotated_sh_rest = sh_rotation(sh_rest, est_mat[:3, :3])

            # Modify 3DGS
            gaussians._xyz[indices] = rotated_xyz
            gaussians._rotation[indices] = rotated_rotation
            gaussians._features_rest[indices] = rotated_sh_rest
            update_indices.append(indices)
        
        if len(update_indices) > 0:
            update_indices = np.concatenate(update_indices, axis=0).tolist()

    gaussians.optimizer_type = opt.optimizer_type
    gaussians.training_setup_pp(opt, hloc_cameras) 

    ## Remove objects 
    prune_mask = np.zeros(len(gaussians.get_xyz), dtype=bool)
    remove_indices = []
    for obj_label in removed_obj_labels:
        indices = pcd_before_loaded[obj_label]['indices']
        remove_indices.append(indices)
        print("Number of points removed : ", indices.shape[0])

    if len(remove_indices) > 0:
        remove_indices = np.concatenate(remove_indices, axis=0)
        remove_indices = np.sort(remove_indices)
        prune_mask[remove_indices] = True
        gaussians.prune_points(prune_mask)    
        
        mapping = np.cumsum(~prune_mask) - 1
        update_indices = [mapping[i] for i in update_indices if i not in prune_mask]

    ## Initialize MASt3R pcds (Refer to InstantSplat's method)
    update_indices = [np.array(update_indices)]
    for obj_label in list(pcd_after_loaded.keys()):
        if obj_label in matching_obj_labels:
            continue
        xyz, rgb = pcd_after_loaded[obj_label]['pcd'], pcd_after_loaded[obj_label]['rgb']
        # print(gaussians.get_xyz.shape[0])
        indices = gaussians.get_xyz.shape[0] + np.linspace(0, len(xyz)-1, len(xyz), dtype=np.int64)
        gaussians.initialize_from_mast3r_pp(xyz, rgb, spatial_lr_scale=scene.cameras_extent)
        # print(indices)
        update_indices.append(indices)
    
    update_indices = np.concatenate(update_indices, axis=0) 
    update = torch.zeros(gaussians.get_xyz.shape[0], dtype=torch.bool, device="cuda")
    update[update_indices] = True

    full_gaussians_xyz = gaussians.get_xyz
    update_gaussians_xyz = full_gaussians_xyz[update]
    with torch.no_grad():
        dist = nearest_distances_ckdtree(update_gaussians_xyz, full_gaussians_xyz)

    # truncation
    # dist_threshold = 1
    # additional_indices = dist < dist_threshold
    # non-truncation
    additional_indices = torch.ones_like(dist, dtype=torch.bool, device="cuda")
    additional_indices = additional_indices & ~update
    additional_indices = torch.where(additional_indices)[0]
    print("# updated points: ", len(update_indices))
    print("# Additional points to be updated: ", len(additional_indices))
    additional_lr = torch.exp(-5 * dist)
    gaussians.activate_per_point_lr(additional_indices, additional_lr)
    
    # ## Visualize regions to be updated
    # points_rgb = np.zeros((len(update), 3))
    # points_rgb[update_indices.astype(int), 0] = 1
    # points_rgb[additional_indices.astype(int), 1] = additional_lr[additional_indices.astype(int)].detach().cpu().numpy()
    # visualize_geometry({}, points3d_xyz=gaussians.get_xyz.detach().cpu().numpy(), points3d_rgb=points_rgb)

    # Add object mask
    # update_mask = [torch.cat((object_masks[2*i], object_masks[2*i+1]),dim=0) for i in range(int(len(object_masks)/2))]
    # update_mask = [mask.sum(0) > 0.01 for mask in update_mask]

    ## Render initialized gaussians
    localize_render_gaussians(dataset, pipe, gaussians, scene, background, source_path, output_dir, render_all_path=render_initialized_path, gt_all_path=gt_all_path)
        
    ## Refine 3DGS with extra training (Training_code)   
    first_iter = 0 
    last_iter = first_iter + opt.iterations
    tb_writer = prepare_output_and_logger(dataset)
    
    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    use_sparse_adam = opt.optimizer_type == "sparse_adam" and SPARSE_ADAM_AVAILABLE 
    depth_l1_weight = get_expon_lr_func(opt.depth_l1_weight_init, opt.depth_l1_weight_final, max_steps=opt.iterations)

    viewpoint_stack = change_cameras.copy()
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
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifier=scaling_modifer, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)["render"]
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
            viewpoint_indices = list(range(len(viewpoint_stack)))
        rand_idx = randint(0, len(viewpoint_indices) - 1)
        viewpoint_cam = viewpoint_stack.pop(rand_idx)
        vind = viewpoint_indices.pop(rand_idx)

        # Render
        bg = torch.rand((3), device="cuda") if opt.random_background else background
        render_pkg = render(viewpoint_cam, gaussians, pipe, bg, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)
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

        loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_Ll1depth_for_log = 0.4 * Ll1depth + 0.6 * ema_Ll1depth_for_log

            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}", "Depth Loss": f"{ema_Ll1depth_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == last_iter:
                progress_bar.close()

            # Log and save
            if iteration == last_iter:
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification -> We don't use densification for updates
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold, radii)
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < last_iter:
                gaussians.exposure_optimizer.step()
                gaussians.exposure_optimizer.zero_grad(set_to_none = True)
        
                if use_sparse_adam:
                    visible = radii > 0
                    visible = visible & update
                    gaussians.optimizer.step(visible, radii.shape[0])
                    gaussians.optimizer.zero_grad(set_to_none = True)
                else:
                    gaussians.optimizer.step()
                    gaussians.optimizer.zero_grad(set_to_none = True)
    
    with torch.no_grad():    
        ## Visualize optimized 3DGS
        # visualize_geometry({}, points3d_xyz=gaussians.get_xyz.detach().cpu().numpy(), points3d_rgb=points_rgb)
        ## Finally render the changed views
        render_gaussians(dataset, pipe, gaussians, change_cameras, background, render_path, gt_path)
        ## Render test sets in timestep 1
        localize_render_gaussians(dataset, pipe, gaussians, scene, background, source_path, output_dir, render_all_path, gt_all_path)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    lp = ModelParams(parser, sentinel=True)
    op = UpdateParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--timestep", default=1, type=int)
    parser.add_argument("--num_sample", default=1000, type=int)
    parser.add_argument("--dist_thres", default=0.25, type=float)
    parser.add_argument("--load_depth_anything", action="store_true")
    parser.add_argument("--load_mast3r_depth", action="store_true")
    
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    update_3dgs(lp.extract(args), op.extract(args), pp.extract(args), args.iteration, args.num_sample, args.load_mast3r_depth, args.load_depth_anything, args.dist_thres)
