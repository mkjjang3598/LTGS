import os
import sys
import numpy as np
import torch

sys.path.append('./gaussian-splatting')
from utils.graphics_utils import focal2fov
from scene.dataset_readers import CameraInfo
from scene.colmap_loader import qvec2rotmat, rotmat2qvec

from hloc import extract_features, match_features, pairs_from_exhaustive, visualization
from hloc.localize_sfm import QueryLocalizer, pose_from_cluster
from hloc.utils import viz_3d
import pycolmap
from pycolmap import Camera, Rigid3d, Rotation3d

import subprocess

sys.path.append('./submodules/mast3r')
from mast3r.cloud_opt.sparse_ga import sparse_global_alignment
from dust3r.utils.geometry import inv, geotrf  
from dust3r.image_pairs import make_pairs
from dust3r.utils.image import load_images
from functools import lru_cache
from pathlib import Path
import shutil
import tempfile
import copy
import json
import re

# def extract_sort_key(filename):
#     number_part = filename.split("/")[-1].split("_")[-1].split(".")[0]
#     return int(number_part)  # Convert to int for correct numerical sorting

def extract_sort_key(path):
    folder, filename = path.split('/')
    ext = "png" if filename.endswith("png") else "jpg"
    number = int(re.search(r'_(\d+)\.'+ext, filename).group(1))
    return (folder, number)

def colmap_localization(dataset):
    source_path = Path(dataset.source_path)
    change_path = source_path / Path("images/change.txt")
    
    image_path = source_path / Path("images")
    sfm_model_path = source_path / Path("sparse/0")  # COLMAP reconstruction folder with cameras, images, points3D
    database_path = source_path / Path("database.db")
    vocab_tree_path = source_path.parent / Path("vocab_tree_flickr100K_words256K.bin")
    output_path = source_path / Path("sparse/new") 
    os.makedirs(output_path, exist_ok=True)

    # Run COLMAP feature_extractor
    subprocess.run([
        "colmap", "feature_extractor",
        "--database_path", database_path,
        "--image_path", image_path,
        "--image_list_path", change_path
    ])
    
    # Run COLMAP vocab_tree_matcher
    subprocess.run([
        "colmap", "vocab_tree_matcher",
        "--database_path", database_path,
        "--VocabTreeMatching.vocab_tree_path", vocab_tree_path,
        "--VocabTreeMatching.match_list_path", change_path
    ])

    # Run COLMAP image_registrator
    subprocess.run([
        "colmap", "image_registrator",
        "--database_path", database_path,
        "--input_path", sfm_model_path,
        "--output_path", output_path
    ])

    # Run COLMAP bundle_adjuster
    subprocess.run([
        "colmap", "bundle_adjuster",
        "--input_path", output_path,
        "--output_path", output_path
    ])

    print("COLMAP localization pipeline done!")

def hloc_localization(dataset, known_intrinsics=False):
    source_path = Path(dataset.source_path) # ends with hloc
    images_path = source_path.parent / "images"
    query_path = images_path / "changes.txt"

    with open(query_path, 'r') as file:            
        image_filenames = file.read().strip().split()
    query_list = sorted(image_filenames, key = extract_sort_key) if image_filenames[0].startswith("IMG_") else sorted(image_filenames)
    
    # original features and matches from SfM
    sfm_features = source_path / "features.h5"
    sfm_matches = source_path / "matches.h5"
    sfm_pairs = source_path / "pairs-netvlad.txt"
    sfm_model_path = source_path / "sparse/0"
    model = pycolmap.Reconstruction(sfm_model_path)
    
    # fig = viz_3d.init_figure()
    # viz_3d.plot_reconstruction(fig, model, color='rgba(255,0,0,0.5)', name="mapping", points_rgb=True)

    outputs = source_path / "localization"
    features = outputs / "features.h5"
    matches = outputs / "matches.h5"
    loc_pairs = outputs / "pairs-loc.txt"
    results = outputs / "results.txt"

    os.makedirs(outputs, exist_ok=True)
    shutil.copy(sfm_features, features)
    shutil.copy(sfm_matches, matches)
    
    feature_conf = extract_features.confs["superpoint_aachen"]  # type: ignore
    matcher_conf = match_features.confs["superglue"]  # type: ignore
    references_registered = [model.images[i].name for i in model.reg_image_ids()]
    
    hloc_results = []

    for query in query_list:
        extract_features.main(feature_conf, images_path, image_list=[query], feature_path=features, overwrite=True)
        pairs_from_exhaustive.main(loc_pairs, image_list=[query], ref_list=references_registered)
        match_features.main(matcher_conf, loc_pairs, features=features, matches=matches, overwrite=True)
        if not known_intrinsics:  
            camera = pycolmap.infer_camera_from_image(images_path / query)
            refine_focal_length = True
        else:
            camera = pycolmap.Camera(
                model=model.cameras[1].model,
                width=model.cameras[1].width,
                height=model.cameras[1].height,
                params=model.cameras[1].params
            )
            refine_focal_length = False
        ref_ids = [model.find_image_with_name(n).image_id for n in references_registered]
        conf = {
            'estimation': {'ransac': {'max_error': 12}},
            'refinement': {'refine_focal_length': refine_focal_length, 'refine_extra_params': True},
        }
        localizer = QueryLocalizer(model, conf)

        ret, log = pose_from_cluster(localizer, query, camera, ref_ids, features, matches)
        print(f'found {ret["num_inliers"]}/{len(ret["inliers"])} inlier correspondences.')
        ret["name"] = query
        ret["points3D_ids"] = np.array(log['points3D_ids'])[ret['inliers']]
        ret["xys"] = np.array(log["keypoints_query"], dtype="float64")[ret['inliers']]
        # ##
        # pose.name = query      
        # if len(hloc_results) == 0:
        #     pose = pycolmap.Image(cam_from_world=ret['cam_from_world'])
        #     viz_3d.plot_camera_colmap(fig, pose, camera, color='rgba(0,255,0,0.5)', name=query, fill=True)
        #     # visualize 2D-3D correspodences
        #     inl_3d = np.array([model.points3D[pid].xyz for pid in np.array(log['points3D_ids'])[ret['inliers']]])
        #     viz_3d.plot_points(fig, inl_3d, color="lime", ps=1, name=query)
        #     fig.show()
        # ##

        hloc_results.append(ret)
    
    return hloc_results

def find_test_cam_poses(dataset, scene_info, known_intrinsics=True):
    source_path = Path(dataset.source_path) # ends with hloc
    images_path = source_path.parent / "images"
    test_set = [camera.image_name for camera in scene_info.test_cameras]
    query_list = sorted(test_set, key = extract_sort_key) if test_set[0].startswith("IMG_") else sorted(test_set)
    
    # original features and matches from SfM
    sfm_features = source_path / "features.h5"
    sfm_matches = source_path / "matches.h5"
    sfm_pairs = source_path / "pairs-netvlad.txt"
    sfm_model_path = source_path / "sparse/0"
    model = pycolmap.Reconstruction(sfm_model_path)
    
    # fig = viz_3d.init_figure()
    # viz_3d.plot_reconstruction(fig, model, color='rgba(255,0,0,0.5)', name="mapping", points_rgb=True)

    outputs = source_path / "localization"
    features = outputs / "features.h5"
    matches = outputs / "matches.h5"
    loc_pairs = outputs / "pairs-loc.txt"
    results = outputs / "results.txt"

    os.makedirs(outputs, exist_ok=True)
    shutil.copy(sfm_features, features)
    shutil.copy(sfm_matches, matches)
    
    feature_conf = extract_features.confs["superpoint_aachen"]  # type: ignore
    matcher_conf = match_features.confs["superglue"]  # type: ignore
    references_registered = [model.images[i].name for i in model.reg_image_ids()]
    
    hloc_results = []

    for query in query_list:
        if query not in references_registered:
            extract_features.main(feature_conf, images_path, image_list=[query], feature_path=features, overwrite=True)
            pairs_from_exhaustive.main(loc_pairs, image_list=[query], ref_list=references_registered)
            match_features.main(matcher_conf, loc_pairs, features=features, matches=matches, overwrite=True)
            if not known_intrinsics:  
                camera = pycolmap.infer_camera_from_image(images_path / query)
                refine_focal_length = True
            else:
                camera = pycolmap.Camera(
                    model=model.cameras[1].model,
                    width=model.cameras[1].width,
                    height=model.cameras[1].height,
                    params=model.cameras[1].params
                )
                refine_focal_length = False
            ref_ids = [model.find_image_with_name(n).image_id for n in references_registered]
            conf = {
                'estimation': {'ransac': {'max_error': 12}},
                'refinement': {'refine_focal_length': refine_focal_length, 'refine_extra_params': True},
            }
            localizer = QueryLocalizer(model, conf)
            
            ret, log = pose_from_cluster(localizer, query, camera, ref_ids, features, matches)
            print(f'found {ret["num_inliers"]}/{len(ret["inliers"])} inlier correspondences.')
            ret["name"] = query
            ret["points3D_ids"] = None # np.array(log['points3D_ids'])[ret['inliers']]
            ret["xys"] = None # np.array(log["keypoints_query"], dtype="float64")[ret['inliers']]
            ret["inliers"] = None
            ret["num_inliers"] = None
            
            # pose.name = query      
            # if len(hloc_results) == 0:
            #     pose = pycolmap.Image(cam_from_world=ret['cam_from_world'])
            #     viz_3d.plot_camera_colmap(fig, pose, camera, color='rgba(0,255,0,0.5)', name=query, fill=True)
            #     # visualize 2D-3D correspodences
            #     inl_3d = np.array([model.points3D[pid].xyz for pid in np.array(log['points3D_ids'])[ret['inliers']]])
            #     viz_3d.plot_points(fig, inl_3d, color="lime", ps=1, name=query)
            #     fig.show()
        
        else:
            ret = {}
            ret["name"] = query
            ret["camera"] = model.cameras[1]
            ret["cam_from_world"] = model.images[model.find_image_with_name(query).image_id].cam_from_world
            ret["inliers"] = None
            ret["num_inliers"] = None
            ret["points3D_ids"] = None
            ret["xys"] = None

        hloc_results.append(ret)
    
    return hloc_results

def save_hloc_results(hloc_results, hloc_result_path):
    output_data = []
    for entry in hloc_results:
        output_data.append({
            "name": entry['name'],
            "model": entry['camera'].model.name,
            "camera_id": entry['camera'].todict()['camera_id'],
            "width": entry['camera'].todict()['width'],            
            "height": entry['camera'].todict()['height'],            
            "rotation": entry['cam_from_world'].todict()['rotation']['quat'].tolist(),
            "translation": entry['cam_from_world'].todict()['translation'].tolist(),
            "params": entry['camera'].todict()['params'].tolist(),            
            "has_prior_focal_length": entry['camera'].todict()['has_prior_focal_length'],            
            "num_inliers": entry['num_inliers'] if entry['num_inliers'] is not None else None,
            "inliers": entry['inliers'].tolist() if entry['inliers'] is not None else None, 
            "points3D_ids": entry['points3D_ids'].tolist() if entry['points3D_ids'] is not None else None,
            "xys": entry["xys"].tolist() if entry["xys"] is not None else None
        })

    with open(hloc_result_path, 'w') as json_file:
        json.dump(output_data, json_file, indent=4)

def load_hloc_results(hloc_path):
    with open(hloc_path, "r") as f:
        loaded_results = json.load(f)

    hloc_results = []
    for entry in loaded_results:
        cam_from_world = Rigid3d(
            rotation = Rotation3d(np.array(entry['rotation'])), 
            translation = entry['translation']
        )
        camera = Camera(
            camera_id = entry['camera_id'],
            model = entry['model'], 
            width = entry['width'], 
            height = entry['height'], 
            params = np.array(entry['params']),
        )

        hloc_results.append({
            "cam_from_world": cam_from_world,
            "num_inliers": entry['num_inliers'] if entry['num_inliers'] is not None else None,
            "inliers": np.array(entry['inliers']) if entry['inliers'] is not None else None, 
            "camera": camera,
            "name": entry['name'],
            "points3D_ids": np.array(entry['points3D_ids'], dtype=np.int64) if entry['points3D_ids'] is not None else None,
            "xys": np.array(entry["xys"], dtype=np.float64) if entry["xys"] is not None else None
        })

    return hloc_results

def readHlocCameras(args, hloc_results, num_original_cameras, load_mast3r_depth=False, load_depth_anything=False):
    cam_infos = []
    scene_name = args.source_path.split("/")[-2] if str(args.source_path).endswith("hloc") else args.source_path.split("/")[-1]
    assert load_mast3r_depth is not True or load_depth_anything is not True, "Please choose only one depth source!"
    if load_depth_anything:
        depth_params_file = os.path.join("output", scene_name, "hloc", "depth_params.json")
        try:
            with open(depth_params_file, "r") as f:
                depths_params = json.load(f)
            all_scales = np.array([depths_params[key]["scale"] for key in depths_params])
            if (all_scales > 0).sum():
                med_scale = np.median(all_scales[all_scales > 0])
            else:
                med_scale = 0
            for key in depths_params:
                depths_params[key]["med_scale"] = med_scale

        except FileNotFoundError:
            print(f"Error: depth_params.json file not found at path '{depth_params_file}'.")
            sys.exit(1)
        except Exception as e:
            print(f"An unexpected error occurred when trying to open depth_params.json file: {e}")
            sys.exit(1)
            
    for idx, ret in enumerate(hloc_results):
        extr = pycolmap.Image(cam_from_world=ret["cam_from_world"])
        w2c = extr.cam_from_world
        intr = ret["camera"]
        height = intr.height
        width = intr.width
        uid = num_original_cameras+idx
        R = np.transpose(w2c.rotation.matrix())
        T = np.array(w2c.translation)
        
        if intr.model.name=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model.name=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model.name=="SIMPLE_RADIAL":
            focal_length_x = intr.params[0] 
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        n_remove = len(ret["name"].split('.')[-1]) + 1
        depth_params = None
        image_name = ret["name"]
        if args.source_path.endswith("hloc"):
            image_path = os.path.join(os.path.dirname(args.source_path), "images", image_name)
        else:
            image_path = os.path.join(args.source_path, "images", image_name)
            
        depth_path = ""
        if load_mast3r_depth:
            depth_path = os.path.join("output", args.source_path.split("/")[-2], "mast3r", "change_recon.npz")
        elif load_depth_anything: 
            depth_path = os.path.join(os.path.dirname(args.source_path), "depths", image_name[:-n_remove]+".png")
            try:
                depth_params = depths_params[image_name[:-n_remove]]
            except:
                print("\n", key, "not found in depths_params")
        
        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, depth_params=depth_params,
                              image_path=image_path, image_name=image_name, depth_path=depth_path,
                              width=width, height=height, is_test=True)
        cam_infos.append(cam_info)

    return cam_infos

class SparseGAState():
    def __init__(self, sparse_ga, should_delete=False, cache_dir=None, outfile_name=None):
        self.sparse_ga = sparse_ga
        self.cache_dir = cache_dir
        self.outfile_name = outfile_name
        self.should_delete = should_delete

    def __del__(self):
        if not self.should_delete:
            return
        if self.cache_dir is not None and os.path.isdir(self.cache_dir):
            shutil.rmtree(self.cache_dir)
        self.cache_dir = None
        if self.outfile_name is not None and os.path.isfile(self.outfile_name):
            os.remove(self.outfile_name)
        self.outfile_name = None

def get_reconstructed_scene(outdir, model, filelist, gradio_delete_cache=None, device='cuda', silent=False, image_size=512, current_scene_state=None,
                            optim_level = 'refine+depth', lr1=0.07, niter1=500, lr2=0.014, niter2=200, matching_conf_thr=5.,
                            scenegraph_type='complete', winsize=1, win_cyclic=False, refid=0, shared_intrinsics=False, cam2w=None, K=None, **kw):
    """
    from a list of images, run mast3r inference, sparse global aligner.
    then run get_3D_model_from_scene
    """
    imgs = load_images(filelist, size=image_size, verbose=not silent)
    if len(imgs) == 1:
        imgs = [imgs[0], copy.deepcopy(imgs[0])]
        imgs[1]['idx'] = 1
        filelist = [filelist[0], filelist[0] + '_2']

    scene_graph_params = [scenegraph_type]
    if scenegraph_type in ["swin", "logwin"]:
        scene_graph_params.append(str(winsize))
    elif scenegraph_type == "oneref":
        scene_graph_params.append(str(refid))
    if scenegraph_type in ["swin", "logwin"] and not win_cyclic:
        scene_graph_params.append('noncyclic')
    scene_graph = '-'.join(scene_graph_params)
    pairs = make_pairs(imgs, scene_graph=scene_graph, prefilter=None, symmetrize=True)
    if optim_level == 'coarse':
        niter2 = 0
    # Sparse GA (forward mast3r -> matching -> 3D optim -> 2D refinement -> triangulation)
    if current_scene_state is not None and \
        not current_scene_state.should_delete and \
            current_scene_state.cache_dir is not None:
        cache_dir = current_scene_state.cache_dir
    elif gradio_delete_cache:
        cache_dir = tempfile.mkdtemp(suffix='_cache', dir=outdir)
    else:
        # cache_dir = os.path.join(outdir, 'cache')
        cache_dir = outdir
    os.makedirs(cache_dir, exist_ok=True)
    
    scene = sparse_global_alignment(filelist, pairs, cache_dir,
                                    model, lr1=lr1, niter1=niter1, lr2=lr2, niter2=niter2, device=device,
                                    opt_depth='depth' in optim_level, shared_intrinsics=shared_intrinsics,
                                    matching_conf_thr=matching_conf_thr, cam2w=cam2w, K=K, **kw)
    if current_scene_state is not None and \
        not current_scene_state.should_delete and \
            current_scene_state.outfile_name is not None:
        outfile_name = current_scene_state.outfile_name
    else:
        outfile_name = tempfile.mktemp(suffix='_scene.glb', dir=outdir)

    scene_state = SparseGAState(scene, gradio_delete_cache, cache_dir, outfile_name)
    
    return scene_state


def transform_colmap2hloc(scene_info, hloc_cameras):
    # Tune test camera's pose based on the HLOC coordinate
    change_images = [camera.image_name for camera in hloc_cameras]
    centers_colmap, centers_hloc = [], []
    for camera in scene_info.train_cameras:
        if camera.image_name not in change_images:
            continue

        idx = change_images.index(camera.image_name)            
        R_colmap, T_colmap = camera.R.T, camera.T # c2w to w2c
        C_colmap = - R_colmap.T @ T_colmap
        R_hloc, T_hloc = hloc_cameras[idx].R.T, hloc_cameras[idx].T
        C_hloc = - R_hloc.T @ T_hloc
        
        centers_colmap.append(C_colmap)
        centers_hloc.append(C_hloc)
        
    # compute relative scale, rotation, centers
    s_rel = np.linalg.norm((centers_hloc[1] - centers_hloc[0])) / np.linalg.norm((centers_colmap[1] - centers_colmap[0])) 
    coord_rel = R_hloc @ R_colmap.T
    
    hloc_testcam = []
    for camera in scene_info.test_cameras:
        # scene_info cameras are in c2w
        C_cam = - camera.R @ camera.T
        R_new = coord_rel @ camera.R.T

        C_new = C_hloc + s_rel * coord_rel.T @ (C_cam - C_colmap)
        T_new = - R_new @ C_new

        cam_info = CameraInfo(uid=camera.uid, R=R_new.T, T=T_new, FovY=camera.FovY, FovX=camera.FovX, depth_params=camera.depth_params,
                        image_path=camera.image_path, image_name=camera.image_name, depth_path=camera.depth_path,
                        width=camera.width, height=camera.height, is_test=camera.is_test)
        hloc_testcam.append(cam_info)
    
    return hloc_testcam

def hloc_results_from_colmap(change_cameras):
    hloc_results = []
    for camera in change_cameras:
        cam_from_world = Rigid3d(
            rotation = Rotation3d(np.transpose(camera.R)), 
            translation = camera.T
        )
        hloc_camera = Camera(
            camera_id = camera.uid,
            model = "SIMPLE_PINHOLE", 
            width = camera.image_width, 
            height = camera.image_height, 
            params = np.array([camera.focal_y, camera.image_width / 2, camera.image_height / 2]),
        )
        hloc_results.append({
            "cam_from_world": cam_from_world,
            "num_inliers": None,
            "inliers": None,  
            "camera": hloc_camera,
            "name": camera.image_name,
            "points3D_ids": None,
            "xys": None
        })

    return hloc_results
