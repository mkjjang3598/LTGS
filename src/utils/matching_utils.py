from cv2 import compare
import torch
import os
import matplotlib.pyplot as plt
from utils.graphics_utils import fov2focal

from pathlib import Path
from PIL import Image
import numpy as np
from collections import defaultdict

from src.utils.localization_utils import get_reconstructed_scene, load_hloc_results, readHlocCameras
from src.utils.visualization_utils import ColorInfo, plot_object_masks, visualize_geometry, visualize_object_points, plot_matches

from scipy.optimize import linear_sum_assignment
from itertools import combinations 

from mast3r.model import AsymmetricMASt3R
from mast3r.utils.misc import hash_md5
from mast3r.cloud_opt.sparse_ga import load_corres, convert_dust3r_pairs_naming, forward_mast3r
from dust3r.utils.device import to_numpy
from dust3r.utils.image import load_images
from dust3r.image_pairs import make_pairs

ratios_resolutions = {
    4 / 3: [384, 512], 32 / 21: [336, 512], 16 / 9: [288, 512], 2 / 1: [256, 512], 16 / 5: [160, 512]
}

def get_h_w(H, W):
    ratio = W / H
    ref_ratios = np.array([*(ratios_resolutions.keys())])
    islandscape = (W >= H)
    if islandscape:
        diff = np.abs(ratio - ref_ratios)
    else:
        diff = np.abs(ratio - (1 / ref_ratios))
    selkey = ref_ratios[np.argmin(diff)]
    res = ratios_resolutions[selkey]
    return res

def remap_object_indices(final_object_list):
    unique_ids = sorted(set(final_object_list.values()))
    remap = {old: new for new, old in enumerate(unique_ids)}
    return {k: remap[v] for k, v in final_object_list.items()}

def dfs(graph, node, component, visited):
    """ Depth-First Search to find all connected objects """
    if node in visited:
        return
    visited.add(node)
    component.append(node)
    for neighbor in graph[node]:
        dfs(graph, neighbor, component, visited)

def disambiguate_objects(object_matches, object_masks):
    object_list = {}
    # Step 1: Build adjacency list
    graph = defaultdict(set)

    for match in object_matches:
        img1, img2 = match['image_pair']
        obj1, obj2 = match['object_pair']
        
        node1 = (img1, obj1)
        node2 = (img2, obj2)
        
        graph[node1].add(node2)
        graph[node2].add(node1)

    # Step 2: Find connected components (DFS/BFS)
    visited = set()
    linked_objects = []

    for node in graph:
        if node not in visited:
            component = []
            dfs(graph, node, component, visited)
            linked_objects.append(component)

    # Step 3: Print the linked objects across images
    for i, group in enumerate(linked_objects):
        for element in group:
            object_list[element] = i
        obj_idx = i
        
    for i in range(len(object_masks)):
        for mask_idx, _ in enumerate(object_masks[i]):
            if (i, mask_idx) not in visited:
                obj_idx += 1
                object_list[(i, mask_idx)] = obj_idx

    return object_list

def track_mast3r_matches(inputfiles, input_object_masks, tmp_pairs, min_conf_thr, num_match_thres, H, W, filter_consistent, symmetrize=True):
    filtered_matches = {}
    
    matches = {i: {j: [None, None] for j in range(i+1, len(inputfiles))} for i in range(len(inputfiles))}
    scores = {}

    for img in inputfiles:
        for (img1, img2), ((path1, path2), path_corres) in tmp_pairs.items():
            i, j = inputfiles.index(img1), inputfiles.index(img2)
            if symmetrize and i > j:
                continue                          
            if img == img1:
                X, C, X2, C2 = torch.load(path1, map_location='cuda')
                score, (xy1, xy2, confs) = load_corres(path_corres, 'cuda', min_conf_thr)
            else: 
                continue

            h, w = get_h_w(H, W)
            kpts_i, kpts_j = xy1.detach().cpu().numpy().astype("float32"), xy2.detach().cpu().numpy().astype("float32")
            kpts_i[:, 0] *= (W-1) / (w-1)
            kpts_i[:, 1] *= (H-1) / (h-1)
            kpts_j[:, 0] *= (W-1) / (w-1)
            kpts_j[:, 1] *= (H-1) / (h-1)
            kpts_i, kpts_j = np.round(kpts_i).astype("int"), np.round(kpts_j).astype("int")

            obj_kpt_ij = []
            for obj_idx_i, obj_mask_i in enumerate(input_object_masks[i]):
                kpt_idx_i = np.where(obj_mask_i[kpts_i[:, 1], kpts_i[:, 0]])[0]
                num_matches = []

                for obj_idx_j, obj_mask_j in enumerate(input_object_masks[j]):
                    kpt_ij = np.where(obj_mask_j[kpts_j[kpt_idx_i, 1], kpts_j[kpt_idx_i, 0]])[0]
                    obj_kpt_ij.append(kpt_idx_i[kpt_ij])
                    num_matches.append(len(kpt_ij))
                    
                num_matches = np.array(num_matches)
                
                obj_j = np.argmax(num_matches).item()
                max_match = np.max(num_matches).item()

                key = ((i, j), obj_j) 

                # Keep only the best object match per (image_pair, obj_j)
                if key not in filtered_matches or max_match > filtered_matches[key][1]:
                    if max_match < num_match_thres:
                        continue
                    filtered_matches[key] = (obj_idx_i, max_match)
            
            obj_kpt_ij = np.concatenate(obj_kpt_ij, 0)    
            matches[i][j][0], matches[i][j][1] = kpts_i[obj_kpt_ij], kpts_j[obj_kpt_ij]
    
    # Convert dictionary back to list format
    object_matches = [
        {"image_pair": img_pair, "object_pair": (obj_i, obj_j), "num_match": num_match}
        for (img_pair, obj_j), (obj_i, num_match) in filtered_matches.items()
    ]
    print("object_matches :", object_matches)
    
    if len(object_matches) > 0:
        object_list = disambiguate_objects(object_matches, input_object_masks)
    else:
        object_list = {}

    if filter_consistent:
        counts = defaultdict(int)
        for (image_idx, _), obj_idx in object_list.items():
            counts[obj_idx] += 1

        valid_objs = {obj for obj in object_list.values() if (counts[obj] == len(inputfiles) or counts[obj] == 0)}
        valid_object_entries = {k: v for k, v in object_list.items() if v in valid_objs}

        unique_ids = sorted(set(valid_object_entries.values()))
        remap = {old: new for new, old in enumerate(unique_ids)}

        object_list = {k: remap[v] for k, v in valid_object_entries.items()}
    else:
        valid_object_entries = {k: v for k, v in object_list.items()}

        unique_ids = sorted(set(valid_object_entries.values()))
        remap = {old: new for new, old in enumerate(unique_ids)}

        object_list = {k: remap[v] for k, v in valid_object_entries.items()}
    
    print("object_list :", object_list)

    return matches, object_list

## 2D instance matching
def intra_instance_matching(dataset, object_masks: list, optim_level: str, matching_conf_thres: float, min_conf_thr:float, filter_consistent: bool, num_match_threshold: int, timestep:int, temporal_indices: list):
    source_path = Path(dataset.source_path)
    scene_name = source_path.parent.stem if str(source_path).endswith("hloc") else source_path.stem
    output_dir = os.path.join("output", scene_name)

    instance_dir = os.path.join(output_dir, "instances")
    os.makedirs(instance_dir, exist_ok=True)

    hloc_path =  os.path.join(output_dir, "hloc", "hloc_results.json")
    hloc_results = load_hloc_results(hloc_path)

    hloc_results = [hloc_results[idx] for idx in temporal_indices]

    hloc_cameras = readHlocCameras(dataset, hloc_results, num_original_cameras=10000)
    hloc_w2c = np.tile(np.eye(4), (len(hloc_cameras), 1, 1)) # w2c : 3dgs
    
    for idx, hloc_cam in enumerate(hloc_cameras):   
        hloc_w2c[idx, :3, :3] = np.transpose(hloc_cam.R) # w2c : change to original COLMAP
        hloc_w2c[idx, :3, 3] = hloc_cam.T 
    hloc_c2w = np.linalg.inv(hloc_w2c) # c2w : original colmap 
    num_cameras = len(hloc_cameras)

    capture_path = os.path.join(output_dir, "change", "capture")
    render_path = os.path.join(output_dir, "change", "renders")
    filenames = [filename for idx, filename in enumerate(sorted(os.listdir(capture_path))) if idx in temporal_indices]

    captures, renderings, imagefiles, image_batch = [], [], [], []
    for idx, filename in enumerate(filenames):
        renderings.append(np.array(Image.open(os.path.join(render_path, filename))))
        captures.append(np.array(Image.open(os.path.join(capture_path, filename))))
        
        image_batch.append(np.array(Image.open(os.path.join(render_path, filename)).convert("RGB")))
        image_batch.append(np.array(Image.open(os.path.join(capture_path, filename)).convert("RGB")))
                            
        imagefiles.append(os.path.join(render_path, filename))
        imagefiles.append(os.path.join(capture_path, filename))
    
    render_indices = [idx for idx in range(len(imagefiles)) if idx % 2 == 0]
    capture_indices = [idx for idx in range(len(imagefiles)) if idx % 2 != 0]

    H, W = image_batch[0].shape[0], image_batch[0].shape[1]
    h, w = get_h_w(H, W)
    object_masks = [object_mask for idx, object_mask in enumerate(object_masks) if (idx // 2) in temporal_indices]
    
    #### Use MASt3R for 1) object instance tracking and 2) pcd initialization ####
    weights_path = 'naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric'
    model = AsymmetricMASt3R.from_pretrained(weights_path).cuda()
    subsample = 8
    
    object_lists = []
    matches_all = []
    for input in list(["Rendering", "Change"]):   
        print("object tracking using MASt3R for", input)

        if input == "Rendering":
            input_indices = render_indices
            inputfiles = [imagefiles[idx] for idx in input_indices]
            input_object_masks =  [object_masks[idx] for idx in input_indices]

            cache_dir = os.path.join(output_dir, "mast3r", "cache", str(timestep), "renders")
            os.makedirs(cache_dir, exist_ok=True)
            
            imgs = load_images(inputfiles, size=512, verbose=False)
            scene_graph = '-'.join(["complete"])
            pairs = make_pairs(imgs, scene_graph=scene_graph, prefilter=None, symmetrize=True)
            # Convert pair naming convention from dust3r to mast3r
            pairs_in = convert_dust3r_pairs_naming(inputfiles, pairs)
            # forward pass
            res_paths, recon_cache_dir = forward_mast3r(pairs_in, model,
                                            cache_path=cache_dir, subsample=subsample,
                                            desc_conf="desc_conf", device="cuda")
            
        elif input == "Change":
            input_indices = capture_indices
            inputfiles = [imagefiles[idx] for idx in input_indices]
            input_object_masks =  [object_masks[idx] for idx in input_indices]
            
            cache_path = os.path.join(output_dir, "mast3r", "cache", str(timestep), "capture")
            os.makedirs(cache_path, exist_ok=True)

            cam2w = torch.from_numpy(hloc_c2w).to(torch.float32).to("cuda")
            focal = fov2focal(hloc_cameras[0].FovY, h) # (h, w) = (288, 512)
            K = torch.eye(3).to("cuda")
            K[0, 0] = focal
            K[1, 1] = focal
            K[0, 2] = w / 2
            K[1, 2] = h / 2
            K = torch.tile(K.unsqueeze(0), (num_cameras, 1, 1))
            
            scene_recon = get_reconstructed_scene(cache_path, model, inputfiles, shared_intrinsics=False, cam2w=cam2w, K=K, silent=False, optim_level=optim_level, matching_conf_thr=matching_conf_thres)
            pairs_in = scene_recon.sparse_ga.pairs_in
            recon_cache_dir = scene_recon.cache_dir
            
            # 3D pointcloud from depthmap, poses and intrinsics
            pts3d, depthmaps, confs = to_numpy(scene_recon.sparse_ga.get_dense_pts3d(clean_depth=True))
            imgs = to_numpy(scene_recon.sparse_ga.imgs)
            mast3r_c2w = to_numpy(scene_recon.sparse_ga.get_im_poses())
            mast3r_intrinsics = to_numpy(scene_recon.sparse_ga.intrinsics)

            mast3r_recon_path = os.path.join(output_dir, "mast3r", f"change_recon_{timestep}.npz")
            mast3r_outputs = {
                "imgs": imgs,
                "pts3d": pts3d,
                "depthmaps": depthmaps,
                "confs": confs,
                "mast3r_c2w": mast3r_c2w,
                "mast3r_intrinsics": mast3r_intrinsics
            }
            
            np.savez(mast3r_recon_path, imgs=imgs, pts3d=pts3d, depthmaps=depthmaps, confs=confs, mast3r_c2w=mast3r_c2w)

        tmp_pairs = {}
        for img1, img2 in pairs_in:
            idx1 = hash_md5(img1['instance'])
            idx2 = hash_md5(img2['instance'])

            path1 = recon_cache_dir + f'/forward/{idx1}/{idx2}.pth'
            path2 = recon_cache_dir + f'/forward/{idx2}/{idx1}.pth'
                    
            path_corres = recon_cache_dir + f'/corres_conf=desc_conf_subsample={subsample}/{idx1}-{idx2}.pth'
            tmp_pairs[img1['instance'], img2['instance']] = (path1, path2), path_corres

        matches, object_list = track_mast3r_matches(inputfiles, input_object_masks, tmp_pairs, min_conf_thr, num_match_threshold, H, W, filter_consistent)
        matches_all.append(matches)
        object_lists.append(object_list)

    plot_matches(matches_all, image_batch, output_dir, render_indices, capture_indices, timestep=timestep)

    return object_lists, mast3r_outputs

def inter_instance_matching(dataset, object_lists: list, object_masks: list, embeddings_loaded: torch.Tensor, filter_consistent:bool, similarity_thres: float, timestep:int, temporal_indices: list, compare_mean=False, use_hungarian=False):
    source_path = Path(dataset.source_path)
    scene = source_path.parent.stem if str(source_path).endswith("hloc") else source_path.stem
    output_dir = os.path.join("output", scene)
    
    capture_path = os.path.join(output_dir, "change", "capture")
    render_path = os.path.join(output_dir, "change", "renders")
    filenames = [filename for idx, filename in enumerate(sorted(os.listdir(capture_path))) if idx in temporal_indices]
    
    imagefiles, image_batch = [], []
    for idx, filename in enumerate(filenames):
        image_batch.append(np.array(Image.open(os.path.join(render_path, filename)).convert("RGB")))
        image_batch.append(np.array(Image.open(os.path.join(capture_path, filename)).convert("RGB")))
                            
        imagefiles.append(os.path.join(render_path, filename))
        imagefiles.append(os.path.join(capture_path, filename))    
    
    render_indices = [idx for idx in range(len(imagefiles)) if idx % 2 == 0]
    capture_indices = [idx for idx in range(len(imagefiles)) if idx % 2 != 0]

    H, W = image_batch[0].shape[0], image_batch[0].shape[1]
    
    object_masks = [object_mask for idx, object_mask in enumerate(object_masks) if (idx // 2) in temporal_indices]
    embeddings_loaded = [embedding for idx, embedding in enumerate(embeddings_loaded) if (idx // 2) in temporal_indices]
    embeddings_loaded = torch.stack(embeddings_loaded, dim=0).to("cuda")
    
    # Obtain SAM embeddings for object regions
    # object_list : (image_idx, mask_idx) -> value : obj_id
    # embeddings key : (image_idx, obj_id) -> value : embedding per object
    render_embeddings, capture_embeddings = {}, {}
    obj_pairs, false_pairs = {}, {}
    ref_time_idx = 1             
    
    if len(object_lists[0].keys()) > 0 and len(object_lists[1].keys()) > 0:
        for idx, object_list in enumerate(object_lists):
            indices = render_indices if idx % 2 == 0 else capture_indices
            
            for key, value in object_list.items(): # key : (image_idx, mask_idx) / value : obj_id
                image_idx, mask_idx = key
                obj_id = value

                embedding = torch.nn.functional.interpolate(embeddings_loaded[indices[image_idx]].unsqueeze(0), [H, W], mode='bilinear').squeeze(0)
                object_mask = object_masks[indices[image_idx]][mask_idx]
                obj_embedding = embedding[:, object_mask].mean(-1)  
                
                if idx % 2 == 0:
                    render_embeddings[(image_idx, obj_id)] = obj_embedding
                else:
                    capture_embeddings[(image_idx, obj_id)] = obj_embedding
        
        # Build cosine similarity matrices
        obj_similarity = np.zeros((max(object_lists[0].values())+1, max(object_lists[1].values())+1))
        for i, (key_i, value_i) in enumerate(render_embeddings.items()):
            for j, (key_j, value_j) in enumerate(capture_embeddings.items()):
                image_idx_i, obj_id_i = key_i
                image_idx_j, obj_id_j = key_j
                
                cos_similarity = torch.nn.functional.cosine_similarity(value_i, value_j, dim=0)
                if not compare_mean and cos_similarity > obj_similarity[obj_id_i, obj_id_j]:
                    obj_similarity[obj_id_i, obj_id_j] = cos_similarity
                elif compare_mean:
                    obj_similarity[obj_id_i, obj_id_j] += cos_similarity / len(render_indices) ** 2

        print("Object Similarity:", obj_similarity)

        if not use_hungarian:
            # flatten and sort
            flat_indices = np.dstack(np.unravel_index(np.argsort(obj_similarity.ravel())[::-1], obj_similarity.shape))[0]
            row_indices, col_indices = [], []
            # greedy match
            for row, col in flat_indices:
                if row not in row_indices and col not in col_indices:
                    row_indices.append(row)
                    col_indices.append(col)
        else:
            # Hungarian algorithm (solving for minimum cost)
            row_indices, col_indices = linear_sum_assignment(1 - obj_similarity)
        print("Assigned Pairs (row, column):", list(zip(row_indices, col_indices)))
        
        # key : (image_idx, obj_id) -> value : paired (image_idx, obj_id)
        for obj_id_i, obj_id_j in list(zip(row_indices, col_indices)):
            if obj_similarity[obj_id_i, obj_id_j] > similarity_thres:
                obj_pairs[obj_id_i] = obj_id_j
            else:
                false_pairs[obj_id_i] = obj_id_j

        print("Object Pairs (obj_id): ", obj_pairs)
        print("Non Pairs (obj_id):", false_pairs)
    
    # Make final matched object list
    # key : (image_idx, mask_idx) / value : obj_id
    final_object_list = {}
    for idx, object_list in enumerate(object_lists):
        indices = render_indices if idx % 2 == 0 else capture_indices
        
        for key, value in object_list.copy().items(): # key : (image_idx, mask_idx) / value : obj_id
            image_idx, mask_idx = key
            obj_id = value

            if (indices[image_idx], mask_idx) in final_object_list.keys():
                continue
            
            if idx != ref_time_idx and obj_id in obj_pairs.keys():
                new_obj_id = obj_pairs[obj_id]
                for (k, v) in object_list.items():
                    if v == new_obj_id and (indices[k[0]], k[1]) not in final_object_list.keys():
                        final_object_list[(indices[k[0]], k[1])] = obj_id
                        # object_list[k] = obj_id

            elif idx != ref_time_idx and obj_id in false_pairs.keys():
                # Unique object id for false pairs
                if len(final_object_list) > 0:
                    max_final_object_list = np.array([*final_object_list.values()]).max()
                else:
                    max_final_object_list = -1
                new_obj_id = max(np.array([*object_lists[0].values()]).max(), np.array([*object_lists[1].values()]).max(), max_final_object_list) + 1
                for (k, v) in object_list.items():
                    if v == obj_id and (indices[k[0]], k[1]) not in final_object_list.keys():
                        final_object_list[(indices[k[0]], k[1])] = new_obj_id
                        # object_list[k] = new_obj_id
                continue
            elif idx != ref_time_idx:
                new_obj_id = obj_id
            else:
                if obj_id in obj_pairs.values():
                    new_obj_id = obj_id
                else:
                    if len(final_object_list) > 0:
                        new_obj_id = np.array([*final_object_list.values()]).max()+1
                    else:
                        new_obj_id = 0
                    for (k, v) in object_list.items():
                        if v == obj_id and (indices[k[0]], k[1]) not in final_object_list.keys():
                            final_object_list[(indices[k[0]], k[1])] = new_obj_id
                    continue

            final_object_list[(indices[image_idx], mask_idx)] = new_obj_id

    final_object_list = remap_object_indices(final_object_list)
    print("Object List (image_idx, mask_idx) -> obj_idx :", final_object_list)
    
    colored_objects = np.zeros((len(object_masks), H, W, 3), dtype=np.uint8)
    color_infos = ColorInfo()

    instance_dir = os.path.join(output_dir, "instances")
    os.makedirs(os.path.join(instance_dir, "capture"), exist_ok=True)
    os.makedirs(os.path.join(instance_dir, "renders"), exist_ok=True)

    for i, object_mask in enumerate(object_masks):
        objects = []
        filename = filenames[int(i/2)].split('.')[0] + ".npy"
        
        if i in render_indices:
            object_mask_path = os.path.join(instance_dir, "renders", filename)
        else:
            object_mask_path = os.path.join(instance_dir, "capture", filename)

        for j in range(object_mask.shape[0]):
            obj = np.zeros((H, W), int)
            if (i,j) not in final_object_list.keys():
                continue
            obj_idx = final_object_list[(i, j)]
            color = color_infos.get_color(obj_idx)
            colored_objects[i][object_mask[j] > 0] = color  

            obj[object_mask[j] > 0] = obj_idx+1
            objects.append(obj)
        np.save(object_mask_path, objects)

    print("Saved Instance Masks!")
    ## Visualize SAM mask colors ##
    plot_object_masks(colored_objects, output_dir, timestep=timestep)
    if 'embedding' in locals():
        del embedding
        del render_embeddings
        del capture_embeddings
    torch.cuda.empty_cache()

    return final_object_list


def multi_sequence_intra_matching(dataset, object_masks, num_match_thres, min_conf_thr, filter_consistent):
    source_path = Path(dataset.source_path)
    scene_name = source_path.parent.stem if str(source_path).endswith("hloc") else source_path.stem
    output_dir = os.path.join("output", scene_name)
    
    capture_path = os.path.join(output_dir, "change", "capture")
    render_path = os.path.join(output_dir, "change", "renders")
    filenames = sorted(os.listdir(capture_path))

    captures, renderings, imagefiles, image_batch = [], [], [], []
    for idx, filename in enumerate(filenames):
        renderings.append(np.array(Image.open(os.path.join(render_path, filename))))
        captures.append(np.array(Image.open(os.path.join(capture_path, filename))))
        
        image_batch.append(np.array(Image.open(os.path.join(render_path, filename)).convert("RGB")))
        image_batch.append(np.array(Image.open(os.path.join(capture_path, filename)).convert("RGB")))
                            
        imagefiles.append(os.path.join(render_path, filename))
        imagefiles.append(os.path.join(capture_path, filename))
    
    render_indices = [idx for idx in range(len(imagefiles)) if idx % 2 == 0]
    capture_indices = [idx for idx in range(len(imagefiles)) if idx % 2 != 0]
    H, W = image_batch[0].shape[0], image_batch[0].shape[1]
    h, w = get_h_w(H, W)
    
    weights_path = 'naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric'
    model = AsymmetricMASt3R.from_pretrained(weights_path).cuda()
    subsample=8
    
    object_lists = []
    matches_all = []

    input_indices = render_indices
    inputfiles = [imagefiles[idx] for idx in input_indices]
    input_object_masks =  [object_masks[idx] for idx in input_indices]

    cache_dir = os.path.join(output_dir, "mast3r", "cache", "0_all", "renders")
    os.makedirs(cache_dir, exist_ok=True)
    
    imgs = load_images(inputfiles, size=512, verbose=False)
    scene_graph = '-'.join(["complete"])
    pairs = make_pairs(imgs, scene_graph=scene_graph, prefilter=None, symmetrize=True)
    # Convert pair naming convention from dust3r to mast3r
    pairs_in = convert_dust3r_pairs_naming(inputfiles, pairs)
    # forward pass
    res_paths, recon_cache_dir = forward_mast3r(pairs_in, model,
                                    cache_path=cache_dir, subsample=subsample,
                                    desc_conf="desc_conf", device="cuda")

    tmp_pairs = {}
    for img1, img2 in pairs_in:
        idx1 = hash_md5(img1['instance'])
        idx2 = hash_md5(img2['instance'])

        path1 = recon_cache_dir + f'/forward/{idx1}/{idx2}.pth'
        path2 = recon_cache_dir + f'/forward/{idx2}/{idx1}.pth'
                
        path_corres = recon_cache_dir + f'/corres_conf=desc_conf_subsample={subsample}/{idx1}-{idx2}.pth'
        tmp_pairs[img1['instance'], img2['instance']] = (path1, path2), path_corres

    matches, object_list = track_mast3r_matches(inputfiles, input_object_masks, tmp_pairs, min_conf_thr, num_match_thres, H, W, filter_consistent=False, symmetrize=True)

    del model
    torch.cuda.empty_cache()

    return object_list


def multi_sequence_inter_matching(object_lists_per_timestep, fixed_object_list, embeddings_loaded, object_masks, filenames, similarity_thres, output_dir, compare_mean=False, use_hungarian=False):
    final_object_list = fixed_object_list.copy()
    render_indices = [idx for idx in range(len(object_masks)) if idx % 2 == 0]
    capture_indices = [idx for idx in range(len(object_masks)) if idx % 2 != 0]

    H, W = object_masks[0].shape[-2:]

    # Pair timesteps
    num_timesteps = len(object_lists_per_timestep)
    timestep_pairs = [(i, j) for i, j in combinations(range(num_timesteps), 2)]

    for ti, tj in timestep_pairs:
        obj_embeddings_i, obj_embeddings_j = {}, {}
        obj_id_pairs, false_id_pairs = {}, {}

        if len(object_lists_per_timestep[ti].keys()) > 0 and len(object_lists_per_timestep[tj].keys()) > 0:
            for idx, object_list in zip([ti,tj], [object_lists_per_timestep[ti], object_lists_per_timestep[tj]]):
                for key, value in object_list.items(): # key : (image_idx, mask_idx) / value : obj_id
                    image_idx, mask_idx = key
                    obj_id = value
                    
                    embedding = torch.nn.functional.interpolate(embeddings_loaded[image_idx].unsqueeze(0), [H, W], mode='bilinear').squeeze(0)
                    object_mask = object_masks[image_idx][mask_idx]
                    
                    if idx == ti:
                        obj_embeddings_i[(image_idx, obj_id)] = embedding[:, object_mask].mean(-1)  
                    else:
                        obj_embeddings_j[(image_idx, obj_id)] = embedding[:, object_mask].mean(-1)        
            
            # Build cosine similarity matrices
            obj2idx_ti = {obj_id: idx for idx, obj_id in enumerate(set(object_lists_per_timestep[ti].values()))}
            idx2obj_ti = {idx: obj_id for idx, obj_id in enumerate(set(object_lists_per_timestep[ti].values()))}
            obj2idx_tj = {obj_id: idx for idx, obj_id in enumerate(set(object_lists_per_timestep[tj].values()))}
            idx2obj_tj = {idx: obj_id for idx, obj_id in enumerate(set(object_lists_per_timestep[tj].values()))}

            obj_similarity = np.zeros((len(obj2idx_ti), len(obj2idx_tj)))
            for i,(key_i, value_i) in enumerate(obj_embeddings_i.items()):
                for j, (key_j, value_j) in enumerate(obj_embeddings_j.items()):
                    image_idx_i, obj_id_i = key_i
                    image_idx_j, obj_id_j = key_j
                    cos_similarity = torch.nn.functional.cosine_similarity(value_i, value_j, dim=0)

                    if not compare_mean and cos_similarity > obj_similarity[obj2idx_ti[obj_id_i], obj2idx_tj[obj_id_j]]:
                        obj_similarity[obj2idx_ti[obj_id_i], obj2idx_tj[obj_id_j]] = cos_similarity 
                    elif compare_mean:
                        obj_similarity[obj2idx_ti[obj_id_i], obj2idx_tj[obj_id_j]] += cos_similarity * np.prod(obj_similarity.shape) / (len(obj_embeddings_i.items())* len(obj_embeddings_j.items()))
            
            print("Object Similarity:", obj_similarity)

            if not use_hungarian:
                # flatten and sort
                flat_indices = np.dstack(np.unravel_index(np.argsort(obj_similarity.ravel())[::-1], obj_similarity.shape))[0]
                row_indices, col_indices = [], []
                # greedy match
                for row, col in flat_indices:
                    if row not in row_indices and col not in col_indices:
                        row_indices.append(row)
                        col_indices.append(col)
            else:
                # Hungarian algorithm (solving for minimum cost)
                row_indices, col_indices = linear_sum_assignment(1 - obj_similarity)
            print("Assigned Pairs (row, column):", list(zip(row_indices, col_indices)))

            # key : (image_idx, obj_id) -> value : paired (image_idx, obj_id)
            for row, col in list(zip(row_indices, col_indices)):
                if obj_similarity[row, col] > similarity_thres:
                    obj_id_pairs[idx2obj_ti[row]] = idx2obj_tj[col]
                else:
                    false_id_pairs[idx2obj_ti[row]] = idx2obj_tj[col]

            print("Object ID Pairs : ", obj_id_pairs)
            print("Non ID Pairs :", false_id_pairs)
        
        obj_pairs = {}
        for key1, value1 in object_lists_per_timestep[ti].items(): 
            for key2, value2 in object_lists_per_timestep[tj].items():
                if key1 in obj_pairs.keys() or key2 in obj_pairs.values():
                    continue
                if value1 in obj_id_pairs.keys() and value2 == obj_id_pairs[value1]:
                    obj_pairs[key1] = key2

        # print("Object Pairs (image_idx, mask_idx): ", obj_pairs)
        
        # Add pairs to final_object_list
        for idx, object_lists in zip([ti,tj], [object_lists_per_timestep[ti], object_lists_per_timestep[tj]]):
            selected_values = []
            value_mapping = {}

            for key, value in object_lists.items(): # key : (image_idx, mask_idx) / value : obj_id                
                if value not in [*value_mapping.keys()]:
                    value_mapping[value] = max(final_object_list.values())+1 if len(final_object_list) > 0 else 0

                if key not in [*final_object_list.keys()]:
                    final_object_list[key] = value_mapping[value]
                    if key in [*obj_pairs.keys()]:
                        final_object_list[obj_pairs[key]] = value_mapping[value]
                    selected_values.append(value)
                    continue

                # This might be needed for longer sequences
                if key in [*final_object_list.keys()] and key in [*obj_pairs.keys()]:
                    if key in fixed_object_list.keys() and obj_pairs[key] not in fixed_object_list.keys():
                        final_object_list[obj_pairs[key]] = final_object_list[key]
                        fixed_object_list[obj_pairs[key]] = fixed_object_list[key]
                    elif obj_pairs[key] in fixed_object_list.keys() and key not in fixed_object_list.keys():
                        final_object_list[key] = final_object_list[obj_pairs[key]]
                        fixed_object_list[key] = fixed_object_list[obj_pairs[key]]
                    elif key in fixed_object_list.keys() and obj_pairs[key] in fixed_object_list.keys():
                        pass
                    elif obj_pairs[key] in [*final_object_list.keys()]:
                        if final_object_list[key] == final_object_list[obj_pairs[key]]:
                            pass
                        else:
                            final_object_list[key] = final_object_list[obj_pairs[key]]
                    else:
                        value_mapping[value] = final_object_list[key]
                        final_object_list[obj_pairs[key]] = value_mapping[value]
                        selected_values.append(value)
                    continue

                if value in selected_values:
                    final_object_list[key] = value_mapping[value]
                    if key in [*obj_pairs.keys()]:
                        final_object_list[obj_pairs[key]] = value_mapping[value]
                    continue
        
        final_object_list = remap_object_indices(final_object_list)
        
        ## Visualize instances
        # print("Multiple Timestep Object List (image_idx, mask_idx) -> obj_idx :", final_object_list)

        colored_objects = np.zeros((len(object_masks), H, W, 3), dtype=np.uint8)
        color_infos = ColorInfo()

        instance_dir = os.path.join(output_dir, "instances")
        os.makedirs(os.path.join(instance_dir, "capture"), exist_ok=True)
        os.makedirs(os.path.join(instance_dir, "renders"), exist_ok=True)
        
        for i, object_mask in enumerate(object_masks):
            objects = []
            filename = filenames[int(i/2)].split('.')[0] + ".npy"
            
            if i in render_indices:
                object_mask_path = os.path.join(instance_dir, "renders", filename)
            else:
                object_mask_path = os.path.join(instance_dir, "capture", filename)

            for j in range(object_mask.shape[0]):
                obj = np.zeros((H, W), int)
                if (i,j) not in final_object_list.keys():
                    continue
                obj_idx = final_object_list[(i, j)]
                color = color_infos.get_color(obj_idx)
                colored_objects[i][object_mask[j] > 0] = color  

                obj[object_mask[j] > 0] = obj_idx+1
                objects.append(obj)
            np.save(object_mask_path, objects)

        # print("Saved Instance Masks!")
        ## Visualize SAM mask colors ##
        plot_object_masks(colored_objects, output_dir)

        if 'embedding' in locals():
            del embedding    
            del obj_embeddings_i
            del obj_embeddings_j
        torch.cuda.empty_cache()

    return final_object_list