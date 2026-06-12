import sys
sys.path.append('./gaussian-splatting')
sys.path.append('./submodules/mast3r')

import torch
import os
import matplotlib.pyplot as plt

from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from pathlib import Path

import numpy as np
import h5py

from collections import defaultdict
            
import shutil
from src.utils.localization_utils import extract_sort_key
from src.utils.matching_utils import intra_instance_matching, inter_instance_matching, multi_sequence_intra_matching, multi_sequence_inter_matching

def load_sam_outputs(dataset : ModelParams):
    source_path = Path(dataset.source_path)
    scene = source_path.parent.stem if str(source_path).endswith("hloc") else source_path.stem
    output_dir = os.path.join("output", scene)
    sam_output_path = os.path.join(output_dir, "sam", "sam_outputs.h5")

    with h5py.File(sam_output_path, "r") as h5_file:
        # Load object masks
        object_masks_loaded = [np.array(h5_file["object_masks"][key]) for key in h5_file["object_masks"].keys()]
        # Load embedding
        embeddings_loaded = torch.tensor(h5_file["embeddings"][:]).to("cuda")

    return object_masks_loaded, embeddings_loaded
    
def instance_matching(dataset: ModelParams, args):
    optim_level = args.optim_level
    matching_conf_thres = args.matching_conf_thres
    min_conf_thres = args.min_conf_thres
    fix_separated = args.fix_separated
    filter_consistent = args.filter_consistent
    num_match_thres = args.num_match_thres
    single_similarity_thres = args.single_similarity_thres
    multi_similarity_thres = args.multi_similarity_thres
    use_hungarian = args.use_hungarian
    compare_mean = args.compare_mean

    source_path = Path(dataset.source_path) # ends with hloc
    scene_name = source_path.parent.stem if str(source_path).endswith("hloc") else source_path.stem
    output_dir = os.path.join("output", scene_name)
    if os.path.exists(os.path.join(output_dir, "mast3r")):
        shutil.rmtree(os.path.join(output_dir, "mast3r"))

    capture_path = os.path.join(output_dir, "change", "capture")
    render_path = os.path.join(output_dir, "change", "renders")
    filenames = sorted(os.listdir(capture_path))

    query_path = Path('output') / scene_name / "hloc" / "changes.txt"
    with open(query_path, 'r') as file:            
        image_filenames = file.read().strip().split()
    query_list = sorted(image_filenames, key = extract_sort_key) if image_filenames[0].startswith("IMG_") else sorted(image_filenames)
    temporal_inputs = [path.split('/')[0] for path in query_list]
    
    # Load change mask and embeddings
    object_masks, embeddings_loaded = load_sam_outputs(dataset) 

    object_lists_per_timestep = []
    temporal_indices = {}
    final_object_list = {}

    for timestep, temporal_input in enumerate(sorted(set(temporal_inputs))):
        ## 2D instance matching
        temporal_indices[timestep] = [idx for idx, name in enumerate(temporal_inputs) if name.startswith(temporal_input)]
        intra_object_lists, mast3r_outputs = intra_instance_matching(dataset, object_masks, optim_level, matching_conf_thres, min_conf_thres, filter_consistent, num_match_thres, timestep=timestep, temporal_indices=temporal_indices[timestep])
        object_lists = inter_instance_matching(dataset, intra_object_lists, object_masks, embeddings_loaded, filter_consistent, single_similarity_thres, timestep=timestep, temporal_indices=temporal_indices[timestep], compare_mean=compare_mean, use_hungarian=use_hungarian)
        
        key_mapping = {local_idx: global_idx for local_idx, global_idx in enumerate(range(2*temporal_indices[timestep][0], 2*(temporal_indices[timestep][-1]+1)))}
        object_lists = {(key_mapping[key[0]], key[1]):value for key, value in object_lists.items()}

        object_lists_per_timestep.append(object_lists)
    
    if len(set(temporal_inputs)) > 1:
        # Mast3r matching for initial state objects
        # Initialize final_object_list
        # key : (image_idx, mask_idx) / value : obj_idx
        initial_object_list = multi_sequence_intra_matching(dataset, object_masks, num_match_thres, min_conf_thres, filter_consistent=False)
        inconsistent_keys = []
        if filter_consistent:
            for timestep, temporal_idx in temporal_indices.items():
                counts = defaultdict(int)
                for key, obj_idx in initial_object_list.items():
                    image_idx, mask_idx = key
                    if (2*image_idx, mask_idx) not in object_lists_per_timestep[timestep].keys():
                        continue
                    if image_idx in temporal_idx:
                        counts[obj_idx] += 1

                for key, obj_idx in initial_object_list.items():
                    if key[0] not in temporal_idx:
                        continue
                    if (counts[obj_idx] == len(temporal_idx)) or counts[obj_idx] == 0:
                        pass
                    else:
                        inconsistent_keys.append(key)

        initial_object_list = {(key[0]*2, key[1]):value for key, value in initial_object_list.items() if key not in inconsistent_keys}

        print("Object lists after multi sequence intra matching:", initial_object_list)
        fixed_object_list = {}
        for idx, object_lists in enumerate(object_lists_per_timestep):
            selected_keys, selected_values = [], []
            value_mapping = {}
            for key, value in object_lists.items():
                # Add the object list to the final object list
                if key in [*initial_object_list.keys()]:
                    value_mapping[value] = initial_object_list[key]
                    fixed_object_list[key] = value_mapping[value] 
                    selected_keys.append(key)
                    selected_values.append(value)
                    continue

                if value in selected_values:
                    fixed_object_list[key] = value_mapping[value]
                    selected_keys.append(key)
                    selected_values.append(value)
                    continue

            # Remove selected objects from object_lists_per_timestep
            if fix_separated:
                for key in selected_keys:
                    del object_lists[key]
            else:
                for key in fixed_object_list.keys():
                    if key in object_lists.keys() and key[0] % 2 == 0:
                        del object_lists[key]

        # # TODO: Compare to fixed object list first
        # render_indices = [idx for idx in range(len(object_masks)) if idx % 2 == 0]
        # capture_indices = [idx for idx in range(len(object_masks)) if idx % 2 != 0]
        # H, W = object_masks[0].shape[-2:]
        # fixed_obj_idx = np.unique(np.array([*fixed_object_list.values()]))

        # for key, value in enumerate(fixed_object_list):
        # for idx, object_list in enumerate(object_lists_per_timestep):
        #     if len(object_list) == 0:
        #         continue

        #     for key, value in object_list.items():
        #         image_idx, mask_idx = key
        #         obj_id = value
        #         embedding = torch.nn.functional.interpolate(embeddings_loaded[image_idx].unsqueeze(0), [H, W], mode='bilinear').squeeze(0)
        #         object_mask = object_masks[image_idx][mask_idx]

        #         if idx == ti:
        #             obj_embeddings_i[(image_idx, obj_id)] = embedding[:, object_mask].mean(-1)  
        #         else:
        #             obj_embeddings_j[(image_idx, obj_id)] = embedding[:, object_mask].mean(-1)   
        #     # target_obj_idx = np.unique(np.array([*object_list.values()]))
        #     # compare it to fixed_obj_idx
        #     # remove object lists[key]

        # Semantic matching for different timesteps
        final_object_list = multi_sequence_inter_matching(object_lists_per_timestep, fixed_object_list, embeddings_loaded, object_masks, filenames, multi_similarity_thres, output_dir, compare_mean=compare_mean, use_hungarian=use_hungarian)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--single_similarity_thres", default=0.8, type=float)
    parser.add_argument("--multi_similarity_thres", default=0.8, type=float)
    parser.add_argument("--optim_level", default="refine", type=str)
    parser.add_argument("--num_match_thres", default=10, type=int)
    parser.add_argument("--matching_conf_thres", default=2.0, type=float)
    parser.add_argument("--min_conf_thres", default=0, type=float)
    parser.add_argument("--filter_consistent", action="store_true")
    parser.add_argument("--fix_separated", action="store_true")
    parser.add_argument("--use_hungarian", action="store_true")
    parser.add_argument("--compare_mean", action="store_true")

    args = get_combined_args(parser)
    print("Detecting changes in " + args.model_path)
    
    instance_matching(model.extract(args), args) 

