import sys
sys.path.append('./gaussian-splatting')
sys.path.append('./submodules/GeSCF')

import cv2
# cv2.namedWindow("Test window", cv2.WINDOW_GUI_NORMAL)
# cv2.waitKey(1000)
# cv2.destroyAllWindows()
import torch
import os

from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args        
from gescf.framework_modified import GeSCF
from segment_anything import SamPredictor

from pathlib import Path
from PIL import Image
import numpy as np
from enum import Enum
from collections import namedtuple
import h5py
from skimage.morphology import remove_small_objects, reconstruction
from scipy.ndimage import label, binary_fill_holes
import matplotlib.pyplot as plt
from src.utils.visualization_utils import plot_sam_embeddings, compute_pca_image, show_change_masks

class ClickType(Enum):
    Background = 0
    Foreground = 1

class Point:
    def __init__(self, x, y, instance_id, type):
        self.x = x
        self.y = y
        self.instance_id = instance_id
        self.type = type

    def __repr__(self):
        return f'x: {self.x}, y: {self.y}, instance_id: {self.instance_id}, type: {self.type}'

def instance_id_to_color(id):
    rgb = [0, 0, 0]
    color_idx = 0
    while id > 0:
        rgb[color_idx] = (id % 2) * 255
        id = int(id / 2)
        color_idx += 1 

    return tuple(rgb)

def run_predictor(predictor, points):
    if not points:
        return []
    masks = []
    input_instance_ids = np.unique(np.array([p.instance_id for p in points]))
    for instance_id in input_instance_ids:
        input_point = np.array([[p.x, p.y] for p in points if p.instance_id == instance_id]).astype(np.float32)
        input_label = np.array([p.type.value for p in points if p.instance_id == instance_id]).astype(int)
        
        mask, _, _ = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=False,
        )
        mask = mask[None, 0, Ellipsis]
        masks.append(mask)

    masks = np.concatenate(masks, axis=0)
    return masks

def save_sam_outputs(object_masks, sam_embeddings, sam_output_path):
    # Save to HDF5
    with h5py.File(sam_output_path, "w") as h5_file:
        # Save object masks
        object_group = h5_file.create_group("object_masks")
        for i, mask in enumerate(object_masks):
            object_group.create_dataset(f"mask_{i:02}", data=mask, compression="gzip")
        # Save embedding
        h5_file.create_dataset("embeddings", data=sam_embeddings, compression="gzip")

def mask_selection(image, masks, predictor):
    """
    Let user click on masks to remove them.
    Args:
        masks: (N, H, W) binary mask array
    Returns:
        masks: filtered list after removing selected masks
    """
    original_masks = masks.copy()
    display_img = image[:, :, [2, 1, 0]] # np.zeros((masks.shape[1], masks.shape[2], 3), dtype=np.uint8)
    # Assign colors to each mask
    colors = [(np.random.randint(50, 255), np.random.randint(50, 255), np.random.randint(50, 255)) for _ in masks]
    for idx, mask in enumerate(masks):
        color = colors[idx]
        display_img[mask] = color

    append_indices = set()
    remove_indices = set()
    points = []
    instance_id = 1
    new_object = True

    def click_event(event, x, y, flags, param):
        nonlocal masks, instance_id, append_indices, remove_indices, new_object, points
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append(Point(x=x, y=y, instance_id=instance_id, type=ClickType.Foreground))
            new_masks = run_predictor(predictor, points)
            if new_object:
                if len(masks) > 0:
                    masks = np.concatenate([masks, new_masks], axis=0)
                else:
                    masks = new_masks
                append_indices.add(len(masks)-1)
                colors.append((np.random.randint(50, 255), np.random.randint(50, 255), np.random.randint(50, 255)))
                new_object = False  
            else:
                masks[-1] = masks[-1] + new_masks[0]

            print(f"Selected mask {len(masks)-1} for append")
            print(append_indices, remove_indices)

        elif event == cv2.EVENT_RBUTTONUP:
            if new_object:
                print("First select foreground points to create a new object")
            else:
                points.append(Point(x=x, y=y, instance_id=instance_id, type=ClickType.Background))
                new_masks = run_predictor(predictor, points)
                masks[-1] = new_masks

        elif event == cv2.EVENT_MBUTTONDOWN:
            for idx, mask in enumerate(masks):
                if mask[y, x]:
                    if idx in remove_indices:
                        remove_indices.remove(idx)
                        # if idx in append_indices:
                        #     append_indices.add(idx)
                        print(f"Restored mask {idx}")
                    else:
                        if idx in append_indices:
                            print(idx, append_indices, remove_indices, idx not in append_indices)
                            append_indices.remove(idx)
                            masks = np.delete(masks, idx, axis=0)
                            # colors.pop(idx)
                            new_object = True
                            points = []
                        elif idx not in append_indices:
                            print(idx, append_indices, remove_indices, idx not in append_indices)
                            remove_indices.add(idx)
                        print(f"Selected mask {idx} for removal")
            append_indices = set([len(original_masks) + i for i, indices in enumerate(append_indices)])
            print("append: ", append_indices, "remove: ",remove_indices)
        
    temp_img = display_img.copy()
    cv2.namedWindow("Select Masks to Remove", cv2.WINDOW_GUI_NORMAL)
    cv2.setMouseCallback("Select Masks to Remove", click_event)

    while True:
        img_show = (0.7*temp_img).copy().astype(np.uint8)

        for point in points:
            color = (255, 0, 0) if point.type == ClickType.Foreground else (0, 0, 255)
            img_show = cv2.circle(img_show, (point.x, point.y), radius=5, color=color, thickness=-1)

        for idx in remove_indices:
            img_show[masks[idx]] = (50, 50, 50)  # dim selected masks
        for idx in append_indices:
            img_show[masks[idx]] = colors[idx - len(original_masks)]
        cv2.imshow("Select Masks to Remove", img_show)

        key = cv2.waitKey(20)

        if key == ord('z'):
            masks = original_masks
            remove_indices = set()
            append_indices = set()
            points = []
            instance_id = 1
            new_object = True
            print("Reset masks")
        elif key == ord('r'):
            masks = original_masks
            remove_indices = set([i for i in range(len(original_masks))])
            append_indices = set()
            points = []
            instance_id = 1
            new_object = True
            print("Remove all masks")
        elif ord('1') <= key <= ord('9'):
            points = []
            instance_id = int(chr(key))
            new_object = True
        elif key==13:
            print("Next image")
            break
        
    cv2.destroyAllWindows()
    if len(remove_indices) == len(original_masks) and len(append_indices) == 0:
        masks = []

    # Keep only unselected masks
    final_masks = [masks[i] for i in range(len(masks)) if i not in remove_indices]
    return np.stack(final_masks, axis=0) if final_masks else np.zeros_like(masks[:1])

def filter_objects_by_size(label_image, min_size=0, max_size=-1, connectivity=1):
    small_removed = remove_small_objects(label_image, min_size, connectivity)
    if max_size > 0:
        mid_removed = remove_small_objects(small_removed, max_size, connectivity)
        large_removed = small_removed & (~mid_removed)
        return large_removed
    else:
        return small_removed
    
def change_detection(dataset : ModelParams, min_size: int, max_size: int, connectivity: int, split_thres: float, kernel_size:int, alpha_t: float, cosine_thr: float, ssim_ratio: float, kernel_ratio: float, manual_selection: bool):
    with torch.no_grad(): 
        source_path = Path(dataset.source_path)
        scene_name = source_path.parent.stem if str(source_path).endswith("hloc") else source_path.stem
        output_dir = os.path.join("output", scene_name)
    
        capture_path = os.path.join(output_dir, "change", "capture")
        render_path = os.path.join(output_dir, "change", "renders")
        filenames = sorted(os.listdir(capture_path))

        query_list, image_batch, imagefiles = [], [], []
        for filename in filenames:
            image_batch.append(np.array(Image.open(os.path.join(render_path, filename)).convert("RGB")))
            image_batch.append(np.array(Image.open(os.path.join(capture_path, filename)).convert("RGB")))
                               
            query_list.append(os.path.join(os.path.basename(render_path), filename))
            query_list.append(os.path.join(os.path.basename(capture_path), filename))

            imagefiles.append(os.path.join(render_path, filename))
            imagefiles.append(os.path.join(capture_path, filename))

        render_indices = [idx for idx in range(len(query_list)) if idx % 2 == 0]
        capture_indices = [idx for idx in range(len(query_list)) if idx % 2 != 0]

        H, W = image_batch[0].shape[0], image_batch[0].shape[1]
    
        # GeSCF model        
        # Tune thres, ssim_ratio, kernel_ratio
        model = GeSCF(img_size=(W,H), alpha_t=alpha_t, cosine_thr=cosine_thr, ssim_ratio=ssim_ratio, kernel_ratio=kernel_ratio)
        if manual_selection:
            predictor = SamPredictor(model.sam_backbone)

        # Inference
        change_masks, embeddings = [], []
        for render_idx, capture_idx in zip(render_indices, capture_indices):
            print(f"Detecting changes for {int(render_idx/2)+1}th set of images ...")
            change_mask_t0, embed_t0, _ = model(imagefiles[render_idx], imagefiles[capture_idx])   
            resized_mask_t0 = cv2.resize(change_mask_t0.astype(np.uint8), (W, H), interpolation=cv2.INTER_LINEAR)
            resized_mask_t0 = (resized_mask_t0 > 0.5).astype(bool)
            change_masks.append(resized_mask_t0)
            embeddings.append(embed_t0)
            
            # in reverse order
            change_mask_t1, embed_t1, _ = model(imagefiles[capture_idx], imagefiles[render_idx])   
            resized_mask_t1 = cv2.resize(change_mask_t1.astype(np.uint8), (W, H), interpolation=cv2.INTER_LINEAR)
            resized_mask_t1 = (resized_mask_t1 > 0.5).astype(bool)
            change_masks.append(resized_mask_t1)
            embeddings.append(embed_t1)

        embeddings = torch.cat(embeddings, dim=0)
        
        # show_change_masks(image_batch, change_masks, scene_name, render_indices, capture_indices)

        # Cluster and separate change masks into objects
        # https://opencv-python.readthedocs.io/en/latest/doc/27.imageWaterShed/imageWaterShed.html
        object_masks, cleaned_masks = [], []
        skipped_idx = []
        if manual_selection:
            print("Click on masks to toggle removal.")
            print("Press 1-9 to define instances.")
            print("Press z to Reset masks.")
            print("Press r to Remove all masks.")
            print("Press enter to confirm.")
        for i, change_mask in enumerate(change_masks):
            # Remove change masks that are smaller than threshold size
            change_mask = filter_objects_by_size(change_mask, min_size, max_size)  

            # Use distance transform + watershed if objects are touching
            if split_thres > 0:
                distance = cv2.distanceTransform(change_mask.astype(np.uint8), distanceType=cv2.DIST_L2, maskSize=5)
                _, sure_fg = cv2.threshold(distance, split_thres, 1, 0)
                sure_fg = np.uint8(sure_fg)
                unknown = cv2.subtract(change_mask.astype(np.uint8), sure_fg)
            else:
                sure_fg = change_mask.astype(np.uint8)

            if connectivity > 1 and len(change_mask) > 0:
                connect_kernel = np.ones((connectivity, connectivity), np.uint8)
                sure_fg = cv2.dilate(sure_fg.astype(float), connect_kernel, iterations=1).astype(np.uint8)

            # Mask labeling
            num_features, labeled_array = cv2.connectedComponents(sure_fg)
            # labeled_array = labeled_array + 1  # So background is not 0
            if split_thres > 0:
                labeled_array[unknown == 1] = 0

            print(f"Mask index: {i}, num_masks: {num_features-1}")
            object_mask = []
            for j in range(1, num_features):
                labeled_mask = (labeled_array == j).astype(np.uint8)
                object_mask.append(labeled_mask.astype(bool))

            if not len(object_mask) == 0:
                object_mask = np.stack(object_mask, axis=0)
            else:
                object_mask = np.zeros((1, H, W), dtype=bool)

            if manual_selection:
                # open gui and select the object to remove
                print(f"manually selecting {i}th mask..")
                predictor.set_image(image_batch[i])
                object_mask = mask_selection(image_batch[i], object_mask, predictor)
                print(f"selecting {len(object_mask)} masks!")

            if kernel_size > 1 and len(object_mask) > 0:
                kernel = np.ones((kernel_size, kernel_size), np.uint8)
                object_mask = [cv2.dilate(mask.astype(float), kernel, iterations=1).astype(bool) for mask in object_mask]

            if len(object_mask) == 0:
                skipped_idx.append(i)
                object_masks.append(None)
                cleaned_masks.append(None)
            else:
                object_mask = np.stack(object_mask, axis=0)
                object_masks.append(object_mask)     
                cleaned_masks.append(object_mask.sum(0).astype(bool))

        for idx in skipped_idx:
            object_mask = np.zeros((1,H,W), dtype=bool)
            cleaned_mask = object_mask.sum(0).astype(bool)
            object_masks.pop(idx)
            cleaned_masks.pop(idx)
            object_masks.insert(idx, object_mask)
            cleaned_masks.insert(idx, cleaned_mask)

        show_change_masks(image_batch, cleaned_masks, scene_name, render_indices, capture_indices)

        os.makedirs(os.path.join(output_dir , "sam"), exist_ok=True)
        sam_output_path = os.path.join(output_dir, "sam", "sam_outputs.h5")

        save_sam_outputs(object_masks, embeddings.detach().cpu().numpy(), sam_output_path)
        print("Done Saving SAM Embeddings")

        del model

        return object_masks, embeddings

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--min_size", default=500, type=int)
    parser.add_argument("--max_size", default=-1, type=int)
    parser.add_argument("--connectivity", default=1, type=int)
    parser.add_argument("--split_thres", default=0.00, type=float) 
    parser.add_argument("--kernel_size", default=1, type=int)
    parser.add_argument("--alpha_t", default=0.65, type=float) 
    parser.add_argument("--cosine_thr", default=0.88, type=float) 
    parser.add_argument("--ssim_ratio", default=0.35, type=float) 
    parser.add_argument("--kernel_ratio", default=0.03, type=float) 
    parser.add_argument("--manual_selection", action="store_true")
    
    args = get_combined_args(parser)
    print("Detecting changes in " + args.model_path)

    object_masks, embeddings = change_detection(model.extract(args), args.min_size, args.max_size, args.connectivity, args.split_thres, args.kernel_size, args.alpha_t, args.cosine_thr, args.ssim_ratio, args.kernel_ratio, args.manual_selection)
