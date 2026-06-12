import sys
import torch
import numpy as np

from PIL import Image
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from src.extractor_dino import ViTExtractor
from src.utils.visualization_utils import compute_pca_image
from src.utils.matching_utils import get_h_w

sys.path.append('./submodules/mast3r')
from mast3r.model import AsymmetricMASt3R
from mast3r.fast_nn import fast_reciprocal_NNs

from dust3r.inference import inference
from dust3r.utils.image import load_images
from dust3r.image_pairs import make_pairs

class Descriptors:
    def __init__(self):
        self.pairs = None
        self.desc1 = None
        self.desc2 = None
        self.desc_conf1 = None
        self.desc_conf2 = None
        self.crop_ranges = None
        self.dino_descriptors = None
        
    # def set_descriptors(self, obj_label, pairs, desc1, desc2, desc_conf1, desc_conf2):
    #     pair_indices = []
    #     for pair in pairs:
    #         pair_indices.append((int(pair[0]['instance']), int(pair[1]['instance'])))
    #     self.pairs[obj_label] = pair_indices
    #     self.desc1[obj_label] = desc1
    #     self.desc2[obj_label] = desc2
    #     self.desc_conf1[obj_label] = desc_conf1
    #     self.desc_conf2[obj_label] = desc_conf2


    # def get_descriptors(self, obj_label):
    #     return self.pairs.get(obj_label), self.desc1.get(obj_label), self.desc2.get(obj_label), self.desc_conf1.get(obj_label), self.desc_conf2.get(obj_label)
    
    # def set_crop_ranges(self, obj_label, crop_ranges):
    #     self.crop_ranges[obj_label] = crop_ranges 

    # def get_crop_ranges(self, obj_label):
    #     return self.crop_ranges.get(obj_label)
    

    def set_descriptors(self, pairs, desc1, desc2, desc_conf1=None, desc_conf2=None):
        pair_indices = []
        for pair in pairs:
            pair_indices.append((int(pair[0]['instance']), int(pair[1]['instance'])))
        self.pairs = pair_indices
        self.desc1 = desc1
        self.desc2 = desc2
        self.desc_conf1 = desc_conf1
        self.desc_conf2 = desc_conf2

    def get_descriptors(self):
        return self.pairs, self.desc1, self.desc2, self.desc_conf1, self.desc_conf2
    
    def set_crop_ranges(self, crop_ranges):
        self.crop_ranges = crop_ranges 

    def get_crop_ranges(self):
        return self.crop_ranges
    
    def fuse_descriptors(self):
        # concat
        # self.dino
        # self.mast3r
        pass

def divisible_by_num(num, dim):
    return num * (dim // num)
        
def extract_2d_descriptors(matching_obj_labels, object_masks, imagefiles, descriptor_path, input_mode: str ="full", matcher: str ="mast3r"):
    h, w = get_h_w(object_masks[1].shape[-2], object_masks[1].shape[-1]) if len(object_masks[0]) == 0 else get_h_w(object_masks[0].shape[-2], object_masks[0].shape[-1])

    assert input_mode in ["full"], "Invalid input mode"

    if input_mode == "full":
        inputfiles = [imagefiles[idx] for idx in range(len(imagefiles))]
        imgs = load_images(inputfiles, size=w, verbose=False)        
        pairs_all = make_pairs(imgs, scene_graph='complete', prefilter=None, symmetrize=True)
        pairs = [pair for pair in pairs_all if (pair[1]['idx']-pair[0]['idx'])%2 == 0]

    descriptors = Descriptors()
    dino_extractor = ViTExtractor(model_type='dinov2_vits14', stride=14, device='cuda') # can change it to another model\
    patch_size = dino_extractor.model.patch_embed.patch_size[0]
    
    # Extract dino descriptors
    image_batch, new_height, new_width = dino_extractor.preprocess_inputfile(inputfiles)
    dino_descriptors = dino_extractor.extract_descriptors(image_batch.to('cuda'), layer=11, facet='token', bin=False)
    
    num_patches = int(new_height / patch_size), int(new_width / patch_size)
    dino_descriptors = dino_descriptors.reshape(len(image_batch), num_patches[0], num_patches[1], dino_descriptors.shape[-1])
    
    ## Memory efficient way - Slower
    # dino_descriptors = []
    # for idx, inputfile in enumerate(inputfiles):
    #     image_batch, new_height, new_width = dino_extractor.preprocess_inputfile([inputfile])
    #     dino_descriptor = dino_extractor.extract_descriptors(image_batch.to('cuda'), layer=11, facet='token', bin=False)
        
    #     num_patches = int(new_height / patch_size), int(new_width / patch_size)
    #     dino_descriptor = dino_descriptor.reshape(len(image_batch), num_patches[0], num_patches[1], dino_descriptor.shape[-1])
        
    #     dino_descriptors.append(dino_descriptor)
    # dino_descriptors = torch.cat(dino_descriptors, 0)

    # Visualize per-object DINO features
    for obj_label in matching_obj_labels:
        img_selected = [len(torch.where(object_masks[idx] == obj_label)[0]) > 0 for idx in range(len(imagefiles))]
        object_mask = [object_masks[idx][torch.where(obj_mask == obj_label)[0][0].item()] for idx, obj_mask in enumerate(object_masks) if img_selected[idx]]
        input_object_masks = [torch.nn.functional.interpolate(obj_mask[None,None], (h,w)).reshape(h,w) for obj_mask in object_mask]
        
        fig, axes = plt.subplots(2, int(len(inputfiles)/2), figsize=(12, 6))
        selected_idx = 0
        for img_idx in range(len(inputfiles)):
            i, j = img_idx % 2, img_idx // 2 
            zeros = np.zeros((h,w,3))
            if img_selected[img_idx]:
                selected_dino = dino_descriptors[selected_idx]
                interpolated_dino = torch.nn.functional.interpolate(selected_dino[None].permute(0,3,1,2), (h,w), mode='bilinear').permute(0,2,3,1).detach().cpu()
                interpolated_dino = interpolated_dino / torch.linalg.norm(interpolated_dino, dim=-1, keepdim=True)
                dino_obj = compute_pca_image(interpolated_dino[0][input_object_masks[selected_idx]>0].detach().cpu().numpy())
                zeros[input_object_masks[selected_idx]>0] = dino_obj
                selected_idx += 1
               
            axes[i,j].imshow(zeros)
    
        # plt.show()    
        plt.savefig(f"{descriptor_path}/dino_descriptors_{obj_label}.png")
        plt.close()
    
    desc1, desc2 = torch.zeros(len(pairs), num_patches[0], num_patches[1], dino_descriptors.shape[-1]), torch.zeros(len(pairs), num_patches[0], num_patches[1],dino_descriptors.shape[-1])
    for i, pair in enumerate(pairs):
        desc1[i] = dino_descriptors[pair[0]['idx']].detach().cpu()
        desc2[i] = dino_descriptors[pair[1]['idx']].detach().cpu()

    descriptors.set_descriptors(pairs, desc1, desc2, None, None)

    del dino_extractor
    del dino_descriptors
    torch.cuda.empty_cache()
    
    return descriptors

def fuse_multiview_descriptors(before_pcds, after_pcds, change_cameras, matching_obj_labels, descriptors, object_masks, src_indices, descriptor_path):
    H, W = object_masks[1].shape[-2:] if len(object_masks[0]) == 0 else object_masks[0].shape[-2:]

    pairs, desc1, desc2, _, _ = descriptors.get_descriptors()
    minimal_count = len(pairs) / 2
    
    # Generate 3D Descriptor Fields
    fused_outputs = {}

    for obj_label in matching_obj_labels:
        fused_outputs[obj_label] = {}

        has_before = obj_label in before_pcds
        has_after = obj_label in after_pcds
        assert has_before or has_after

        for idx, object_mask in enumerate(object_masks):
            if obj_label not in torch.unique(object_mask):
                if idx in src_indices:
                    has_before = False 
                else:
                    has_after = False
                break
    
        if has_before:
            pcd1 = torch.from_numpy(before_pcds[obj_label]).to(torch.float32).cuda()
            pcd1_homo = torch.cat((pcd1, torch.ones((pcd1.shape[0],1), device=pcd1.device, dtype=torch.float32)), dim=1)

            # Projection for before only
            if len(src_indices) == len(change_cameras):
                full_proj_transform_1 = torch.stack([cam.full_proj_transform for cam in change_cameras], dim=0)
            else:
                full_proj_transform_1 = torch.stack(
                    [cam.full_proj_transform for idx, cam in enumerate(change_cameras) if idx in src_indices],
                    dim=0
                )
            proj_1 = pcd1_homo @ full_proj_transform_1 
            proj_1 = proj_1[..., :3] / proj_1[..., 3:4]
            proj1_xy = proj_1[...,:2].cpu().numpy()
        
            # Validity check
            validity_1 = (proj1_xy[...,0] >= -1) & (proj1_xy[...,0] <= 1) & (proj1_xy[...,1] >= -1) & (proj1_xy[...,1] <= 1) 
            validity_1 = np.all(validity_1, 0)
            valid_idx_1 = np.where(validity_1)[0]

            proj1_xy[...,0], proj1_xy[...,1] = (W-1) / 2 * (proj1_xy[...,0]+1), (H-1) / 2 * (proj1_xy[...,1]+1)         

        if has_after:
            pcd2 = torch.from_numpy(after_pcds[obj_label]).to(torch.float32).cuda()
            pcd2_homo = torch.cat((pcd2, torch.ones((pcd2.shape[0],1), device=pcd2.device, dtype=torch.float32)), dim=1)
            
            # Projection for after only
            if len(src_indices) == len(change_cameras):
                full_proj_transform_2 = torch.stack([cam.full_proj_transform for cam in change_cameras], dim=0)
            else:
                full_proj_transform_2 = torch.stack(
                    [cam.full_proj_transform for idx, cam in enumerate(change_cameras) if idx not in src_indices],
                    dim=0
                )
            proj_2 = pcd2_homo @ full_proj_transform_2 
            proj_2 = proj_2[..., :3] / proj_2[..., 3:4]
            proj2_xy = proj_2[...,:2].cpu().numpy()

            validity_2 = (proj2_xy[...,0] >= -1) & (proj2_xy[...,0] <= 1) & (proj2_xy[...,1] >= -1) & (proj2_xy[...,1] <= 1)
            validity_2 = np.all(validity_2, 0)
            valid_idx_2 = np.where(validity_2)[0]

            proj2_xy[...,0], proj2_xy[...,1] = (W-1) / 2 * (proj2_xy[...,0]+1), (H-1) / 2 * (proj2_xy[...,1]+1) 

        if not has_before:
            proj_1 = torch.empty_like(proj_2)
            obj_mask_idx = [
                torch.where(object_mask == obj_label)[0][0].item() if idx not in src_indices else None
                for idx, object_mask in enumerate(object_masks)
            ]
        elif not has_after:
            proj_2 = torch.empty_like(proj_1)
            obj_mask_idx = [
                torch.where(object_mask == obj_label)[0][0].item() if idx in src_indices else None
                for idx, object_mask in enumerate(object_masks)
            ]
        else:
            # Acccumulate descriptors for all pairs
            obj_mask_idx = [torch.where(object_mask==obj_label)[0][0].item() for object_mask in object_masks]

        proj_xy = []
        idx_1, idx_2 = 0, 0
        for i in range(len(proj_1)+len(proj_2)):
            if i in src_indices:
                if not has_before:
                    proj_xy.append(None)
                else:
                    proj_xy.append(np.round(proj1_xy[idx_1][valid_idx_1]).astype(int))
                idx_1 += 1
            else:
                if not has_after:
                    proj_xy.append(None)
                else:
                    proj_xy.append(np.round(proj2_xy[idx_2][valid_idx_2]).astype(int))
                idx_2 += 1
        
        # cam_idx = 0
        # plt.figure()
        # plt.scatter(proj1_xy[cam_idx, :, 0], proj1_xy[cam_idx, :, 1], c='blue', s=5, label='Before', alpha=0.6)
        # plt.scatter(proj2_xy[cam_idx, :, 0], proj2_xy[cam_idx, :, 1], c='red', s=5, label='After', alpha=0.6)

        # plt.legend()
        # plt.title(f'2D Projection (Camera {cam_idx})')
        # plt.xlim([0, W])
        # plt.ylim([H, 0])

        # plt.xlabel('NDC X')
        # plt.ylabel('NDC Y')
        # plt.grid(True)
        # plt.savefig(f"{descriptor_path}/registration_{cam_idx}.png")
        # plt.tight_layout()
        # plt.show()

        # Accumulate descriptors
        desc_3d_1, desc_3d_2 = [], []
        count_1, count_2 = [], []
        
        fig, axes = plt.subplots(2, int(len(object_masks)/2), figsize=(12, 6))
        for i, pair in enumerate(pairs):
            for j, pair_elem in enumerate(pair):
                if proj_xy[pair_elem] is None:
                    continue

                x, y = proj_xy[pair_elem][:,0], proj_xy[pair_elem][:,1]
                object_mask = object_masks[pair_elem][obj_mask_idx[pair_elem]]
                count = (object_mask[y, x] == obj_label).to(int)
                
                desc = desc1[i] if j == 0 else desc2[i]
                desc = torch.nn.functional.interpolate(desc[None].permute(0,3,1,2), (H,W)).reshape(-1,H,W).permute(1,2,0)
                
                sampled_desc = desc[y,x]
                sampled_desc[count==0] = 0
                
                if pair_elem in src_indices:
                    desc_3d_1.append(sampled_desc)
                    count_1.append(count)
                else:
                    desc_3d_2.append(sampled_desc)
                    count_2.append(count)

                # # debug & visualize
                pca_desc = compute_pca_image(sampled_desc)
                row, col = pair_elem % 2, pair_elem // 2  
                axes[row, col].scatter(x, y, c=pca_desc, s=5, alpha=0.6)

                axes[row, col].set_title(f'Camera {pair_elem}')
                axes[row, col].set_xlim([0, W])
                axes[row, col].set_ylim([H, 0])

                # axes[row, col].set_xlabel('NDC X')
                # axes[row, col].set_ylabel('NDC Y')
                axes[row, col].grid(True)
                
                # print(count.sum())

        # # plt.show() 
        plt.savefig(f"{descriptor_path}/2d_pca_{obj_label}.png")
        plt.tight_layout()
        plt.close()
    
        count_1 = torch.stack(count_1, dim=0).sum(0) if has_before else None
        count_2 = torch.stack(count_2, dim=0).sum(0) if has_after else None
        zero_count_filter_1 = torch.where(count_1 > minimal_count)[0] if has_before else None
        zero_count_filter_2 = torch.where(count_2 > minimal_count)[0] if has_after else None
        valid_idx_1 = valid_idx_1[zero_count_filter_1] if has_before else None
        valid_idx_2 = valid_idx_2[zero_count_filter_2] if has_after else None

        desc_3d_1 = torch.stack(desc_3d_1, dim=0).sum(0) if has_before else None
        desc_3d_2 = torch.stack(desc_3d_2, dim=0).sum(0) if has_after else None 
        desc_3d_1 = desc_3d_1[zero_count_filter_1] / count_1[zero_count_filter_1,None] if has_before else None
        desc_3d_2 = desc_3d_2[zero_count_filter_2] / count_2[zero_count_filter_2,None] if has_after else None
        
        fused_outputs[obj_label]["desc_3d_1"] = desc_3d_1
        fused_outputs[obj_label]["desc_3d_2"] = desc_3d_2 
        fused_outputs[obj_label]["valid_idx_1"] = valid_idx_1
        fused_outputs[obj_label]["valid_idx_2"] = valid_idx_2
        fused_outputs[obj_label]["proj1_xy"] = proj1_xy if has_before else None
        fused_outputs[obj_label]["proj2_xy"] = proj2_xy if has_after else None

    return fused_outputs 