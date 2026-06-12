# import open3d as o3d
import os
import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import open3d as o3d
from sklearn.decomposition import PCA
from skimage import measure

class ColorInfo:
    colors = {
        "red": (255, 0, 0),       
        "green": (0, 255, 0),      
        "blue": (0, 0, 255),       
        "yellow": (255, 255, 0),    
        "cyan": (0, 255, 255),     
        "magenta": (255, 0, 255),     
        "dark_red": (128, 0, 0),    
        "dark_green": (0, 128, 0),
        "dark_blue": (0, 0, 128),
        "olive": (128, 128, 0),    
        "orange": (255, 165, 0), 
        "pink": (255, 192, 203),
        "purple": (128, 0, 128)
    }
    def get_color(self, index):
        # Get the color by index from the dictionary (index should be 0 to len(colors)-1)
        color_names = list(self.colors.keys())
        color_name = color_names[index % len(color_names)]  # Ensure we loop if index is out of bounds
        return self.colors[color_name]
    
def get_camera_mesh(pose,depth=1,vis_2d=False, fovs=[2*np.arctan(torch.tensor(0.5)), 2*np.arctan(torch.tensor(0.5))]):    
    def invert(pose,use_inverse=False):
        # invert a camera pose
        R,t = pose[...,:3],pose[...,3:] 
        R_inv = R.inverse() if use_inverse else R.transpose(-1,-2)
        t_inv = (-R_inv@t)[...,0]
        pose_inv = torch.cat([R_inv,t_inv[...,None]],dim=-1) # [...,3,4]
        
        assert(pose_inv.shape[-2:]==(3,4))

        return pose_inv

    def to_hom(X):
        # get homogeneous coordinates of the input
        X_hom = torch.cat([X,torch.ones_like(X[...,:1])],dim=-1)
        return X_hom

    def cam2world(X,pose):
        X_hom = to_hom(X)
        pose_inv = invert(pose)

        return X_hom@pose_inv.transpose(-1,-2)

    if not vis_2d:
        vertices = torch.tensor([[-np.tan(fovs[0]/2),-np.tan(fovs[1]/2),1],
                                [np.tan(fovs[0]/2),-np.tan(fovs[1]/2),1],
                                [np.tan(fovs[0]/2),np.tan(fovs[1]/2),1],
                                [-np.tan(fovs[0]/2),np.tan(fovs[1]/2),1],
                                [0,0,0]])*depth
        faces = torch.tensor([[0,1,2],
                            [0,2,3],
                            [0,1,4],
                            [1,2,4],
                            [2,3,4],
                            [3,0,4]])
        vertices = cam2world(vertices[None],pose)
        wireframe_idx = [0,1,2,3,0,4,1,2,4,3]
        center_idx = 5
    
    else: 
        # project 3d camera to 2d
        axis = torch.tensor([[0,0,0],
                            [0,0,1]])*depth
        faces = None
        
        axis = cam2world(axis[None],pose)
        
        thetas = [-np.arctan(0.5), np.arctan(0.5)] 
        origin_2d = axis[:,:1,:2]
        direction_2d = axis[:,1:,:2]-axis[:,:1,:2]
        direction_2d = direction_2d / torch.linalg.norm(direction_2d, dim=-1, keepdim=True)
        direction_2d = direction_2d.transpose(-1, -2)

        rot1 = torch.tensor([[np.cos(thetas[0]), -np.sin(thetas[0])],
                              [np.sin(thetas[0]), np.cos(thetas[0])]], dtype=torch.float32)
        rot2 = torch.tensor([[np.cos(thetas[1]), -np.sin(thetas[1])],
                              [np.sin(thetas[1]), np.cos(thetas[1])]], dtype=torch.float32)
        
        vertex1 = origin_2d + depth * (rot1 @ direction_2d).transpose(-1, -2)
        vertex2 = origin_2d + depth * (rot2 @ direction_2d).transpose(-1, -2)     
        vertices = torch.cat([origin_2d, vertex1, vertex2], dim=1)

        wireframe_idx = [0,1,2,0]
        center_idx = 0

    wireframe = vertices[:,wireframe_idx]

    return vertices,faces,wireframe,center_idx

def create_vector_arrow(origin, vector, color, scale=1.0):
    arrow = o3d.geometry.LineSet()
    start_point = np.array(origin, dtype=np.float64)
    end_point = start_point + scale * np.array(vector, dtype=np.float64)
    arrow.points = o3d.utility.Vector3dVector([start_point, end_point])
    arrow.lines = o3d.utility.Vector2iVector([[0, 1]])
    arrow.colors = o3d.utility.Vector3dVector([color])  # RGB color
    return arrow

def visualize_geometry(camera_sets, camera_viewpoint=False, points3d_xyz=None, points3d_rgb=None, iteration=0):
    vis = o3d.visualization
    assets = []

    # coor_frame = o3d.geometry.TriangleMesh.create_coordinate_frame()
    # vis.add_geometry(coor_frame)
    
    ## Add cameras to the visualizer
    for _, cam_type in enumerate(camera_sets):
        cameras = torch.from_numpy(camera_sets[cam_type]).float()
        w2c = cameras[:,:3]
        # w2c = [cam.world_view_transform[:3] for cam in cameras]
        # w2c = torch.stack(w2c, axis=0).detach().cpu()
        # fovX, fovY = 2*np.arctan(cameras[0].width / (2*cameras[0].fx)), 2*np.arctan(cameras[0].height / (2*cameras[0].fy))
        
        vertices,faces,_,_ = get_camera_mesh(w2c, depth=0.15, vis_2d=False)
        vertices, faces = vertices.numpy().astype(np.float64), faces.numpy().astype(np.int32)

        for idx in range(len(cameras)):    
            # Get the world-to-camera transform
            R_w2c = w2c[idx, :3, :3]  # Rotation matrix
            t_w2c = w2c[idx, :3, 3]   # Translation vector

            # Convert to camera-to-world (required for Open3D visualization)
            R_c2w = R_w2c.T
            t_c2w = -R_w2c.T @ t_w2c

            # Create coordinate frame for the camera
            coor_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.25, origin=[0, 0, 0])

            # Apply the camera-to-world transformation
            transform = np.eye(4)
            transform[:3, :3] = R_c2w  # Set rotation
            transform[:3, 3] = t_c2w   # Set translation
            coor_frame.transform(transform)

            # Add the coordinate frame to the assets
            assets.append(coor_frame)

            mesh = o3d.geometry.TriangleMesh()
            mesh.vertices = o3d.utility.Vector3dVector(vertices[idx])
            mesh.triangles = o3d.utility.Vector3iVector(faces)
            mesh.compute_vertex_normals()
            
            lines = o3d.geometry.LineSet()
            lines = lines.create_from_triangle_mesh(mesh)
            
            lines_array = np.asarray(lines.lines)
            new_lines_array = np.delete(lines_array, 2, axis=0)

            lines.lines = o3d.utility.Vector2iVector(new_lines_array)
            if cam_type == "initial" or cam_type == "hloc":
                lines.paint_uniform_color(np.array([0.0,0.0,1.0],dtype=np.float64))
            elif cam_type == "refined" or cam_type == "mast3r":
                lines.paint_uniform_color(np.array([1.0,0.0,0.0],dtype=np.float64))
            elif cam_type == "mast3r_render":
                lines.paint_uniform_color(np.array([0.0,1.0,0.0],dtype=np.float64))
            elif cam_type == "mast3r_gt":
                lines.paint_uniform_color(np.array([1.0,0.0,1.0],dtype=np.float64))

            assets.append(lines)
    
    ## 3. Add COLMAP points to the visualizer
    if points3d_xyz is not None:
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(points3d_xyz)
        if points3d_rgb is None:
            point_cloud.paint_uniform_color(np.array([0.0,0.0,0.0],dtype=np.float64)) # if renderer else 0
        else:
            point_cloud.colors = o3d.utility.Vector3dVector(points3d_rgb[:,:3].astype(np.float64))
        assets.append(point_cloud) 

    vis.draw(assets, show_skybox=False)


def visualize_object_points(object_pcds):
    vis = o3d.visualization
    assets = []

    for obj_idx, pcd_xyz in object_pcds.items():
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(pcd_xyz)
        if obj_idx == 0:
            point_cloud.paint_uniform_color(np.array([0.0,0.0,0.0],dtype=np.float64))
        else:
            color_infos = ColorInfo()
            pcd_rgb = np.array(color_infos.get_color(obj_idx-1)) / 255.0
            point_cloud.paint_uniform_color(pcd_rgb.astype(np.float64))
        assets.append(point_cloud) 

    vis.draw(assets, show_skybox=False)


def plot_rendering(renderings, gts, output_dir, refined_renderings=None, iteration=0):
    # Create a subplot

    if refined_renderings is not None:
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))  # Adjust figsize as needed

        # Populate the subplots
        axes[0, 0].imshow(renderings[0].permute(1,2,0).detach().cpu().numpy())
        axes[0, 0].set_title("Rendering 1")
        axes[0, 0].axis("off")  

        axes[0, 1].imshow(refined_renderings[0].permute(1,2,0).detach().cpu().numpy())
        axes[0, 1].set_title("Refined 1")
        axes[0, 1].axis("off")

        axes[0, 2].imshow(gts[0].permute(1,2,0).detach().cpu().numpy())
        axes[0, 2].set_title("Change 1")
        axes[0, 2].axis("off")

        axes[1, 0].imshow(renderings[1].permute(1,2,0).detach().cpu().numpy())
        axes[1, 0].set_title("Rendering 2")
        axes[1, 0].axis("off")

        axes[1, 1].imshow(refined_renderings[1].permute(1,2,0).detach().cpu().numpy())
        axes[1, 1].set_title("Refined 2")
        axes[1, 1].axis("off")

        axes[1, 2].imshow(gts[1].permute(1,2,0).detach().cpu().numpy())
        axes[1, 2].set_title("Change 2")
        axes[1, 2].axis("off")

        # Adjust layout and show the plot
        plt.tight_layout()
        plt.savefig(f"output/localization_iter={iteration}.png")

    else:
        fig, axes = plt.subplots(2, len(renderings), figsize=(15, 8))  # Adjust figsize as needed

        for i in range(len(renderings)):
            # Populate the subplots
            axes[0, i].imshow(renderings[i].permute(1,2,0).detach().cpu().numpy())
            axes[0, i].set_title(f"Rendering {i+1}")
            axes[0, i].axis("off")  

            axes[1, i].imshow(gts[i].permute(1,2,0).detach().cpu().numpy())
            axes[1, i].set_title(f"Change {i+1}")
            axes[1, i].axis("off")

        # Adjust layout and show the plot
        fig.tight_layout()
        fig.savefig(f"{output_dir}/hloc/hloc_localization.png")

    # plt.show()


def plot_sam_embeddings(renderings, gts, output_dir, cos_similarity, embedding_masks):
    if isinstance(cos_similarity[0], torch.Tensor):
        cos_similarity[0] = cos_similarity[0].detach().cpu().numpy()
        cos_similarity[1] = cos_similarity[1].detach().cpu().numpy()

    if isinstance(embedding_masks[0], torch.Tensor):
        embedding_masks[0] = embedding_masks[0].detach().cpu().numpy()
        embedding_masks[1] = embedding_masks[1].detach().cpu().numpy()

    # Create a subplot
    fig, axes = plt.subplots(2, 4, figsize=(15, 8))  # Adjust figsize as needed

    # Populate the subplots
    axes[0, 0].imshow(renderings[0])#.permute(1,2,0).detach().cpu().numpy())
    # for kp in keypoints[render_indices[0]]:  
    #     x, y = kp
    #     axes[0, 0].scatter(x, y, c="blue", s=1) 
    axes[0, 0].set_title("Rendering 1")
    axes[0, 0].axis("off")  

    axes[0, 1].imshow(gts[0])#.permute(1,2,0).detach().cpu().numpy())
    # for kp in keypoints[gt_indices[0]]:  # Loop over keypoints
    #     x, y = kp
    #     axes[0, 1].scatter(x, y, c="blue", s=1)  # Use scatter to plot keypoints
    axes[0, 1].set_title("Change 1")
    axes[0, 1].axis("off")

    cos1 = axes[0, 2].imshow(cos_similarity[0])
    axes[0, 2].set_title("Cos Similarity 1")
    axes[0, 2].axis("off")
    # bar1 = fig.colorbar(cos1, ax=axes[0,2], fraction=0.046, pad=0.04)  # Add colorbar

    axes[0, 3].imshow(embedding_masks[0])
    axes[0, 3].set_title("Change Masks")
    axes[0, 3].axis("off")

    axes[1, 0].imshow(renderings[1])#.permute(1,2,0).detach().cpu().numpy())
    # for kp in keypoints[render_indices[1]]:  
    #     x, y = kp
    #     axes[1, 0].scatter(x, y, c="blue", s=1) 
    axes[1, 0].set_title("Rendering 2")
    axes[1, 0].axis("off")  

    axes[1, 1].imshow(gts[1])#.permute(1,2,0).detach().cpu().numpy())
    # for kp in keypoints[gt_indices[1]]:  
    #     x, y = kp
    #     axes[1, 1].scatter(x, y, c="blue", s=1)  
    axes[1, 1].set_title("Change 2")
    axes[1, 1].axis("off")

    cos2 = axes[1, 2].imshow(cos_similarity[1])
    axes[1, 2].set_title("Cos Similarity 2")
    axes[1, 2].axis("off")
    # bar2 = fig.colorbar(cos2, ax=axes[1,2], fraction=0.046, pad=0.04)  # Add colorbar

    axes[1, 3].imshow(embedding_masks[1])
    axes[1, 3].set_title("Change Masks")
    axes[1, 3].axis("off")

    # Adjust layout and show the plot
    plt.tight_layout()
    plt.savefig(f"{output_dir}/sam/sam.png")
    # plt.show()
    # plt.close('all')


def plot_object_masks(colored_objects, output_dir, timestep=None):
    fig, axes = plt.subplots(2, int(len(colored_objects)/2), figsize=(15, 8))  # Adjust figsize as needed
    for i, colored_object in enumerate(colored_objects):
        row, col = int(i % 2), int(i / 2)    
        axes[row, col].imshow(colored_object)
        if row == 0:
            axes[row, col].set_title(f"Rendering {row}")
        else:
            axes[row, col].set_title(f"Change {row}")
        axes[row, col].axis("off")

    plt.tight_layout()
    if timestep is None:
        plt.savefig(f"{output_dir}/instances/object_masks_mast3r.png")
    else:
        plt.savefig(f"{output_dir}/instances/object_masks_mast3r_{timestep}.png")
    # plt.show()
    plt.close('all')


def plot_matches(matches_all, image_batch, output_dir, render_indices, gt_indices, n_viz=80, timestep=None):
    fig, axes = plt.subplots(int(len(gt_indices)*(len(gt_indices)-1)), 2) #, figsize=(15, 4))  
    if len(axes.shape) == 1:
        axes = axes.reshape(1, 2)
    
    plt_col = 0
    for idx, input in enumerate(list(["Rendering", "Change"])):   
        print("Instance Matching using MASt3R for ", input)
        if input == "Rendering":
            input_indices = render_indices
        else:
            input_indices = gt_indices
        input_batch = [image_batch[idx] for idx in input_indices]
        matches = matches_all[idx]

        for i, image_i in enumerate(input_batch):
            if i == len(input_batch) - 1:
                continue
            for j, image_j in enumerate(input_batch[i+1:], i+1):
                matched_kpts = matches[i][j]
                # score = scores[i][j]

                if n_viz > len(matched_kpts[0]):
                    kpts0, kpts1 = matched_kpts[0], matched_kpts[1]
                else:
                    kpts0, kpts1 = matched_kpts[0][np.random.permutation(len(matched_kpts[0]))[:n_viz]], matched_kpts[1][np.random.permutation(len(matched_kpts[0]))[:n_viz]]

                axes[plt_col, 0].imshow(image_i)
                axes[plt_col, 0].set_title(f"{input} {i+1}")
                axes[plt_col, 0].axis("off")  

                axes[plt_col, 1].imshow(image_j)
                axes[plt_col, 1].set_title(f"{input} {j+1}")
                axes[plt_col, 1].axis("off")  

                for kpt_idx in range(len(kpts0)):
                    fig.add_artist(
                        matplotlib.patches.ConnectionPatch(
                            xyA=(kpts0[kpt_idx, 0], kpts0[kpt_idx, 1]),
                            coordsA=axes[plt_col, 0].transData,
                            xyB=(kpts1[kpt_idx, 0], kpts1[kpt_idx, 1]),
                            coordsB=axes[plt_col, 1].transData,
                            zorder=1,
                            color="red",
                            linewidth=1.0,
                            alpha=1.0,
                        )
                    )
                # freeze the axes to prevent the transform to change
                axes[plt_col, 0].autoscale(enable=False)
                axes[plt_col, 1].autoscale(enable=False)

                axes[plt_col, 0].scatter(kpts0[:, 0], kpts0[:, 1], c="red", s=1)
                axes[plt_col, 1].scatter(kpts1[:, 0], kpts1[:, 1], c="red", s=1)
                plt_col += 1

    # Adjust layout and show the plot
    plt.tight_layout()
    if timestep is None:
        plt.savefig(os.path.join(output_dir, "instances", "matches_mast3r_intra.png"))
    else:
        plt.savefig(os.path.join(output_dir, "instances", f"matches_mast3r_intra_{timestep}.png"))

    # Save the plot
    # plt.show()
    # plt.close('all')


## SAM visualization utils
def show_mask(mask, ax, random_color=False, borders = True):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask = mask.astype(np.uint8)
    mask_image =  mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    if borders:
        import cv2
        contours, _ = cv2.findContours(mask,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
        # Try to smooth contours
        contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
        mask_image = cv2.drawContours(mask_image, contours, -1, (1, 1, 1, 0.5), thickness=2) 
    ax.imshow(mask_image)

# from GeSCF
def show_change_masks(image_batch, change_masks, scene_name, render_indices, gt_indices):
    fig, axes = plt.subplots(2, len(render_indices), figsize=(15, 8))   
    for i in range(len(render_indices)):
        # Populate the subplots
        axes[0, i].imshow(image_batch[render_indices[i]])
        show_mask_new(change_masks[render_indices[i]].astype(np.float32), axes[0,i])
        axes[0, i].set_title(f"Rendering {i}")
        axes[0, i].axis("off")  

        axes[1, i].imshow(image_batch[gt_indices[i]])
        show_mask_new(change_masks[gt_indices[i]].astype(np.float32), axes[1,i])
        axes[1, i].set_title(f"Change {i}")
        axes[1, i].axis("off")  
    
    fig.tight_layout()
    fig.savefig(f"output/{scene_name}/change/gescf_masks.png")


def show_mask_new(mask, ax, random_color=False, edge_color='black', contour_thickness=0.0, darker=False):
    # Generate random or fixed color for the mask interior
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    elif darker:
        color = np.array([0, 0, 0, 0.4])  # black with transparency
    else:
        color = np.array([255/255, 80/255, 255/255, 0.6])  # pinkish color with transparency

    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)

    # Show the mask interior
    ax.imshow(mask_image)

    # Find contours for the mask to draw the edges
    contours = measure.find_contours(mask, 0.5)

    for contour in contours:
        # Draw each contour
        ax.plot(contour[:, 1], contour[:, 0], color=edge_color, linewidth=contour_thickness)

def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))    

def show_masks(image, masks, scores, point_coords=None, box_coords=None, input_labels=None, borders=True):
    for i, (mask, score) in enumerate(zip(masks, scores)):
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        show_mask(mask, plt.gca(), borders=borders)
        if point_coords is not None:
            assert input_labels is not None
            show_points(point_coords, input_labels, plt.gca())
        if box_coords is not None:
            # boxes
            show_box(box_coords, plt.gca())
        if len(scores) > 1:
            plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
        plt.axis('off')
        plt.show()

def show_anns(anns, borders=True):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:, :, 3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.5]])
        img[m] = color_mask 
        if borders:
            import cv2
            contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
            # Try to smooth contours
            contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
            cv2.drawContours(img, contours, -1, (0, 0, 1, 0.4), thickness=1) 

    ax.imshow(img)

def visualize_matches(matches_im0, matches_im1, view1, view2, vis_idx = 0, within_objects=False):
    im1, im2 = view1['img'][vis_idx].detach().cpu().numpy(), view2['img'][vis_idx].detach().cpu().numpy()
    
    H0, W0 = view1['true_shape'][0]
    valid_matches_im0 = (matches_im0[:, 0] >= 3) & (matches_im0[:, 0] < int(W0) - 3) & (
        matches_im0[:, 1] >= 3) & (matches_im0[:, 1] < int(H0) - 3) 

    H1, W1 = view2['true_shape'][0]
    valid_matches_im1 = (matches_im1[:, 0] >= 3) & (matches_im1[:, 0] < int(W1) - 3) & (
        matches_im1[:, 1] >= 3) & (matches_im1[:, 1] < int(H1) - 3) 

    if within_objects:
        valid_matches_im0 = valid_matches_im0 & np.all(im1[:, matches_im0[:,1], matches_im0[:,0]]>0, 0)
        valid_matches_im1 = valid_matches_im1 & np.all(im2[:, matches_im1[:,1], matches_im1[:,0]]>0, 0)

    valid_matches = valid_matches_im0 & valid_matches_im1
    matches_im0, matches_im1 = matches_im0[valid_matches], matches_im1[valid_matches]

    # visualize a few matches
    num_matches = matches_im0.shape[0]
    n_viz = 20
    match_idx_to_viz = np.round(np.linspace(0, num_matches - 1, n_viz)).astype(int)
    viz_matches_im0, viz_matches_im1 = matches_im0[match_idx_to_viz], matches_im1[match_idx_to_viz]

    image_mean = torch.as_tensor([0.5, 0.5, 0.5], device='cpu').reshape(1, 3, 1, 1)
    image_std = torch.as_tensor([0.5, 0.5, 0.5], device='cpu').reshape(1, 3, 1, 1)

    viz_imgs = []
    for i, view in enumerate([view1, view2]):
        rgb_tensor = view['img'] * image_std + image_mean
        viz_imgs.append(rgb_tensor[vis_idx].permute(1, 2, 0).cpu().numpy())

    H0, W0, H1, W1 = *viz_imgs[0].shape[:2], *viz_imgs[1].shape[:2]
    img0 = np.pad(viz_imgs[0], ((0, max(H1 - H0, 0)), (0, 0), (0, 0)), 'constant', constant_values=0)
    img1 = np.pad(viz_imgs[1], ((0, max(H0 - H1, 0)), (0, 0), (0, 0)), 'constant', constant_values=0)
    img = np.concatenate((img0, img1), axis=1)
    plt.figure()
    plt.imshow(img)
    cmap = plt.get_cmap('jet')
    for i in range(n_viz):
        (x0, y0), (x1, y1) = viz_matches_im0[i].T, viz_matches_im1[i].T
        plt.plot([x0, x1 + W0], [y0, y1], '-+', color=cmap(i / (n_viz - 1)), scalex=False, scaley=False)
    plt.show(block=True)


def compute_pca_image(desc, n_components=3):
    """Computes PCA on descriptors and returns a normalized RGB image."""
    if len(desc.shape) == 3:
        H, W, C = desc.shape  # Shape: [H, W, 24]
        output_shape = [H, W, 3]
    elif len(desc.shape) == 2:
        HW, C = desc.shape
        output_shape = [HW, 3]

    # Reshape descriptor to (H*W, C) for PCA
    desc_reshaped = desc.reshape(-1, C)  # Shape: [H*W, 24]
    
    # Apply PCA to reduce from 24D -> 3D
    pca = PCA(n_components=n_components)
    fit_pca = pca.fit(desc_reshaped)  # Shape: [H*W, 3]
    pca_colors = fit_pca.transform(desc_reshaped)  # Transformed data shape: [N, 3]
    # Normalize PCA output to [0, 1] for visualization
    pca_colors = (pca_colors - pca_colors.min()) / (pca_colors.max() - pca_colors.min())

    # Reshape back to image format [H, W, 3]
    return pca_colors.reshape(output_shape)


# Convert source and destination points to Open3D PointCloud format
def visualize_registration(src, dst, est_mat):
    pcd_src = o3d.geometry.PointCloud()
    pcd_src.points = o3d.utility.Vector3dVector(src.T)  # Convert (3, N) → (N, 3)
    pcd_src.paint_uniform_color([0, 0, 1])  # Blue color for source points

    pcd_dst = o3d.geometry.PointCloud()
    pcd_dst.points = o3d.utility.Vector3dVector(dst.T)
    pcd_dst.paint_uniform_color([0, 1, 0])  # Green color for destination points

    # Apply transformation to source points
    src_transformed = (est_mat[:3, :3] @ src + est_mat[:3, 3:4]).T  # Apply R*src + t
    pcd_src_transformed = o3d.geometry.PointCloud()
    pcd_src_transformed.points = o3d.utility.Vector3dVector(src_transformed)
    pcd_src_transformed.paint_uniform_color([1, 0, 0])  # Red color for transformed source points

    # Visualize
    o3d.visualization.draw([pcd_src, pcd_dst, pcd_src_transformed])


