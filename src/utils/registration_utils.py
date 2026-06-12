import numpy as np
import torch
from scipy.spatial.transform import Rotation as R
from matplotlib import pyplot as plt
from scipy.spatial import cKDTree 
from src.utils.visualization_utils import compute_pca_image
from scipy.optimize import linear_sum_assignment
import teaserpp_python
import time
from src.utils.visualization_utils import visualize_registration

def estimate_rigid_transform(A, B):
    """Estimate rigid transform using SVD."""
    assert A.shape == B.shape
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)

    AA = A - centroid_A
    BB = B - centroid_B

    H = AA.T @ BB
    U, S, Vt = np.linalg.svd(H)
    R_mat = Vt.T @ U.T
    if np.linalg.det(R_mat) < 0:
        Vt[2, :] *= -1
        R_mat = Vt.T @ U.T

    t = centroid_B - R_mat @ centroid_A
    return R_mat, t

def apply_transform(points, R_mat, t):
    return (R_mat @ points.T).T + t

def ransac_3d_rigid(source_points, target_points, threshold=0.01, max_iterations=1000):
    best_inliers = []
    best_transform = None
    N = source_points.shape[0]

    for _ in range(max_iterations):
        # Randomly sample 3 points
        idx = np.random.choice(N, 3, replace=False)
        src_sample = source_points[idx]
        tgt_sample = target_points[idx]

        # Estimate transform
        R_est, t_est = estimate_rigid_transform(src_sample, tgt_sample)

        # Apply transform to all source points
        transformed = apply_transform(source_points, R_est, t_est)

        # Compute error (L2 norm)
        errors = np.linalg.norm(transformed - target_points, axis=1)

        # Find inliers
        inliers = np.where(errors < threshold)[0]

        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_transform = (R_est, t_est)

    print(f"Best inliers: {len(best_inliers)} / {N}")
    return best_transform, best_inliers

# # Example usage:
# # Source and target points (Nx3)
# source_points = np.random.rand(100, 3)
# # Apply known transform to create target points
# true_R = R.from_euler('xyz', [20, 10, 5], degrees=True).as_matrix()
# true_t = np.array([0.5, -0.3, 0.2])
# target_points = (true_R @ source_points.T).T + true_t

# # Add some noise
# target_points += np.random.normal(0, 0.01, target_points.shape)

# # Run RANSAC
# R_est, t_est = ransac_3d_rigid(source_points, target_points, threshold=0.02, max_iterations=500)

# print("\nEstimated Rotation:\n", R_est)
# print("Estimated Translation:\n", t_est)

def mutual_nearest_neighbors(features_A, features_B):
    """
    Finds mutual nearest neighbors (MNN) between two sets of features.

    Args:
        features_A (np.array): Feature matrix A (NxD).
        features_B (np.array): Feature matrix B (MxD).

    Returns:
        matches (list of tuples): List of (index_A, index_B) pairs that are mutual nearest neighbors.
    """
    # Step 1: Build KD-Trees for fast nearest neighbor search
    tree_A = cKDTree(features_A)
    tree_B = cKDTree(features_B)

    # Step 2: Find the nearest neighbor in B for each feature in A
    _, idx_B = tree_A.query(features_B, k=1)  # A -> B mapping

    # Step 3: Find the nearest neighbor in A for each feature in B
    _, idx_A = tree_B.query(features_A, k=1)  # B -> A mapping

    # Step 4: Mutual check (A->B and B->A should be consistent)
    mutual_matches = [(i, idx_B[i]) for i in range(len(idx_B)) if idx_A[idx_B[i]] == i]

    return mutual_matches

def find_3d_correspondences(before_pcds, after_pcds, fused_outputs, matching_obj_labels, descriptor_path, num_sample=5000, matching_algorithm="hungarian"):
    obj_kpts_1, obj_kpts_2 = {}, {}

    for obj_label in matching_obj_labels:
        desc_3d_1, desc_3d_2 = fused_outputs[obj_label]["desc_3d_1"], fused_outputs[obj_label]["desc_3d_2"]
        valid_idx_1, valid_idx_2 = fused_outputs[obj_label]["valid_idx_1"], fused_outputs[obj_label]["valid_idx_2"]
        proj1_xy, proj2_xy = fused_outputs[obj_label]["proj1_xy"], fused_outputs[obj_label]["proj2_xy"]

        pcd1 = torch.from_numpy(before_pcds[obj_label]).to(torch.float32).cuda()
        pcd2 = torch.from_numpy(after_pcds[obj_label]).to(torch.float32).cuda()
        pca_desc_1 = compute_pca_image(desc_3d_1)
        pca_desc_2 = compute_pca_image(desc_3d_2)
        
        # Find 3D correspondences
        # Build cosine similarity matrix and Hungarian matching
        # kp_idx_1, kp_idx_2 = torch.argsort(desc_3d_1, descending=True)[:num_sample], torch.argsort(desc_3d_2, descending=True)[:num_sample]
        kp_idx_1, kp_idx_2 = torch.randperm(len(desc_3d_1))[:num_sample], torch.randperm(len(desc_3d_2))[:num_sample]
        
        kp_desc_1 = desc_3d_1[kp_idx_1] 
        kp_desc_2 = desc_3d_2[kp_idx_2] 
        
        # Normalize keypoints
        kp_desc_1 = kp_desc_1 / torch.linalg.norm(kp_desc_1, dim=-1, keepdim=True)
        kp_desc_2 = kp_desc_2 / torch.linalg.norm(kp_desc_2, dim=-1, keepdim=True)
        cos_similarity = torch.sum(kp_desc_1[:,None] * kp_desc_2[None, :], dim=-1)
        
        ## Hungarian algorithm (solving for minimum cost)
        if matching_algorithm == "hungarian":
            row_indices, col_indices = linear_sum_assignment(1-cos_similarity)
            matches = np.stack([row_indices,col_indices], 1)
 
        ## Mutual Nearest Neighbors
        elif matching_algorithm == "mutual_nearest_neighbors":
            matches = mutual_nearest_neighbors(kp_desc_1, kp_desc_2)
            matches = np.array(matches)
        valid_match_idx = np.where(cos_similarity[matches[:,0], matches[:,1]] > 0.5)[0]
        valid_matches = matches[valid_match_idx]

        kpts_1 = pcd1[valid_idx_1][kp_idx_1].detach().cpu().numpy()
        kpts_2 = pcd2[valid_idx_2][kp_idx_2].detach().cpu().numpy()
        num_vis = min(50, len(valid_match_idx))
        visualize_match_idx = valid_match_idx[:num_vis]

        fig, axes = plt.subplots(1, 2, figsize=(12, 6)) 
        for i, cam_idx in enumerate([0, 1]): 
            ax = axes[i]
            ax.scatter(proj1_xy[cam_idx, valid_idx_1, 0], proj1_xy[cam_idx, valid_idx_1, 1], c=pca_desc_1, s=5, alpha=0.6)
            ax.scatter(proj2_xy[cam_idx, valid_idx_2, 0], proj2_xy[cam_idx, valid_idx_2, 1], c=pca_desc_2, s=5, alpha=0.6)

            # Plot matches as lines
            for idx in visualize_match_idx:
                src_idx, tgt_idx = kp_idx_1[matches[idx, 0]], kp_idx_2[matches[idx, 1]]

                x1, y1 = proj1_xy[cam_idx, valid_idx_1][src_idx]
                x2, y2 = proj2_xy[cam_idx, valid_idx_2][tgt_idx]

                ax.plot([x1, x2], [y1, y2], color='red', alpha=1, linewidth=1.0)

            ax.set_title(f'2D Projection (Camera {cam_idx})')
            ax.set_xlim([0, 960])
            ax.set_ylim([540, 0])

            ax.set_xlabel('NDC X')
            ax.set_ylabel('NDC Y')
            ax.grid(True)

        plt.tight_layout()
        plt.savefig(f"{descriptor_path}/3d_pca_w_matches_{obj_label}.png")
        plt.close()

        # TODO: double check if valid_match_idx is correct
        obj_kpts_1[obj_label] = kpts_1[valid_matches[:,0]]
        obj_kpts_2[obj_label] = kpts_2[valid_matches[:,1]]

    return obj_kpts_1, obj_kpts_2

def compose_mat4_from_teaserpp_solution(solution):
    """
    Compose a 4-by-4 matrix from teaserpp solution
    """
    s = solution.scale
    rotR = solution.rotation
    t = solution.translation
    T = np.eye(4)
    T[0:3, 3] = t
    R = np.eye(4)
    R[0:3, 0:3] = rotR
    M = T.dot(R)

    if s == 1:
        M = T.dot(R)
    else:
        S = np.eye(4)
        S[0:3, 0:3] = np.diag([s, s, s])
        M = T.dot(R).dot(S)

    return M

## Teaser++ ##
# Populating the parameters
# def define_teaserpp_solver_params():
#     solver_params = teaserpp_python.RobustRegistrationSolver.Params()
#     solver_params.cbar2 = 1
#     solver_params.noise_bound = 0.05
#     solver_params.estimate_scaling = False
#     solver_params.rotation_estimation_algorithm = teaserpp_python.RobustRegistrationSolver.ROTATION_ESTIMATION_ALGORITHM.GNC_TLS
#     solver_params.rotation_gnc_factor = 1.4
#     solver_params.rotation_max_iterations = 100
#     solver_params.rotation_cost_threshold = 1e-12

#     return solver_params

def run_teaserpp(obj_kpts_1, obj_kpts_2, matching_obj_labels):
    solver_params = teaserpp_python.RobustRegistrationSolver.Params()
    solver_params.cbar2 = 1
    solver_params.noise_bound = 0.05
    solver_params.estimate_scaling = False
    solver_params.rotation_estimation_algorithm = teaserpp_python.RobustRegistrationSolver.ROTATION_ESTIMATION_ALGORITHM.GNC_TLS
    solver_params.rotation_gnc_factor = 1.4
    solver_params.rotation_max_iterations = 100
    solver_params.rotation_cost_threshold = 1e-12

    est_mats = {}
    
    for obj_label in matching_obj_labels:
        solver = teaserpp_python.RobustRegistrationSolver(solver_params)
        # start = time.time()
        src = obj_kpts_1[obj_label].transpose(1,0)
        dst = obj_kpts_2[obj_label].transpose(1,0)
        solver.solve(src, dst)
        # end = time.time()

        solution = solver.getSolution()
        est_mat = compose_mat4_from_teaserpp_solution(solution)
        est_mats[obj_label] = est_mat

        # print("=====================================")
        # print("          TEASER++ Results           ")
        # print("=====================================")

        # print("Time taken (s): ", end - start)
        # print("Estimated rotation: ")
        # print(solution.rotation)
        # print("Estimated translation: ")
        # print(solution.translation)

        # visualize_registration(src, dst, est_mat)

    return est_mats

def chamfer_distance(point_cloud_src, point_cloud_tgt):
    # point_cloud shape : B x N x 3
    src_expanded = point_cloud_src[:,:,None]  # B x N x 1 x 3
    tgt_expanded = point_cloud_tgt[:, None]  # B x 1 x M x 3

    dist = np.linalg.norm(src_expanded - tgt_expanded, axis=-1)

	# For each source point, find the closest point in the target
    min_dist_src_to_tgt = np.min(dist, axis=2)[0]  # B x N

	# For each target point, find the closest point in the source
    min_dist_tgt_to_src = np.min(dist, axis=1)[0]  # B x M

	# Average the distances
    loss_chamfer = np.mean(min_dist_src_to_tgt) + np.mean(min_dist_tgt_to_src)
    
    return loss_chamfer

def earth_movers_distance(point_cloud_src,point_cloud_tgt):
    B, N, _ = point_cloud_src.shape
    emd_batch = []

    for b in range(B):
        src = point_cloud_src[b]  # (N, 3)
        tgt = point_cloud_tgt[b]  # (N, 3)

        # Compute pairwise L2 distance matrix (N, N)
        dist_matrix = np.linalg.norm(src[:, np.newaxis, :] - tgt[np.newaxis, :, :], axis=2)

        # Greedy matching (approximation of EMD)
        match1 = np.min(dist_matrix, axis=1)  # from src to tgt
        match2 = np.min(dist_matrix, axis=0)  # from tgt to src

        emd = (np.mean(match1) + np.mean(match2)) / 2.0
        emd_batch.append(emd)

    loss_emd = np.mean(emd_batch)
    return loss_emd

