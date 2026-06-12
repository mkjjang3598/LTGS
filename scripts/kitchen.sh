# Using Depth-Anything's depth (Optional)
# python src/utils/make_depth_scale.py --base_dir ${IMAGE_PATH} --depths_dir ${DEPTH_PATH}

DATA_DIR=data/cl_nerf
SCENE=kitchen
GPU=0
IMAGE_PATH=${DATA_DIR}/${SCENE}/hloc
DEPTH_PATH=${DATA_DIR}/${SCENE}/depths
OUTPUT=output/${SCENE}_first_state

CUDA_VISIBLE_DEVICES=${GPU} python localization.py -m gaussian-splatting/${OUTPUT} --skip_localization # HLOC localization

CUDA_VISIBLE_DEVICES=${GPU} python change_detection.py -m gaussian-splatting/${OUTPUT} --min_size 1500 --kernel_size 5 --cosine_thr 0.92  # --manual_selection # Change detection

# Instance matching
CUDA_VISIBLE_DEVICES=${GPU} python instance_matching.py -m gaussian-splatting/${OUTPUT} --filter_consistent --single_similarity_thres 0.85 --multi_similarity_thres 0.85 --optim_level refine
CUDA_VISIBLE_DEVICES=${GPU} python pcd_initialization.py -m gaussian-splatting/${OUTPUT} --slackness 0.8

# Updating
# CUDA_VISIBLE_DEVICES=${GPU} python long_term_update.py -m gaussian-splatting/${OUTPUT} --skip_localization --update_dist_thres 0.05 --optimizer_type sparse_adam --invalid_initialization 1,2,4 --initial_time_loss_weight 0.1 --refine_iterations 1500


# # v1
# CUDA_VISIBLE_DEVICES=${GPU} python long_term_update.py -m gaussian-splatting/${OUTPUT} --skip_localization --update_dist_thres 0.05 --optimizer_type sparse_adam --invalid_initialization 1,2,4 --initial_time_loss_weight 0.1 --refine_iterations 1500

# v2
CUDA_VISIBLE_DEVICES=${GPU} python long_term_update.py -m gaussian-splatting/${OUTPUT} --refine_iterations 5000 --skip_localization --use_previous_viewpoints

python metrics.py -m output/${SCENE}  # Metrics