# Using Depth-Anything's depth (Optional)
# python src/utils/make_depth_scale.py --base_dir ${IMAGE_PATH} --depths_dir ${DEPTH_PATH}

DATA_DIR=data/ltgs_dataset
SCENE=livingroom
GPU=0
IMAGE_PATH=${DATA_DIR}/${SCENE}/hloc
DEPTH_PATH=${DATA_DIR}/${SCENE}/depths
OUTPUT=output/${SCENE}_first_state

# HLOC localization
CUDA_VISIBLE_DEVICES=${GPU} python localization.py -m gaussian-splatting/${OUTPUT}  
# Change detection
CUDA_VISIBLE_DEVICES=${GPU} python change_detection.py -m gaussian-splatting/${OUTPUT} --min_size 1000 --kernel_size 5  # --manual_selection 

# Instance matching
CUDA_VISIBLE_DEVICES=${GPU} python instance_matching.py -m gaussian-splatting/${OUTPUT} --single_similarity_thres 0.93 --multi_similarity_thres 0.9 --optim_level refine --filter_consistent --num_match_thres 50
CUDA_VISIBLE_DEVICES=${GPU} python pcd_initialization.py -m gaussian-splatting/${OUTPUT} --slackness 0

# Updating
# CUDA_VISIBLE_DEVICES=${GPU} python long_term_update.py -m gaussian-splatting/${OUTPUT} --overlap_thres 0.1 --update_dist_thres 0.05 #--optimizer_type sparse_adam
# python metrics.py -m output/${SCENE}   
# v2
CUDA_VISIBLE_DEVICES=${GPU} python long_term_update.py -m gaussian-splatting/${OUTPUT}  --overlap_thres 0.1 --refine_iterations 5000 --use_previous_viewpoints  
python metrics.py -m output/${SCENE}   