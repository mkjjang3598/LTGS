# Using Depth-Anything's depth (Optional)
# python src/utils/make_depth_scale.py --base_dir ${IMAGE_PATH} --depths_dir ${DEPTH_PATH}

DATA_DIR=data/WAT               
SCENE=mac        
GPU=0
IMAGE_PATH=${DATA_DIR}/${SCENE}     
DEPTH_PATH=${DATA_DIR}/${SCENE}/depths
OUTPUT=output/${SCENE}_first_state_sfm_all

# HLOC localization
CUDA_VISIBLE_DEVICES=${GPU} python localization.py -m gaussian-splatting/${OUTPUT} --skip_localization
# Change detection
CUDA_VISIBLE_DEVICES=${GPU} python change_detection.py -m gaussian-splatting/${OUTPUT} --min_size 1500 --kernel_size 5 --manual_selection

# Instance matching
CUDA_VISIBLE_DEVICES=${GPU} python instance_matching.py -m gaussian-splatting/${OUTPUT} --filter_consistent --multi_similarity_thres 0.93 --optim_level refine+depth --fix_separated
CUDA_VISIBLE_DEVICES=${GPU} python pcd_initialization.py -m gaussian-splatting/${OUTPUT} --slackness 0.0

# v2
CUDA_VISIBLE_DEVICES=${GPU} python long_term_update.py -m gaussian-splatting/${OUTPUT} --overlap_thres 0.15 --refine_iterations 5000 --skip_localization --use_previous_viewpoints  
python metrics.py -m output/${SCENE}   