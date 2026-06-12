# Using Depth-Anything's depth (Optional)
# python src/utils/make_depth_scale.py --base_dir ${IMAGE_PATH} --depths_dir ${DEPTH_PATH}

DATA_DIR=data/ltgs_dataset
SCENE=hall
GPU=0
IMAGE_PATH=${DATA_DIR}/${SCENE} #/hloc
DEPTH_PATH=${DATA_DIR}/${SCENE}/depths
OUTPUT=output/${SCENE}_first_state_sfm_all

# HLOC localization
CUDA_VISIBLE_DEVICES=${GPU} python localization.py -m gaussian-splatting/${OUTPUT} --skip_localization
# Change detection
CUDA_VISIBLE_DEVICES=${GPU} python change_detection.py -m gaussian-splatting/${OUTPUT} --min_size 1500 --connectivity 2 --cosine_thr 0.9 --kernel_size 3

# Instance matching
CUDA_VISIBLE_DEVICES=${GPU} python instance_matching.py -m gaussian-splatting/${OUTPUT} --filter_consistent --fix_separated --compare_mean --optim_level refine+depth
CUDA_VISIBLE_DEVICES=${GPU} python pcd_initialization.py -m gaussian-splatting/${OUTPUT} --slackness 0 

# Updating
CUDA_VISIBLE_DEVICES=${GPU} python long_term_update.py -m gaussian-splatting/${OUTPUT} --refine_iterations 2000 --overlap_thres 0.15 --conf_thres 2.5 --obj_pose_lr 0.001 --skip_localization
python metrics.py -m output/${SCENE}

# v2
CUDA_VISIBLE_DEVICES=${GPU} python long_term_update.py -m gaussian-splatting/${OUTPUT} --refine_iterations 5000 --overlap_thres 0.15 --conf_thres 2.5 --skip_localization --use_previous_viewpoints  
python metrics.py -m output/${SCENE}   