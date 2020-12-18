
export CUDA_VISIBLE_DEVICES=1

python special_demo.py \
experiments/seg_detector/fakepages_resnet18_deform_thre.yaml \
--image_path ../data/DingXiu \
--resume \
models/model_epoch_490_minibatch_396000 \
--box_thresh 0.5 \
--sort_boxes \
