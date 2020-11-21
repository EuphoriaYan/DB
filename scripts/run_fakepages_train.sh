
export CUDA_VISIBLE_DEVICES=2,3

python train.py \
experiments/seg_detector/fakepages_resnet18_deform_thre.yaml \
--num_gpus 2 \
