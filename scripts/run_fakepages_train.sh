
export CUDA_VISIBLE_DEVICES=2,3

python train.py \
experiments/seg_detector/fakepages_resnet18_deform_thre.yaml \
--resume models/td500_resnet18 \
--num_gpus 2 \
--validate \
