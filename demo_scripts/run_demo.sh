
export CUDA_VISIBLE_DEVICES=0

python demo.py \
experiments/seg_detector/fakepages_resnet18_deform_thre.yaml \
--image_path datasets/chinese_books \
--resume models/model_epoch_57_minibatch_36000 \
--box_thresh 0.5 \
--visualize \
