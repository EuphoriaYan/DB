

python demo.py \
experiments/seg_detector/td500_resnet18_deform_thre.yaml \
--image_path datasets/chinese_books \
--resume models/td500_resnet18 \
--box_thresh 0.5 \
--polygon \
--visualize \
