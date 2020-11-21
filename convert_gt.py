
import os
import sys

from shutil import copy
import json
import random


def convert_LTRB_to_poly(bbox):
    left = bbox[0]
    top = bbox[1]
    right = bbox[2]
    bottom = bbox[3]
    return [left, top, right, top, right, bottom, left, bottom]


if __name__ == '__main__':
    imgs_path = 'datasets/book_pages/imgs_vertical'
    gt_path = 'datasets/book_pages/book_pages_tags_vertical_3.txt'

    train_imgs_path = 'datasets/book_pages/train_images'
    train_gts_path = 'datasets/book_pages/train_gts'

    test_imgs_path = 'datasets/book_pages/test_images'
    test_gts_path = 'datasets/book_pages/test_gts'

    train_imgs_list_path = 'datasets/book_pages/train_list.txt'
    test_imgs_list_path = 'datasets/book_pages/test_list.txt'

    total_timgs = os.listdir(imgs_path)
    total_cnt = len(total_timgs)

    test_imgs_list = random.choices(total_timgs, k = total_cnt // 200)
    test_imgs_set = set(test_imgs_list)
    train_imgs_list = [img for img in total_timgs if img not in test_imgs_set]
    train_imgs_set = set(train_imgs_list)

    with open(train_imgs_list_path, 'w', encoding='utf-8') as fp:
        for timg in train_imgs_list:
            fp.write(timg + '\n')

    with open(test_imgs_list_path, 'w', encoding='utf-8') as fp:
        for timg in test_imgs_list:
            fp.write(timg + '\n')

    with open(gt_path, 'r', encoding='utf-8') as fp:
        for line in fp:
            timg, img_json = line.strip().split('\t')
            timg_real_path = os.path.join(imgs_path, timg)
            img_json = json.loads(img_json)
            bboxs = img_json['text_bbox_list']
            texts = img_json['text_list']
            records = []
            for bbox, text in zip(bboxs, texts):
                poly = convert_LTRB_to_poly(bbox)
                poly = list(map(str, poly))
                text = ''.join(text)
                records.append(','.join(poly + [text]))
            if timg in train_imgs_set:
                gt_path = os.path.join(train_gts_path, timg + '.txt')
                dst_path = os.path.join(train_imgs_path, timg)
            elif timg in test_imgs_set:
                gt_path = os.path.join(test_gts_path, timg + '.txt')
                dst_path = os.path.join(test_imgs_path, timg)
            else:
                raise ValueError
            with open(gt_path, 'w', encoding='utf-8') as fp:
                for record in records:
                    fp.write(record + '\n')
            copy(timg_real_path, dst_path)
