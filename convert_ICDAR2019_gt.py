# -*- encoding: utf-8 -*-
__author__ = 'Euphoria'

import os
import sys

from shutil import copy
import json
import random
import argparse

import xml
from xml.dom.minidom import parse


def convert_LTRB_to_poly(bbox):
    left = bbox[0]
    top = bbox[1]
    right = bbox[2]
    bottom = bbox[3]
    return [left, top, right, top, right, bottom, left, bottom]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('imgs_path', type=str)
    parser.add_argument('gt_path', type=str)
    args = parser.parse_args()
    return args


IMG_EXT = {'.jpg', '.png', '.tif', '.tiff', '.bmp', '.gif'}


def parse_xml(xml_file):
    gt = []
    domTree = parse(xml_file)
    rootNode = domTree.documentElement
    pageNode = rootNode.getElementsByTagName('Page')[0]
    textRegionNodes = pageNode.getElementsByTagName('TextRegion')
    for textRegionNode in textRegionNodes:
        textLineNodes = textRegionNode.getElementsByTagName('TextLine')
        for textLineNode in textLineNodes:
            coords = textLineNode.getElementsByTagName('Coords')[0].getAttribute('points')
            label = textLineNode.getElementsByTagName('TextEquiv')[0].getElementsByTagName('Unicode')[0].firstChild.data
            gt.append((coords, label))
    return gt


if __name__ == '__main__':

    args = parse_args()

    imgs_path = args.imgs_path
    gt_path = args.gt_path

    if imgs_path.endswith('/') or imgs_path.endswith('\\'):
        imgs_path = imgs_path[:-1]

    imgs_root = imgs_path
    print('image root: {}'.format(imgs_root))

    train_imgs_path = os.path.join(imgs_root, 'train_images')
    train_gts_path = os.path.join(imgs_root, 'train_gts')
    os.makedirs(train_imgs_path, exist_ok=True)
    os.makedirs(train_gts_path, exist_ok=True)
    print('train imgs path: {}'.format(train_imgs_path))
    print('train gts path: {}'.format(train_gts_path))

    test_imgs_path = os.path.join(imgs_root, 'test_images')
    test_gts_path = os.path.join(imgs_root, 'test_gts')
    os.makedirs(test_imgs_path, exist_ok=True)
    os.makedirs(test_gts_path, exist_ok=True)
    print('test imgs path: {}'.format(test_imgs_path))
    print('test gts path: {}'.format(test_gts_path))

    train_imgs_list_path = os.path.join(imgs_root, 'train_list.txt')
    test_imgs_list_path = os.path.join(imgs_root, 'test_list.txt')

    total_timgs = []

    for root, dirs, files in os.walk(imgs_path):
        for file in files:
            filename, ext = os.path.splitext(file)
            if ext in IMG_EXT:
                total_timgs.append(os.path.join(root, file))

    total_cnt = len(total_timgs)

    test_imgs_list = random.choices(total_timgs, k=total_cnt // 100)
    test_imgs_set = set(test_imgs_list)
    train_imgs_list = [img for img in total_timgs if img not in test_imgs_set]
    train_imgs_set = set(train_imgs_list)

    with open(train_imgs_list_path, 'w', encoding='utf-8') as fp:
        for timg in train_imgs_list:
            fp.write(os.path.basename(timg) + '\n')

    with open(test_imgs_list_path, 'w', encoding='utf-8') as fp:
        for timg in test_imgs_list:
            fp.write(os.path.basename(timg) + '\n')

    for timg in total_timgs:
        gt_file = os.path.join(gt_path, os.path.splitext(os.path.basename(timg))[0] + '.xml')
        timg_real_path = os.path.join(imgs_path, timg)
        gt_xml = parse_xml(gt_file)
        records = []
        for bbox, text in gt_xml:
            poly = ','.join(bbox.split(' '))
            # poly = convert_LTRB_to_poly(bbox)
            # poly = list(map(str, poly))
            records.append(poly+','+text)
        if timg in train_imgs_set:
            timg = os.path.basename(timg)
            dst_gt_path = os.path.join(train_gts_path, timg + '.txt')
            dst_path = os.path.join(train_imgs_path, timg)
        elif timg in test_imgs_set:
            timg = os.path.basename(timg)
            dst_gt_path = os.path.join(test_gts_path, timg + '.txt')
            dst_path = os.path.join(test_imgs_path, timg)
        else:
            raise ValueError
        with open(dst_gt_path, 'w', encoding='utf-8') as fp:
            for record in records:
                fp.write(record + '\n')
        copy(timg_real_path, dst_path)


