
import os
import sys

import argparse
from PIL import Image, ImageDraw
import cv2
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img', type=str, required=True)
    parser.add_argument('--gt', type=str, required=True)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    img = cv2.imread(args.img)
    font = cv2.FONT_HERSHEY_SIMPLEX
    with open(args.gt, 'r', encoding='utf-8') as fp:
        for idx, line in enumerate(fp):
            box = line.strip().split(',')
            box = list(map(int, box))
            box = np.array(box).astype(np.int32).reshape(-1, 2)
            cv2.polylines(img, [box], True, (0, 0, 255), 2)
            cv2.putText(img, str(idx), (int(np.min(box[:, 0])), int(np.mean(box[:, 1]))),
                        font, 1, (0, 255, 255), 2)
        cv2.imshow('img', img)
        cv2.waitKey(0)
