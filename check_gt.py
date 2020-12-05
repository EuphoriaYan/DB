
import os
import sys

import argparse
from PIL import Image, ImageDraw


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img', type=str, required=True)
    parser.add_argument('--gt', type=str, required=True)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    img = Image.open(args.img).convert('RGB')
    draw = ImageDraw.Draw(img)
    with open(args.gt, 'r', encoding='utf-8') as fp:
        for line in fp:
            line = line.strip().split(',')
            line = list(map(float, line))
            x = line[:-1:2]
            y = line[1::2]
            score = line[-1]
            for i in range(4):
                nxt = (i + 1) % 4
                draw.line([(x[i], y[i]), (x[nxt], y[nxt])]
                    , fill=(255, 0, 0), width=2)
            img.show()
