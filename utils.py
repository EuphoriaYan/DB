
# -*- coding: utf-8 -*-
import itertools
from functools import reduce

import numpy as np
import cv2
import math
import collections
from PIL import Image
from typing import List
from matplotlib import pyplot as plt
from sklearn.cluster import DBSCAN, KMeans, MeanShift, OPTICS, Birch


# (n, 2) poly -> box_dict
def trans_poly_to_rec(idx, poly):
    # poly = np.array(poly).reshape(-1, 2)
    poly = poly.tolist()
    l = min([i[0] for i in poly])
    r = max([i[0] for i in poly])
    u = min([i[1] for i in poly])
    d = max([i[1] for i in poly])
    Rec = collections.namedtuple('Rec', 'l r u d idx')
    rec = Rec(l, r, u, d, idx)
    return rec


def cluster_recs(recs, type='DBSCAN'):
    switch = {
        'DBSCAN': DBSCAN(min_samples=1, eps=10),
        'MeanShift': MeanShift(bandwidth=0.3),
        'OPTICS': OPTICS(min_samples=1, eps=20),
        'Birch': Birch(n_clusters=None)
    }
    try:
        cluster = switch[type]
    except ValueError as e:
        raise ValueError('type should be DBSCAN, MeanShift, OPTICS or Birch')
    boxes_data = [[rec.l, rec.r] for rec in recs]
    boxes_data = np.array(boxes_data)
    labels = cluster.fit_predict(boxes_data)
    '''
    plt.scatter(boxes_data[:, 0], boxes_data[:, 1], s=1, c=labels)
    plt.show()
    '''
    classified_box_ids = collections.defaultdict(list)
    for idx, label in enumerate(labels):
        classified_box_ids[label].append(idx)
    return classified_box_ids


def check_one_over_two(cur, nxt, recs, cover_threshold):
    cur_l = np.mean([i.l for i in cur])
    cur_r = np.mean([i.r for i in cur])
    nxt_l = np.mean([i.l for i in nxt])
    nxt_r = np.mean([i.r for i in nxt])
    cur_len = cur_r - cur_l
    nxt_len = nxt_r - nxt_l
    cover = min(cur_r, nxt_r) - max(cur_l, nxt_l)
    if nxt_len * 1.4 <= cur_len <= nxt_len * 2.5 and cover > cover_threshold * nxt_len:
        return True


def read_out(classified_recs, recs, cover_threshold=0.2):
    output_idx = []
    total_clusters = len(classified_recs)
    for i in range(total_clusters):
        # i - 1 can't be one-column.
        if classified_recs[i]:
            # check if cur cluster is one-column and any of next clusters is two-column
            cur = classified_recs[i]
            nxt_list = []
            for j in range(1, 5):
                if i + j > total_clusters - 1:
                    break
                nxt = classified_recs[i + j]
                if nxt and check_one_over_two(cur, nxt, recs, cover_threshold):
                    nxt_list.append(nxt)
            if not nxt_list:
                while classified_recs[i]:
                    output_idx.append(classified_recs[i].pop(0))
                    continue
            while cur or reduce(lambda x, y: x or y, nxt_list, False):
                if not cur:
                    for nxt in nxt_list:
                        while nxt:
                            output_idx.append(nxt.pop(0))
                    break
                cur_u = cur[0].u
                for nxt in nxt_list:
                    while nxt and nxt[0].u < cur_u:
                        output_idx.append(nxt.pop(0))
                output_idx.append(cur.pop(0))
        else:
            continue
    return output_idx


def list_sort(box_list, cover_threshold=0.3):
    r = np.mean([b.r for b in box_list])
    length = np.mean([b.r - b.l for b in box_list])
    return r + length * cover_threshold


def box_sort(box):
    return box.u - box.r