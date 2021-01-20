
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
from sklearn.cluster import DBSCAN, KMeans, MeanShift, OPTICS, Birch, SpectralClustering, AgglomerativeClustering


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


def cluster_recs_with_lr(recs, type='DBSCAN'):
    switch = {
        'DBSCAN': DBSCAN(min_samples=1, eps=0.012),
        'MeanShift': MeanShift(bandwidth=0.3),
        'OPTICS': OPTICS(min_samples=1, eps=20),
        'Birch': Birch(n_clusters=None)
    }
    try:
        cluster = switch[type]
    except ValueError as e:
        raise ValueError('type should be DBSCAN, MeanShift, OPTICS or Birch')
    recs_data = [[rec.l, rec.r] for rec in recs]
    recs_data = np.array(recs_data)
    recs_min, recs_max = recs_data.min(), recs_data.max()
    recs_data = (recs_data - recs_min) / (recs_max - recs_min)
    labels = cluster.fit_predict(recs_data)
    # plt.scatter(recs_data[:, 0], recs_data[:, 1], s=1, c=labels)
    # plt.show()
    classified_box_ids = collections.defaultdict(list)
    for idx, label in enumerate(labels):
        classified_box_ids[label].append(idx)
    return classified_box_ids


def cal_dis(x1, y1, x2, y2):
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def cluster_recs_with_width(recs, boxes, type='Birch', n_clusters=None):
    switch = {
        'KMeans': KMeans(n_clusters=n_clusters),
        'Birch': Birch(n_clusters=n_clusters),
        'SpectralClustering': SpectralClustering(n_clusters=n_clusters),
        'AgglomerativeClustering_ward': AgglomerativeClustering(n_clusters=n_clusters, linkage='ward'),
        'AgglomerativeClustering_complete': AgglomerativeClustering(n_clusters=n_clusters, linkage='complete'),
    }
    try:
        cluster = switch[type]
    except ValueError as e:
        raise ValueError('type should be KMeans, Birch, SpectralClustering, AgglomerativeClustering')
    recs_data = [cal_dis(box[0][0], box[0][1], box[1][0], box[1][1]) for box in boxes]
    recs_data = np.array(recs_data).reshape(-1, 1)
    # recs_max = np.max(recs_data)
    # recs_data = recs_data / recs_max * 5
    labels = cluster.fit_predict(recs_data)
    # plt.scatter(recs_data[:], recs_data[:], s=1, c=labels)
    # plt.show()
    classified_box_ids = collections.defaultdict(list)
    for idx, label in enumerate(labels):
        classified_box_ids[label].append(idx)

    return classified_box_ids


def check_one_over_two(cur, nxt, recs, cover_threshold=0.3):
    cur_l = np.mean([i.l for i in cur])
    cur_r = np.mean([i.r for i in cur])
    nxt_l = np.mean([i.l for i in nxt])
    nxt_r = np.mean([i.r for i in nxt])
    cur_len = cur_r - cur_l
    nxt_len = nxt_r - nxt_l
    cover = min(cur_r, nxt_r) - max(cur_l, nxt_l)
    if nxt_len * 1.4 <= cur_len and cover > cover_threshold * nxt_len:
        return True
    else:
        return False


def read_out(classified_recs, recs, cover_threshold, bigger_idx=None):
    output_idx = []
    total_clusters = len(classified_recs)
    for i in range(total_clusters):
        # i - 1 can't be one-column.
        if classified_recs[i]:
            # check if cur cluster is one-column and any of next clusters is two-column
            cur = classified_recs[i]
            # weight = np.mean([0.5 * (i.r - i.l) + i.r for i in cur])
            flag = False
            for rec in cur:
                if rec.idx in bigger_idx:
                    flag = True
                    break
            if not flag:
                while classified_recs[i]:
                    output_idx.append(classified_recs[i].pop(0))
                continue
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


def check_cover(cur, nxt, cover_threshold=0.3):
    cur_len = cur.r - cur.l
    nxt_len = nxt.r - nxt.l
    cover = min(cur.r, nxt.r) - max(cur.l, nxt.l)
    if cover > cover_threshold * nxt_len and cover > cover_threshold * cur_len:
        return True
    else:
        return False


def check_two_column(cur, nxt, cover_threshold=0.3):
    cur_len = cur.r - cur.l
    nxt_len = nxt.r - nxt.l
    cover = min(cur.r, nxt.r) - max(cur.l, nxt.l)
    if cover > cover_threshold * nxt_len and cur_len > nxt_len:
        return True
    else:
        return False


def read_out_2(recs, bigger_idx=None, sort_hp=0.02):
    output_idx = []
    recs = sorted(recs, key=lambda x: x.r - sort_hp * x.u, reverse=True)
    rec_cnt = len(recs)
    vis = [False for _ in range(rec_cnt)]

    while len(output_idx) < rec_cnt:
        # cur = 0
        for i in range(rec_cnt):
            if vis[i]:
                continue
            cur = recs[i]
            vis[i] = True
            break
        # one-column state, find the same position one-column, and find two-column between them.
        if cur.idx in bigger_idx:
            one_column_list = [cur]
            for i in range(rec_cnt):
                if vis[i]:
                    continue
                nxt = recs[i]
                if nxt.idx not in bigger_idx:
                    continue
                if check_cover(cur, nxt, cover_threshold=0.7):
                    one_column_list.append(nxt)
                    vis[i] = True
            two_column_list = []
            for i in range(rec_cnt):
                if vis[i]:
                    continue
                nxt = recs[i]
                if nxt.idx in bigger_idx:
                    continue
                if check_one_over_two(one_column_list, [nxt], recs):
                    two_column_list.append(nxt)
                    vis[i] = True
            while one_column_list or two_column_list:
                cur_nxt = None
                if one_column_list:
                    output_idx.append(one_column_list.pop(0))
                    if one_column_list:
                        cur_nxt = one_column_list[0]
                    else:
                        cur_nxt = None
                if two_column_list:
                    if cur_nxt is not None:
                        new_two_column_list = []
                        for rec in two_column_list:
                            if rec.u < cur_nxt.u:
                                output_idx.append(rec)
                            else:
                                new_two_column_list.append(rec)
                        two_column_list = new_two_column_list
                    else:
                        output_idx.extend(two_column_list)
                        two_column_list = []
        # two-column state, find the cover one-column, and find two-column been covered.
        else:
            one_column_list = []
            two_column_list = [cur]
            for i in range(rec_cnt):
                if vis[i]:
                    continue
                nxt = recs[i]
                if nxt.idx not in bigger_idx:
                    continue
                if check_two_column(nxt, cur):
                    one_column_list.append(nxt)
                    vis[i] = True
            for i in range(rec_cnt):
                if vis[i]:
                    continue
                nxt = recs[i]
                if nxt.idx in bigger_idx:
                    continue
                if one_column_list and check_one_over_two(one_column_list, [nxt], recs):
                    two_column_list.append(nxt)
                    vis[i] = True

            while one_column_list or two_column_list:
                if one_column_list:
                    cur_nxt = one_column_list[0]
                else:
                    cur_nxt = None
                if two_column_list:
                    if cur_nxt is not None:
                        new_two_column_list = []
                        for rec in two_column_list:
                            if rec.u < cur_nxt.u:
                                output_idx.append(rec)
                            else:
                                new_two_column_list.append(rec)
                        two_column_list = new_two_column_list
                    else:
                        output_idx.extend(two_column_list)
                        two_column_list = []
                if one_column_list:
                    output_idx.append(one_column_list.pop(0))
    return output_idx


def list_sort(box_list, cover_threshold=0.6):
    r = np.mean([b.r for b in box_list])
    length = np.mean([b.r - b.l for b in box_list])
    return r + length * cover_threshold


def width_sort(box_list):
    width = np.mean([b.r - b.l for b in box_list])
    return width


def box_sort(box):
    return box.u - box.r


def cv2read(path, mode=cv2.IMREAD_COLOR):
    return cv2.imdecode(np.fromfile(path, dtype=np.uint8), mode)


def cv2save(img, path):
    cv2.imencode(".jpg", img)[1].tofile(path)
