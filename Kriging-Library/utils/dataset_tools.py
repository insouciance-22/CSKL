import pandas as pd
from math import radians, cos, sin, asin, sqrt
import numpy as np
import h5py


def haversine(lng1, lat1, lng2, lat2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    """
    lng1, lat1, lng2, lat2 = map(radians, [lng1, lat1, lng2, lat2])
    # haversine
    dlng = lng2 - lng1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlng / 2) ** 2
    c = 2 * asin(sqrt(a))
    r = 6371.0
    return c * r * 1000


def get_long_lat(sensor_index, locations, lng_idx=2, lat_idx=1):
    lng = locations[lng_idx].loc[sensor_index]
    lat = locations[lat_idx].loc[sensor_index]
    return lng, lat


def get_dist(locations, lng_idx=2, lat_idx=1):
    n = locations.shape[0]
    distance_matrix = np.zeros(shape=(n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                lon1, lat1 = get_long_lat(i, locations, lng_idx, lat_idx)
                lon2, lat2 = get_long_lat(j, locations, lng_idx, lat_idx)
                distance_matrix[i, j] = haversine(lon1, lat1, lon2, lat2)
    return distance_matrix


def kernel_adj(dist, thr=0.1, aqi=False):
    # 把过大的距离设定为inf
    if aqi:
        dist_ = dist[:36, :36].copy()
        sigma = np.std(dist_[~np.isinf(dist_)])
    else:
        sigma = np.std(dist[~np.isinf(dist)])
    adj = np.exp(-np.square(dist / sigma))
    adj[adj < thr] = 0.
    return adj, sigma


def read_h5(path):
    with h5py.File(path, 'r') as f:
        print("文件的所有键：")
        for key in f.keys():
            print(key)