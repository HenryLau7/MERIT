# -*- coding:utf-8 -*-
import numpy as np
from skimage import filters, measure

def find_central(img):
    thres = filters.threshold_otsu(img)
    mask = remove_noise(img>thres)
    mask[:5], mask[-5:], mask[:,:5], mask[:,-5:] = 0, 0, 0, 0
    x, y = np.where(mask)
    if len(x) == 0:
        return (0, 0)
    l,r,b,u = np.min(x), np.max(x), np.min(y), np.max(y)
    return (int(b/4*3+u/4),int((l+r)/2))

def remove_noise(mask):
    labels, num = measure.label(mask,background=0, return_num=True)
    res = np.zeros_like(labels)
    min_a = 1
    for i in range(1,num+1):
        area = np.sum(labels==i)
        if area >= min_a:
            res += (labels==i).astype(np.uint8)
    return res
