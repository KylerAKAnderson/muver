import time, warnings, logging

import numpy as np

'''
Created on Oct 2019
Last Edited Sept 10, 2020

@author: andersonkk

'''

def quietly(func,*args,**kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return func(*args,**kwargs)

def folded_quantiles(d, n=5):
    T = np.array_split(d,2*n)
    Q = [np.append(T[i],T[-1-i])
            for i in range(n)]
    return Q

def clip_data(xs, ds, fs, cs, interval=(0.5, 2)):
    masks = [(interval[0] < f) & (f < interval[1]) for f in fs]
    cxs = _clip(xs, masks)
    cds = _clip(ds, masks)
    cfs = _clip(fs, masks)
    ccs = _clip(cs, masks)
    return cxs, cds, cfs, ccs

def _clip(ds, masks):
    return [d[mask] for d, mask in zip(ds, masks)]

def isin(x, a, b):
    return (a < x) & (x < b)

def overlaps(x0, x1, a, b):
    return isin(x0,a,b) | isin(x1,a,b)

def reflat(lst):
    flat = []
    if hasattr(lst, '__iter__'):
        for l in lst:
            flat += reflat(l)
        return flat
    else: return [lst]

def roundTo(x,n):
    return n*round(x/n)