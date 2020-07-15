import time, warnings, logging

import numpy as np
import matplotlib.colors as mplc

'''
Created on Oct 2019
Last Edited Jul 15, 2020

'''

chrlsl03 = np.array([225002, 813149, 312447, 1531659,
                     575719, 270162, 1090861, 562650,
                     439081, 745341, 666447, 1078089,
                     923449, 784290, 1089332, 948058])

cntrsl03 = np.array([[151177, 151294], [238188, 238304], [112730, 112846], [449649, 449759], 
                     [150889, 151006], [148512, 148629], [496909, 497027], [105586, 105703], 
                     [354822, 354938], [435960, 436078], [439783, 439900], [150748, 150867],
                     [268025, 268143], [628714, 628831], [324626, 324744], [555950, 556066]])

chrlsw303 = []

cntrsw303 = np.array([[150835, 150953], [237076, 237193], [109958, 110075], [444388, 444499], 
                      [150889, 151007], [143322, 143440], [491230, 491349], [101479, 101597], 
                      [339414, 339531], [422061, 422180], [439779, 439897], [144151, 144271],
                      [256904, 257023], [620466, 620584], [321149, 321268], [539870, 539987]])

def quietly(func,*args,**kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return func(*args,**kwargs)

def isin(x, a, b):
    return (a < x) & (x < b)


def spectrum(n, ci=0, cc=1, s=1, l=0.85):
    return hsl_to_rgb(np.linspace(ci,cc,n)[:,np.newaxis],s,l)
        
def hsl_to_rgb(h, s=1, l=1):
    z = np.arange(0,3)/3 - h
    z = abs((6*z)%6 - 3) - 1
    z = np.clip(z,0,1)
    z = s*(z-1)+1
    return l*z

binaryNorm = mplc.BoundaryNorm([0,0.5,1],2)
def binaryMap(c1=(0.5,0,1,0.05), c0=(0,0,0,0)):
    return mplc.LinearSegmentedColormap.from_list('Tot', [c0,c1], 2)

def nNaryMap(*colors, name='nNary'):
    n = len(colors)
    cmap = mplc.LinearSegmentedColormap.from_list('nNary', colors, n)
    norm = mplc.BoundaryNorm(np.linspace(0,1,n+1), n)
    return cmap, norm

def tintshade(n,h,tl=0.2,sl=0.2,ltd=False):
    so = 2 - sl
    to = 2 - tl
    out = []
    for i in np.linspace(0,1,n):
        if ltd:
            i = 1-i
        s = np.clip(to*i + tl*(1-i),0,1)
        l = np.clip(sl*i + so*(1-i),0,1)
        out.append(hsl_to_rgb(h,s,l))
    return out

red = (1,0,0)
orange = (1,0.5,0)
yellow = (1,1,0)
lime = (0.5,1,0)
green = (0,1,0)
teal = (0,1,0.5)
cyan = (0,1,1)
cerulean = (0,0.5,1)
blue = (0,0,1)
purple = (0.5,0,1)
magenta = (1,0,1)
fuschia = (1,0,0.5)