import os
import numpy as np
import pandas as pd

import scipy.signal as sgnl
import scipy.optimize as optm
import scipy.stats as sps

'''
Created Feb 2020
Last edited on Sept 9, 2020

This module may probably also include the shoulder simulation code in the future.
While muver will probably ship with the Shoulder Parameters csv,
it should be here so that the user can generate it.

@author: Kyler Anderson
'''

parameterFile = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '../..', 'shoulder_parameters.txt')
with open(parameterFile, 'r') as datain:
    DF = pd.read_csv(datain)
    
FDF = DF.set_index(keys=['Read Length', 'Mean Fragment Length', 
    'Fragment Length Std. Dev.']).drop(columns=['Unnamed: 0'])

def floorTo(x, b):
    return b * np.floor(x/b)

def ceilTo(x, b):
    return b * np.ceil(x/b)

def bigRollRMSD(x1, x2):
    lp = len(x2)-1
    rmsd = np.full(len(x1)-lp, 0, np.float64)
    for i in range(lp):
        rmsd += (x1[i:-lp+i] - x2[i])**2
    rmsd += (x1[lp:] - x2[lp])**2
    return np.sqrt(rmsd/len(x2))

def describeContainment(xs, lims):
    containingLim = np.full_like(xs, -1, np.int32)
    interpolationAlpha = np.full_like(xs, -1, np.float64)
    
    for i,x in enumerate(xs):
        for j,(l,r) in enumerate(lims):
            if l < x < r:
                containingLim[i] = j
                interpolationAlpha[i] = (x-l)/(r-l)
    
    return containingLim, interpolationAlpha

def interpMuSig(r, m, s):
    h = np.array([50, 50, 25])
    x = np.array([r, m, s])
    a = np.array([floorTo(v, hk) for v, hk in zip(x, h)])
    i = (x-a)/h
    c = np.array(FDF.loc[tuple(a)])
    d = np.array([FDF.loc[tuple(a+np.eye(3)[k]*hk)]
                  for k, hk in enumerate(h)])
    return np.dot((d-c).T, i) + c

def shoulderMaker(x, h, r, m, s):
    f1 = np.piecewise(
        x.astype(np.float64),
        ( (x<-r/2), ((-r/2<=x) & (x<r/2)) ),
        (lambda x: h/2, lambda x: h/2 *(1/2-x/r), lambda x: 0))
    mu, sig = interpMuSig(r, m, s)
    f2 = sps.norm.sf(x, mu-980-r/2, sig) * h/2
    return f1 + f2

def identifyShoulderedRegions(depths, ploidy, readLength=150, rmsdT=0.90, output=None):
    tP, bP = 2*readLength, 4*readLength
    corrspace = 3*readLength
    
    lengths = [len(d) for d in depths]
    slant = np.linspace(1, 0.5, readLength)
    padding = np.full(readLength//2, np.nan)
    
    allShoulders = []
    # nurbs = []
    
    leftShoulders = []
    rightShoulders = []
    
    for i in range(16):
        xr = depths[i]
        rmsdDown = np.concatenate((padding, bigRollRMSD(xr, slant)))
        rmsdUp = np.concatenate((padding, bigRollRMSD(xr, slant[::-1])))
        
        DownSlantMatches, _ = \
            sgnl.find_peaks(1-rmsdDown, height=rmsdT, distance=readLength)
        UpSlantMatches, _ = \
            sgnl.find_peaks(1-rmsdUp, height=rmsdT, distance=readLength)
        
        corr = pd.Series(xr)\
            .rolling(readLength, center=True)\
            .corr(other=pd.Series(np.arange(lengths[i])))
        corr = np.clip(corr, -0.4, 0.4)
        
        _, DownSlopeRegions = \
            sgnl.find_peaks(-corr, height=0.4, plateau_size=corrspace)
        _, UpSlopeRegions = \
            sgnl.find_peaks(corr, height=0.4, plateau_size=corrspace)

        DownRegionIndices, DownRegionAlignments = describeContainment(
            DownSlantMatches,
            list(zip(DownSlopeRegions['left_edges'],
                     DownSlopeRegions['right_edges'])))
        UpRegionIndices, UpRegionAlignments = describeContainment(
            UpSlantMatches,
            list(zip(UpSlopeRegions['left_edges'],
                     UpSlopeRegions['right_edges'])))

        DownShoulders = DownSlantMatches[
            (DownRegionIndices > -1) & \
            (DownRegionAlignments < 0.33)]
        UpShoulders = UpSlantMatches[
            (UpRegionIndices > -1) & \
            (UpRegionAlignments < 0.33)]
        
        leftShoulders.append(DownShoulders)
        rightShoulders.append(UpShoulders)
        
        for x in DownShoulders:
            sh = xr[x-tP:x+bP]
            if len(sh) < 6*readLength: continue
            allShoulders.append(sh)

        for x in UpShoulders:
            sh = xr[x+tP-1:x-bP-1:-1]
            if len(sh) < 6*readLength: continue
            allShoulders.append(sh)
    
    allShoulders = np.atleast_2d(allShoulders)
    allShoulders = allShoulders[(allShoulders[:,-1] < 0.3)]
    medianProfile = np.median(allShoulders, axis=0)
    X = np.arange(-tP, bP)
    popt, _ = optm.curve_fit(shoulderMaker, X, medianProfile, 
        bounds=([0.5, 50, 200, 25], [2, 150, 500, 175]))
    with open(os.path.join(output,\
         'non-unique_sim_param_fit.txt'), 'w') as OUT:
        OUT.write('Height\tRead Length\tFragment Mean\tFragment StD.\n')
        OUT.write('{}\t{}\t{}\t{}'.format(*popt))
    
    nuwhere, nulims = [] ,[]
    for i in range(16):
        nuwhere.append([])
        nulims.append([])
        
        xr = depths[i]
        xrLeftShoulders = leftShoulders[i]
        xrRightShoulders = rightShoulders[i]
        
        if not (len(xrLeftShoulders) and len(xrRightShoulders)): continue
        
        left = xrLeftShoulders[0]
        right = xrRightShoulders[0]
        li, ri = 0, 0
        
        while (ri < len(xrRightShoulders)): # we are here
            right = xrRightShoulders[ri]
            
            if left + 200 < right - 200:
                f = np.sum(xr[left+200:right-200] < 0.1)/(right - left - 400)
                
                if f > 1/2:
                    nuwhere[i] += [left, right]
                    nulims[i].append((left,right))
                    
                while ((left < right) & (li < len(xrLeftShoulders)-1)):
                    li += 1
                    left = xrLeftShoulders[li]
                
            ri += 1
    
    return [np.array(xrnuwhere) for xrnuwhere in nuwhere], \
           [np.array(xrnulims) for xrnulims in nulims]
