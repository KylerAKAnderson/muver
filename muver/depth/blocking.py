import gc

import numpy as np
import pandas as pd
import scipy.signal as sgnl

from . import util, stats, curves

'''
Created Feb 2020
Last edited on Jun 22, 2020
'''

DFOpts = {'keys': np.arange(1,17), 'names':['Chr Num']}

def pkblocks(depths, gmed, gstd, ploidy, b=1000, saBins=1000, pkAlpha=0.05, **pkOpts):
    ls = np.array([len(d) for d in depths])
    ns = ls//b 
    blocks = [summarizeBlocks(np.array_split(d, n)) 
              for d, n in zip(depths, ns)]
    
    blocks = pd.concat(blocks, **DFOpts)
    
    os = (ls//ns)//2
    offsetblocks = [summarizeBlocks(np.array_split(d[o:-o+1], n-1), o)
             for d, n, o in zip(depths, ns, os)]
    offsetblocks = pd.concat(offsetblocks, **DFOpts)
    
    meds, stds = blocks.Median, blocks.Std
    medM = np.abs(meds - gmed) < 3*gstd
    stdM = util.isin(stds, 0.01/ploidy, 1/ploidy)
    bivTools = getBiv(meds, stds, medM, stdM, saBins)
    
    pks, _, _ = peaksInSlope(depths, **pkOpts)
    pksm = [np.full(len(xrpks), False, np.bool) 
             for xrpks in pks]
    
    def addToMask(i, j, L, R):
        pksm[i-1][util.isin(pks[i-1], L, R)] = True
    
    inOutliers(blocks, bivTools, addToMask, pkAlpha)
    inOutliers(offsetblocks, bivTools, addToMask, pkAlpha)
    sigpks = [xrpks[xrpksm] for xrpks, xrpksm in zip(pks, pksm)]
    
    where = sigpks
    sigpkblocks = [summarizeBlocks(np.split(d, xrw))
                   for d, xrw in zip(depths, where)]
    sigpkblocks = pd.concat(sigpkblocks, **DFOpts)
    sigpkblocks = addPloidyEst(
        sigpkblocks,
        ploidy,
        bivTools[0].norm.args[0],
        bivTools[0].nct.modeEst)
    
    return sigpkblocks, sigpks, bivTools
    
def getBiv(meds, stds, medM=None, stdM=None, saBins=1000):
    if medM is None: 
        medM = np.abs(meds - 1) < 0.5
    if stdM is None: 
        stdM = util.isin(stds, 0.01, 0.5)
    
    mmeds = np.array(meds[medM], dtype=np.float64)
    mstds = np.array(stds[stdM], dtype=np.float64)
    biv = stats.NormNct.fit(mmeds, mstds)
    X, Y, Z, d2V = biv.simsApprox2D(saBins)
    def alphaToZ(alpha):
        return curves.findRoot(
            stats.sumUnder, alpha, (Z,), 
            bounds=(0,np.max(Z)))
    alphaToZ.Zs = Z
    
    return biv, d2V, alphaToZ
    
def summarizeBlocks(blocks, offset=0, 
        summarize=lambda x: [np.median(x), np.std(x), np.max(x)-np.min(x)],
        labels=['Median', 'Std', 'Height']):
    dets = [summarize(block) for block in blocks]
    dets = pd.DataFrame(dets, columns=labels)
    sizes = [len(block) for block in blocks]
    ends = np.hstack((0, np.cumsum(sizes))) + offset
    return dets.assign(Left=ends[:-1], Right=ends[1:], Size=sizes)

def inOutliers(blocks, bivInfo, func, alpha=0.05):
    biv, d2V, a2Z = bivInfo
    z = a2Z(alpha) / d2V
    meds = blocks.Median.to_numpy(dtype=np.float64)
    stds = blocks.Std.to_numpy(dtype=np.float64)
    outlierMask = biv.pdf(meds, stds) < z
    
    for (xrn, blockn), L, R in blocks[outlierMask][['Left', 'Right']].itertuples():
        func(xrn, blockn, L, R)
    
def addPloidyEst(blocks, ploidy, mu, mo):
    meds = blocks.Median.to_numpy(dtype=np.float64)
    stds = blocks.Std.to_numpy(dtype=np.float64)
    medPl = np.round(meds*ploidy/mu).astype(np.int32)
    stdPl = np.round(stds*ploidy/mo).astype(np.int32)
    return blocks.assign(Med_Pl=medPl, Std_Pl=stdPl)

def peaksInSlope(depths, medWin=200, slpWin=200, height=None,
        width=200, rel_height=0.95, **pkOpts):
    pkOpts.update(height=height, width=width, rel_height=rel_height)
    if pkOpts.pop('height', None) is None:
        _, _, std = stats.globalSummary(depths)
        pkOpts['height'] = 3*std
    
    xrms = [pd.Series(d) for d in depths]
    rMeds = [xrm.rolling(medWin, center=True).median() for xrm in xrms]
    rSlps = [stats.discreteDerivative(rMed, slpWin) for rMed in rMeds]
    pks = [sgnl.find_peaks(rSlp.abs(), **pkOpts)[0] for rSlp in rSlps]
        
    return pks, rMeds, rSlps
