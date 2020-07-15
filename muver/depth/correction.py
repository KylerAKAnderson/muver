from itertools import (
    combinations as com, 
    chain as plus, 
    product as cross
)

import numpy as np
import numpy.ma as npma
from scipy.optimize import curve_fit

import expositor as xpst
from util import quietly
from stats import rmsd
from curves import BiHypa

'''
Created Oct 2019
Last Edited Jun 22, 2020
'''

# disconnected from former kkit.sample system
# Send in masked arrays, mask by threshold first, returns corrected_depths, calls to expositor
def correctSmile(depths, sml=250000, statn=500, fitn=20):
    
    if not sml%(statn * fitn): sml, statn, fitn = 250000, 500, 20
    ds = [d for i,d in enumerate(depths) if i not in [0, 5, 2, 8, 11]]
    
    ends = np.zeros((2*len(ds), sml))
    for i, d in enumerate(ds):
        ends[2*i] = d[:sml]
        ends[2*i+1] = d[:-sml-1:-1]
    ends = ends.reshape((2*len(ds), -1, statn))
    ends = quietly(npma.median, ends, axis=(0, 2))
    
    # patches nans in the profile with last non nan value (highly unlikely to be a problem)
    for i in npma.nonzero(np.isnan(ends))[0]: 
        if (i == 0):
            ends[0] = ends[np.nonzero(~np.isnan(ends))[0][0]]
        ends[i] = ends[i-1]
    
    ends = ends.reshape((-1, fitn)).T
    fitx = np.arange((statn-1)/2, statn*fitn, statn)
    coefs = np.polyfit(fitx, ends, 1)
    
    longx = np.arange(statn*fitn)
    smirk = np.array([np.poly1d(co)(longx) for co in coefs.T]).ravel()
    smirk[:20000] = smirk[20000]
    
    def stitch(l):
        k = np.ones(l)
        fl = min(sml,int(l/2))
        if (2*sml < l):
            k[sml:-sml] = smirk[-1]
        k[:fl] = smirk[:fl]
        k[-fl:] = smirk[fl-1::-1]
        return k

    fit_smiles = [stitch(len(d)) for d in depths]
    corrected_depths = [npma.array(d.data/fs, mask=d.mask)
                        for d, fs in zip(depths, fit_smiles)]
    
    #xpst.smileCorrection(depths, fit_smiles, corrected_depths)
    return corrected_depths

def correctBulge(depths, n=5, rmsdT=0.05, resT=0.5):
    
    lengths = np.array([len(d) for d in depths])
    X = np.tile(np.log10(lengths), (n, 1)).T
    Y = np.tile(np.arange(n), (16, 1))
    
    Z = []
    for d in depths:
        split = np.array_split(d, 2*n)
        split = [np.append(split[i], split[-1-i]) for i in range(n)]
        split = [quietly(npma.median, q) for q in split]
        Z.append(split)
        
    Z = np.array(Z)
    M = Z > 0.1
    M[11,:] = False
    
    XYZ = np.array([X,Y,Z])
    
    xi, xm, xf = X[(0,4,3),0]
    biHypa = BiHypa(xi, xm, xf, 0, n-1)
    RMSD = fitBiHypa(*XYZ[:, M], biHypa)
    
    rems = []
    if RMSD >= rmsdT:
        RMSD, rems = refineHypa(XYZ, M, biHypa, RMSD, rmsdT)
        
    residuals = biHypa.F(X, Y) - Z
    resMask = np.abs(residuals) < resT
    fitBiHypa(*XYZ[:, resMask & M], biHypa)
    
    def stitch(l):
        half = l//2
        X = np.ones(half) * np.log10(l)
        Y = np.linspace(0, n, half) - 0.5
        k = np.ones(l)
        k[:half] = biHypa.F(X,Y)
        k[-half:] = k[half-1::-1]
        if l%2: k[half] = k[half-1]
        return k
    
    fit_bulge = [stitch(l) for l in lengths]
    corrected_depths = [npma.array(d.data/fb, mask=d.mask)
                        for d, fb in zip(depths, fit_bulge)]
    
    #xpst.bulgeCorrection(XYZ, M, rems, resMask, biHypa)
    return corrected_depths, biHypa.Z

def fitBiHypa(X, Y, Z, hypa):
    curve_fit(hypa, [X, Y], Z)
    return rmsd(Z, hypa.F(X, Y))

def reflat(lst):
    flat = []
    if hasattr(lst, '__iter__'):
        for l in lst:
            flat += reflat(l)
        return flat
    else: return [lst]

def refineHypa(XYZ, M, biHypa, cRMSD, rmsdT=0.05):
    xrM = ~np.all(~M, axis=1)
    remLim = 6 - sum(~xrM) # Always <= 5 since chr12 is manually cut
    
    left, right = np.array([0, 5, 2, 8, 7, 4]), np.array([9, 10, 12, 1, 13, 15, 14, 6, 3, 11])
    left = left[xrM[left,],]
    right = right[xrM[right,],]
    
    avL = max(0, len(left) - 4)
    avR = max(0, len(right) - 4)

    def splitChoose(r):
        if (avL+avR) < r: return []
        rc = []
        if r == 1:
            if avL > 0: rc = com(left, 1)
            if avR > 0: rc = plus(rc, com(right, 1))
        elif r < 6:
            if avL == 2: rc = cross(com(left, 2), com(right, r-2))
            if avL >= 1 and avR >= r-1: rc = plus(rc, cross(left, com(right, r-1)))
            if avR >= r: rc = plus(rc, com(right, r))
        return rc
    
    def tryRemove(rem):
        rem = reflat(rem)
        M2 = M.copy()
        M2[rem, :] = False
        return fitBiHypa(*XYZ[:, M2], biHypa), rem
    
    rem = []
    rems = []
    for i in range(1, remLim+1):
        pRMSD, pZ = cRMSD, biHypa.Z
        
        try: 
            cRMSD, rem = min([tryRemove(rm) for rm in splitChoose(i)],
                key=lambda f: f[0])
        except:
            pass
        
        if cRMSD >= pRMSD: # Didn't Improve
            biHypa.update(*pZ)
            return pRMSD, rems
        
        rems.append(rem)
        if cRMSD <= rmsdT: # Good Enough
            return cRMSD, rems
    
    # Exhausted Tries
    return cRMSD, rems
