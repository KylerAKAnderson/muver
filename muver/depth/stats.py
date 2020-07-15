import pandas as pd
import numpy as np
import scipy.stats as sps

import curves
from util import quietly

'''
Created Jan 2020
Last edited on Jul 15, 2020

'''

def globalSummary(ds):
    allxrs = np.concatenate(ds)
    return np.mean(allxrs), np.median(allxrs), np.std(allxrs)

def rmsd(o,e):
    o = np.array(o)
    e = np.array(e)
    return np.sqrt(sum((o-e)**2)/len(o))

def discreteDerivative(s, w=75, scale=1000):
    s = pd.Series(s)
    X = pd.Series(np.arange(len(s))/scale)
    roll = s.rolling(w,center= True)
    r = roll.corr(other = X)
    sy = roll.std()
    sx = X.rolling(w,center=True).std()
    rslp = (r*(sy/sx))
    rslp[np.isnan(rslp) & \
         (~np.isfinite(r) | \
          ~np.isfinite(r))] = 0
    
    return rslp

def sumUnder(z, Z):
    return np.sum(Z[Z < z])

def rollCorr(s, w=100, scale=1000):
    X = pd.Series(np.arange(len(s))/scale)
    roll = s.rolling(w,center= True)
    return roll.corr(other = X)

class NormNct():
    '''A 2-D distribution, the outer product of a Normal and Noncentral-T, assumed independent.'''
    
    def fit(xData, yData):
        normps = quietly(sps.norm.fit, xData)
        nctps = quietly(sps.nct.fit, yData)
        return NormNct(*normps,*nctps)
    
    def __init__(self, mu, sg, df, nc, lc, sc):
        self.norm = sps.norm(mu, sg)
        self.nct = sps.nct(df=df, nc=nc, loc=lc, scale=sc)
        
        modeL = nc*np.sqrt(df/(df+5/2))
        modeU = nc*np.sqrt(df/(df+1))
        self.nct.modeEst = sc*(modeL + modeU)/2 + lc
        
        self.norm.max = self.norm.pdf(mu)
        self.nct.max = self.nct.pdf(self.nct.modeEst)
        self.max = self.norm.max * self.nct.max
        
    def pdf(self, x, y):
        return self.norm.pdf(x) * \
               self.nct.pdf(y)
               
    def simsApprox2D (self, N):
        (nl, nu) = self.norm.interval(0.9999)
        (tl, tu) = self.nct.interval(0.9999)
        
        XX, YY = np.mgrid[nl:nu:(N+1)*1j,tl:tu:(N+1)*1j]
        Dx = XX[1,0]-XX[0,0]
        Dy = YY[0,1]-YY[0,0]
        ZZ = curves.simsApprox(self.norm.pdf, XX[:-1,:-1], Dx) * \
             curves.simsApprox(self.nct.pdf, YY[:-1,:-1], Dy)
        
        return XX, YY, ZZ, Dx*Dy
    
    def contour(self, z, n=100):
        mu, sg = self.norm.args
        
        nctz = z * sg * np.sqrt(2*np.pi)
        yl = curves.findRoot(self.nct.pdf, nctz, (0,self.nct.modeEst))
        yu = curves.findRoot(self.nct.pdf, nctz, (self.nct.modeEst,1))
        
        hY = np.linspace(yl,yu,n)
        contX = np.sqrt(-2*sg**2*np.log(
            sg*np.sqrt(2*np.pi)*z/self.nct.pdf(hY)))
        Xl = mu - contX
        Xr = mu + contX
        
        X = np.concatenate((Xl,Xr[::-1]))
        X = np.where(np.isnan(X), mu, X)
        Y = np.concatenate((hY,hY[::-1]))
        return X, Y
        