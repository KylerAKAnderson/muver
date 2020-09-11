import numpy as np
import scipy.optimize as optm

'''
Created on Nov 2019
Last Edited on Jan 22, 2020

@author: andersonkk
'''

def reduce(f,params):
    def reduced(x):
        return f(x,*params)
    return reduced

def findRoot(func, z=0, args=(), bounds=(None,None)):
    def shifted(x, *args):
        return func(x,*args) - z
    return optm.root_scalar(shifted, args=args, bracket=bounds).root

def simsApprox(f,a,h):
    return h/6*( f(a) + 4*f(a+h/2) + f(a+h) )

class BiHypa():
    """ Two hyperbolic paraboloids sharing a line at xm, defined in terms of 6 points."""

    def getBiParams(zii,zif,zmi,zmf,zfi,zff,
            xi = 0, xm = 0.5, xf = 1, yi = 0, yf = 1):
        return getSwivelParams(zmi,zmf,zii,zif, xm,xi,yi,yf),\
               getSwivelParams(zmi,zmf,zfi,zff, xm,xf,yi,yf)
           
    def __init__(self, xi, xm, xf, yi, yf):
        self.XY = (xi, xm, xf, yi, yf)
        self.xm = xm
        self.Z = (1,1,1,1,1,1)
        self.update(*self.Z)
        
    def update(self, zii, zif, zmi, zmf, zfi, zff):
        Z = (zii, zif, zmi, zmf, zfi, zff)
        self.Z = Z
        self.P = BiHypa.getBiParams(*Z,*self.XY)
        
    def F(self, x, y):
        x = np.atleast_1d(x)
        y = np.atleast_1d(y)
        
        M = x <= self.xm
        res = np.zeros_like(x)
        
        res[M] = swivel(x[M], y[M], *self.P[0])
        res[~M] = swivel(x[~M], y[~M], *self.P[1])
        
        return res if res.shape is not (1,) else res[0]
        
    def __call__(self, xy, zii, zif, zmi, zmf, zfi, zff):
        x,y = xy
        self.update(zii, zif, zmi, zmf, zfi, zff)
        return self.F(x,y)

# Axy + Bx + Cy + D
def swivel(x,y,A,B,C,D):
    return np.array(A*x*y + B*x + C*y + D)

def getSwivelParams(zii,zif,zfi,zff,
        xi = 0, xf = 1, yi = 0, yf = 1):
    Sfx = (zff-zif)/(xf-xi)
    Siy = (zif-zii)/(yf-yi)
    Sfy = (zff-zfi)/(yf-yi)
    
    A = (Sfy-Siy)/(xf-xi)
    B = Sfx - A*yf
    C = Siy - A*xi
    D = zii - swivel(xi,yi,A,B,C,0)
    
    return A,B,C,D
