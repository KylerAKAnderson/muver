import os

import numpy as np
import numpy.ma as npma
import matplotlib.pyplot as plt
import matplotlib.ticker as tck
import matplotlib.patches as ptch
from mpl_toolkits.mplot3d import Axes3D

import util

'''
Created Nov 2019
Last edited on Jan 22, 2020

@author: Kyler Anderson
'''

producing = True

opf = 'output'
cSam = 'recent' 
out_form = 'pdf'

def pathTo(n):
    return os.path.join(opf, cSam, n + '.' + out_form)

def geom(cW, rH, ms=(1/2,1/2,1/2,1)):
    cH = cW * rH
    w = cW + ms[0] + ms[2]
    h = cH + ms[1] + ms[3]
    fs = (w,h)
    rect = [ms[0]/w, ms[1]/h, 1-ms[2]/w, 1-ms[3]/h]
    return fs,rect

def vgrid(ax, c=(0.65,0.65,0.65), w=0.2):
    xax = ax.xaxis
    xax.set_minor_locator(tck.AutoMinorLocator())
    xax.grid(which='minor', color=c, linewidth=w, alpha = 0.5)
    xax.grid(which='major', color=c, linewidth=w)

def getPlots(r=1, c=1, d=(6,1), m=[1/2,1/2,1/2,1], **spkws):
    fs, rect = geom(*d, m)
    fig, axs = plt.subplots(r, c, squeeze=True, figsize=fs, **spkws)
    fig.tight_layout(rect=rect)
    
    return fig, axs

def smileCorrection(ds, ks, cs, b=1000):
    if not producing: return
    
    fig, axs = getPlots(8, 2, d=(7,9/7))
    axs = np.hstack(axs)
    title = cSam + ' Smile Correction'
    fig.suptitle(title, fontsize='x-large')
    axs[-1].set_xlabel('Distance (kb)')
    
    # this puts the axes in an order such that iterating
    # through the chromosomes in order results in them sorted by length
    # down the graphic
    permute = np.lexsort(([len(d) for d in ds],))
    axs[permute] = axs
    
    for i,(d,c,k,ax) in enumerate(zip(ds,cs,ks,axs)):
        dbms, cbms, kbms, mxs = [[npma.median(vb) for vb in np.array_split(v,b)]
                                  for v in [d,c,k,np.arange(len(d))/1000]]
        
        ssargs = {'marker':'.','s':1}
        ax.scatter(mxs, dbms, c=[(1,0,0,0.5)], **ssargs)
        ax.scatter(mxs, cbms, c=[(0,0.5,1,0.5)], **ssargs)
        ax.plot(mxs, kbms, c=(0,0.8,0,0.75), lw=1)
        ax.axhline(1, color='k', linewidth=0.5)
        ax.set_ylabel('Chr{}'.format(i+1), fontsize='xx-small')
    
    fileName = cSam + ' Smile Correction Summary'
    save(fig, fileName)

def bulgeCorrection(XYZ, dM, rems, rM, hypa):
    if not producing: return
    
    X, Y, _ = XYZ
    xi, xf = np.min(XYZ[0]), np.max(XYZ[0])
    yi, yf = np.min(XYZ[1]), np.max(XYZ[1])
    n = XYZ[0].shape[1]
    
    fig = plt.figure(figsize=(8,5))
    gs = fig.add_gridspec(1, 2, width_ratios=(3,1),
                          top=0.95, left=0.025, wspace=0.2)
    axl = fig.add_subplot(gs[0,0], projection='3d')
    axr = fig.add_subplot(gs[0,1])
    
    # left axis: 3D plot of data and hyperbolic paraboloid fit
    XX, YY = np.mgrid[xi:xf:16j,yi:yf:n*1j]
    axl.scatter(*XYZ, c=[(0,0,0,0.5)], s=16, marker='|')
    axl.plot_wireframe(XX ,YY, hypa.F(XX, YY), color=(0,0.8,0,0.5))
    
    # right axis: 2D exposition of refinement
    axr.scatter((Y+1)[~dM], X[~dM], c=['r'], marker='o', s=2)
    axr.scatter((Y+1)[dM], X[dM], c=['k'], marker='o', s=2)
    axr.scatter((Y+1)[~rM], X[~rM], c=[(0,0,0,0)], s=25,
                marker='o', edgecolors=[(0,0,1,1)], linewidths=1)
    axr.axvline(n+1, c='k', linewidth=0.5)
    
    axrXticks = np.arange(1,n+1)
    axrXtickLs = np.arange(1,n+1)
    
    if rems:
        nr = len(rems)
        remx = np.linspace(n+1.5,n+2.5,nr)
        remc = util.spectrum(nr,-1/3,0,l=1)
        
        axrXticks = np.hstack((axrXticks,remx))
        axrXtickLs = np.hstack((axrXtickLs,np.arange(1,nr+1)))
        
        for rem, rx, rc in zip(rems, remx, remc):
            for r in rem: 
                axr.scatter(rx, X[:,0][r], c=[rc], s=9, 
                            marker='${}$'.format(r+1),
                            linewidths=0.5)
            
    axr.set_xlim((0,n+3))
    
    # final formatting
    fig.suptitle(cSam + ' BiHypa Process')
    
    axl.set_title('Final Fit')
    axl.set(xlabel='Log Position (log(bp))',
            ylabel='Quantile',
            zlabel='Depth')
    axl.set_yticks(np.arange(n))
    axl.set_yticklabels(np.arange(1,n+1))
    
    axr.set(xlabel='Quantile (Left),\n Removal Round (Right)',
            ylabel='Chromosome (Length Order)')
    axr.set_xticks(axrXticks)
    axr.set_xticklabels(axrXtickLs)
    axr.set_yticks(X[:,0])
    axr.set_yticklabels(np.arange(1,17), 
        rotation='vertical', fontsize='x-small')
    
    save(fig, cSam + ' Bulge Correction Process Summary')

def bivNalpha(meds, stds, z, alpha, X, Y, Z):
    if not producing: return
    
    fig, ax = getPlots(d=(3,1), m=[0.25]*4)
    colors = util.spectrum(16,0,9/12)

    for i in range(1,17):
        ax.scatter(meds.loc[i], stds.loc[i],
            c=[colors[i-1]], s=9, alpha=0.05,
            marker='${}$'.format(i))
    
    sigRegC = util.purple
    cmap = util.binaryMap(sigRegC+(0.02,),(0,0,0,0))
    ax.pcolormesh(X, Y, (Z > z).astype(np.int32), 
                  cmap=cmap, norm=util.binaryNorm)
    
    pcms = ptch.Patch(color=sigRegC+(0.5,), label='Sig. Region')
    ax.legend(handles = [pcms], fontsize='x-small')

    fig.suptitle('Normal Non-central T Bivariate\n over 1000 BP bins')
    ax.set(xlabel='Median', ylabel='Std. Dev.')
    ax.set_xlim(0.25,1.75)
    ax.set_ylim(0,0.4)
    
    save(fig, 'Bivariate Alpha Region')

def histNdist(histData, dist=None, stacked=False, bins=50, title='Histogram'):
    if not producing: return
    """ 
        Plots the cdf and pdf of hist, along with its fit model 'dist'
        if provided. 
    """
    
    fig, (axp, axc) = getPlots(2, d=(3,1.5), m=[0.1]*4, sharex=True)
    
    if stacked:
        n = len(histData)
        pdfcolor = util.tintshade(n, 7/12)
        cdfcolor = util.tintshade(n, 1/12)
        label = ['Chr 1', '...', 'Chr 16']
    else :
        pdfcolor = util.cerulean
        cdfcolor = util.orange
        label = ['Data']
    
    kws = {'density': True, 'bins': bins, 'stacked': stacked}
    _, pdfx, pBxs = axp.hist(histData, color=pdfcolor, **kws)
    _, cdfx, cBxs = axc.hist(histData, color=cdfcolor, 
                             cumulative=True, **kws)
    
    if stacked:
        pRep = [pBxs[0][0], ptch.Rectangle((0,0),1,1,visible=False), pBxs[-1][0]]
        cRep = [cBxs[0][0], ptch.Rectangle((0,0),1,1,visible=False), cBxs[-1][0]]
    else:
        pRep = [pBxs[0]]
        cRep = [cBxs[0]]
        
    if dist:
        pline = axp.plot(pdfx, dist.pdf(pdfx), c=util.blue, linewidth=2)
        label += ['Fit']
        pRep += [pline[0]]
        
        cline = axc.plot(cdfx, dist.cdf(cdfx), c=util.red, linewidth=2)
        cRep += [cline[0]]

    fig.suptitle(title)
    axp.set_ylabel('Point Density')
    axp.legend(pRep, label, fontsize='x-small')
    axc.set_ylabel('Cumulative Density')
    axc.legend(cRep, label, fontsize='x-small')
    
    save(fig, title)

def save(fig, name):
    name = pathTo(cSam+' '+name)
    makeFolderFor(name)
    with open(name,'wb') as out:
        fig.savefig(out, format=out_form)

def makeFolderFor(name):
    parts = name.split(os.sep)[:-1]
    progress = ''
    for p in parts:
        progress += p
        if (not os.path.isdir(progress)):
            os.mkdir(progress)
        progress += os.sep
