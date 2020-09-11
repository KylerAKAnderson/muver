import os.path as pth

import numpy as np
import numpy.ma as npma
import pandas as pd
import scipy.stats as sps

import correction as crct
import blocking as blkg
import curves
from stats import globalSummary
from shoulder import identifyShoulderedRegions

"""Top-level package for Smile and Bulge Depth Correction Extension to Muver."""

__author__ = """Kyler Anderson"""
__email__ = 'andersonkk@nih.gov'

def correct_depths(depthsByName, sample_name, ploidy, 
  output_folder='depth correction output', options={}):
    
    # Initialization
    xpst.cSam = sample_name
    if not output_folder:
        xpst.producing = False
    else: xpst.opf = output_folder
    
    # Preparation
    depthsKey = [(name, d, len(d)) for name, d in depthsByName.items()]
    depthsKey = sorted(depthsKey, key=lambda x: x[2])
    depthsKey = depthsKey[1:] # exclude the short mitochondrial genome
    depths = [key[1] for key in depthsKey]
    
    rDNAxrm = None
    rDNAi = None
    rDNAf = None
    if 'rDNAxrm' in options:
        rDNAxrm = next((i for i,(name,_,_) in enumerate(depthsKey)
                    if name == options['rDNAxrm']), None)
        if 'rDNAreg' in options:
            reej = options['rDNAreg']
            reeji, reejf = reej.split(',')
            rDNAi = int(reeji.strip())
            rDNAf = int(reejf.strip())
    
    globalmean = globalSummary(depths)[0]
    depths = [d/globalmean for d in depths]
    lengths = np.array([len(d) for d in depths])
    
    # Masking
    stdsms = [np.std(d[l//8:7*l//8]) for d, l in zip(depths, lengths)]
    cut = max(0.1, 1 - 5*np.amin(stdsms))
    
    depths = [npma.masked_less(d, cut) for d in depths]
    gmean, gmed, gstd = globalSummary(depths)
    
    pkblocks, _, bivTools = blkg.pkblocks(depths, gmed, gstd, ploidy)
    outlierMasks = [np.full(l, False, np.bool) for l in lengths]
    def addToMask(xrn, blckn, L, R):
         outlierMasks[xrn-1][L:R] = True
    blkg.inOutliers(pkblocks, bivTools, addToMask,
                    options.get('outlier_alpha', 0.05))
    
    for d, oM in zip(depths, outlierMasks):
        d.mask |= oM
    
    # Correction
    if rDNAxrm: rDNAxrmO = depths[rDNAxrm].data.copy()
    
    depths = crct.correctSmile(
        depths, 
        int(options.get('sml', 250000)), 
        int(options.get('statn', 500)), 
        int(options.get('fitn', 20)))
    
    if rDNAxrm: rDNAxrmS = depths[rDNAxrm].data.copy()
    
    depths, Zs = crct.correctBulge(
        depths,
        int(options.get('n', 5)), 
        float(options.get('rmsdT', 0.05)), 
        float(options.get('resT', 0.5))/ploidy)
    
    if rDNAxrm:
        depths[rDNAxrm], rDNArepeats \
            = rDNAFix(rDNAxrmO, rDNAxrmS.data, lengths,
                   Zs, int(options.get('n', 5)), rDNAi, rDNAf)
        if output_folder:
            with open(pth.join(output_folder, 'rDNA_fix_info.txt'), 'w') as OUT:
                OUT.write('Estimated rDNA repeats on {}: {}'.format(rDNAxrm, rDNArepeats))
    
    # Reversion
    depths = [(d.data*globalmean).astype(np.int64) for d in depths]
    depthsOut = dict()
    depthsOut.update(depthsByName)
    for i, (name, _, _) in enumerate(depthsKey):
        depthsOut[name] = depths[i]
    
    return depthsOut
    
def rDNAFix(rDNAxrmO, rDNAxrmS, lengths, Zs, n, ri, rf):
    dy = Zs[5] - Zs[3]
    dx = np.log10(lengths[5]/lengths[-1])
    mq5 = dy/dx
    dlink = np.max(rDNAxrmO[ri:rf])
    nlink = int((dlink + 13.434*mq5 + 1)/(0.7973*mq5 + 1))
    rDNArepeats = nlink+1

    # determine final length
    l0 = len(rDNAxrmO)
    lul = 8222 + 915 # rDNAunit = 8222, rDNAlink = 915 # these should probably be made options as well
    dl = (nlink-1)*lul
    lf = l0 + dl

    # recreate hypa
    xi, xm, xf = np.log10(lengths[(0,5,15),])
    yi, yf = 0, n-1

    biHypa = curves.BiHypa(xi, xm, xf, yi, yf)
    biHypa.update(*Zs)

    # recreate stitcher
    def stitch(l):
        half = l//2
        X = np.ones(half) * np.log10(l)
        Y = np.linspace(0, n, half) - 0.5
        k = np.ones(l)
        k[:half] = biHypa.F(X,Y)
        k[-half:] = k[half-1::-1]
        if l%2: k[half] = k[half-1]
        return k

    # create correction, clip relevant middle, apply
    k = stitch(lf)
    k = np.concatenate((k[:ri], k[ri+dl:]))
    
    return rDNAxrmS/k, rDNArepeats

def describe_regions(depthsByName, ploidy, sample, cnv_bedgraph,
                     filter_bed, distribution_info=None, 
                     output_folder='blocking output', options={}):
    depths = list(depthsByName.values())
    depths = sorted(depths, key=lambda x: len(x))
    depths = depths[1:]
                   
    gmean, gmed, gstd = globalSummary(depths)
    depths = [d/gmean for d in depths]
    lengths = [len(d) for d in depths]
    
    nuwhere, nulims = identifyShoulderedRegions(
        depths,
        ploidy,
        options.get('readLength', 150),
        options.get('slant_rmsdT', 0.90),
        output_folder)
    nuMs = [np.full(l, False, np.bool) for l in lengths]
    for xrnuM, xrnulims in zip(nuMs, nulims):
        for (left, right) in xrnulims:
            xrnuM[left:right] = True
    
    nulims = [xrnulim.T for xrnulim in nulims]
    
    # initial blocks from significant slope peaks in noteable areas
    pkblocks, pks, bivTools = blkg.pkblocks(
        depths,
        gmed,
        gstd,
        ploidy,
        pkAlpha = options.get('outlier_alpha', 0.05),
        width=options.get('peak_width', 100),
        distance=options.get('peak_spread',200)
    )
    biv = bivTools[0]
    mu = biv.norm.args[0]
    mo = biv.nct.modeEst
    
    if distribution_info:
        with open(distribution_info, 'w') as OUT:
            OUT.write('Normal Distribution of block Medians\n')
            OUT.write('mu\t{}\nsig\t{}\n\n'.format(biv.norm.args[0], biv.norm.args[1]))
            
            OUT.write('Non-central T Distribution of block Std. Deviations\n')
            OUT.write('loc\t{2}\nscale\t{3}\ndf\t{0}\nnc\t{1}'\
                      .format(*biv.nct.kwds.values()))
                      
    # incorporate non-unique regions
    where = []
    for i in range(16):
        xrpks = pks[i]

        xrnuM = nuMs[i]
        upkM = xrnuM[xrpks]
        upks = xrpks[upkM == 0]

        xrnuwhere = nuwhere[i]
        xrwhere = np.sort(np.union1d(upks, xrnuwhere))
        where.append(xrwhere.astype(np.int64))

    blockQls = []
    for xrwhere, xrlims in zip(where, nulims):            
        xrnuIs, xrnuFs = [], [] if not len(xrlims) else xrlims
        xrblqls = np.full(len(xrwhere)+1, False, np.bool)
        xrblqls[1:] = np.isin(xrwhere, xrnuIs)
        xrblqls[:-1] = np.isin(xrwhere, xrnuFs)
        blockQls.append(np.where(xrblqls, 'Non-unique', 'Unique'))

    blockQls = np.concatenate(blockQls)

    qdblocks = [blkg.summarizeBlocks(np.split(d, xrw))
                for d, xrw in zip(depths, where)]
    qdblocks = pd.concat(qdblocks, **blkg.DFOpts)
    qdblocks = blkg.addPloidyEst(qdblocks, ploidy, mu, mo)
    qdblocks = qdblocks.assign(Quality=blockQls)
    
    # welche's t, merge statistically not different
    ts = []
    ps = []
    for i in range(16):
        xrblocks = qdblocks.loc[i+1]
        if len(xrblocks) < 2:
            ts.append([])
            ps.append([])
            continue
        meds = xrblocks.Median.to_numpy()
        Ns = xrblocks.Size.to_numpy()
        stds = xrblocks.Std.to_numpy() / np.sqrt(np.pi/(2*Ns))
        xrts, xrps = util.quietly(sps.ttest_ind_from_stats,
            meds[:-1], stds[:-1], Ns[:-1],
            meds[1:], stds[1:], Ns[1:], 
            equal_var=False)
        ts.append(xrts)
        xrps = np.where(np.isfinite(xrps), xrps, 0)
        ps.append(xrps)
    
    where0 = where
    where = []
    for i in range(16):
        xrqs = qdblocks.loc[i+1].Quality.to_numpy()
        xrps = ps[i]
        if not len(ps[i]):
            where.append([])
            continue
        xrbndM = (xrps < 0.05) | \
                 (xrqs[:-1] == 'Non-unique') | \
                 (xrqs[1:] == 'Non-unique')
        xrwhere = where0[i][xrbndM]
        where.append(xrwhere)

    blockQls = []
    for xrwhere, xrlims in zip(where, nulims):
        xrnuIs, xrnuFs = [], [] if not len(xrlims) else xrlims
        xrblqls = np.full(len(xrwhere)+1, False, np.bool)
        xrblqls[1:] = np.isin(xrwhere, xrnuIs)
        xrblqls[:-1] = np.isin(xrwhere, xrnuFs)
        blockQls.append(np.where(xrblqls, 'Non-unique', 'Unique'))

    blockQls = np.concatenate(blockQls)

    tblocks = [blkg.summarizeBlocks(np.split(d, xrw))
               for d, xrw in zip(depths, where)]
    tblocks = pd.concat(tblocks, **blkg.DFOpts)
    tblocks = blkg.addPloidyEst(tblocks, ploidy, mu, mo)
    tblocks = tblocks.assign(Quality=blockQls)
    
    # clasify resulting regions, guess ploidies
    ploidies = []
    probablyCNVs = []
    probablyNoise = []
    
    smallSpec = options.get('small_region', 500) # this should be roughly fragment size

    for i in range(16):
        xrblocks = tblocks.loc[i+1]

        xrploidy = np.round(np.median(depths[i][depths[i] > 0.1]) * ploidy)
        
        ploidies.append(xrploidy) # This is not currently output anywhere

        pl = xrblocks.Med_Pl.to_numpy()
        n = xrblocks.Size.to_numpy()
        ql = xrblocks.Quality.to_numpy()

        # Places that match genome ploidy
        canonPloid = pl == ploidy

        # Non-unique regions
        nonUnique = ql == 'Non-unique'
        
        # Little Divets
        lilDiv = n < smallSpec
        
        if len(pl) > 3:
            # small, 0 regions between non-uniques
            wedged = (pl[1:-1] == 0) &\
                     (n[1:-1] <= smallSpec) &\
                     (ql[:-2] == 'Non-unique') &\
                     (ql[2:] == 'Non-unique')
            wedged = np.concatenate(([False], wedged, [False]))
            
            # particularly wide shoulder step blocks
            slope = (n[1:-1] <= smallSpec) &\
                    (pl[:-2] != pl[2:]) &\
                    (((pl[:-2] >= pl[1:-1]) & (pl[1:-1] >= pl[2:])) |\
                     ((pl[:-2] <= pl[1:-1]) & (pl[1:-1] <= pl[2:])))
            slope = np.concatenate(([False], slope, [False]))
        else:
            wedged = np.full(len(pl), False, np.bool)
            slope = np.full(len(pl), False, np.bool)
        
        xrProbableCNVs = ~(canonPloid | nonUnique | wedged | slope | lilDiv)
        probablyCNVs.append(xrProbableCNVs)
        
        xrNoise = wedged | slope
        probablyNoise.append(xrNoise)
    
    probablyCNVs = np.concatenate(probablyCNVs)
    probablyNoise = np.concatenate(probablyNoise)
    
    # output cnv calls in bedgraph form
    names = list(depthsByName.keys())
    xrCNVs = tblocks[probablyCNVs][['Left', 'Right', 'Med_Pl']]
        
    with open(cnv_bedgraph, 'w') as OUT:
        for (xrn, blockn), L, R, pl in xrCNVs.itertuples():
            OUT.write('{}\t{}\t{}\t{}\n'.format(names[xrn-1], L, R, pl))
    
    # output filter calls in bed form
    filterInfo = tblocks[['Left', 'Right', 'Quality']]
        
    with open(filter_bed, 'w') as OUT:
        for i, ((xrn, blockn), L, R, ql) in enumerate(filterInfo.itertuples()):
            if ql is 'Non-unique':
                OUT.write('{}\t{}\t{}\t{}\n'.format(names[xrn-1], L, R, ql))
            elif probablyNoise[i]:
                OUT.write('{}\t{}\t{}\t{}\n'.format(names[xrn-1], L, R, 'Noise'))
                