from pixell import enmap

import numpy as np
import matplotlib.pyplot as plt

from orphics import io,catalogs as cats,stats

import astropy.io.fits as fits

import scipy
from scipy import optimize
import scipy.stats






def bin_theory(l, lcl, bin_edges):
    sums = scipy.stats.binned_statistic(l, l, statistic = 'sum', bins = bin_edges)
    cl = scipy.stats.binned_statistic(l, lcl, statistic = 'sum', bins = bin_edges)
    cl = cl[0]/sums[0]
    return cl
    
def getspec(map1, map2 = None, lmin = 100, lmax = 4000, deltal = 100):
    
    if map2 is None:
        map2 = map1
    
    shape,wcs = map1.shape, map1.wcs
    from orphics import maps as omaps    
    fc = omaps.FourierCalc(shape,wcs)
    p2d,kmap,_ = fc.power2d(map1, map2)
    bin_edges = np.arange(lmin, lmax, deltal)
    modlmap = enmap.modlmap(shape,wcs)
    binner = stats.bin2D(modlmap,bin_edges)
    cents, p1d = binner.bin(p2d)
    
    return cents, p1d      

def getspec_from_bins(map1, map2 = None, bin_edges = None):
    
    if map2 is None:
        map2 = map1
    
    shape,wcs = map1.shape, map1.wcs
   
    fc = maps.FourierCalc(shape,wcs)
    p2d,kmap,_ = fc.power2d(map1, map2)
    modlmap = enmap.modlmap(shape,wcs)
    binner = stats.bin2D(modlmap, bin_edges)
    cents, p1d = binner.bin(p2d)
    
    return cents, p1d 

def getspec2dbinned(p2d, lmin = 100, lmax = 4000, deltal = 100):

    shape,wcs = p2d.shape, p2d.wcs
    bin_edges = np.arange(lmin, lmax, deltal)
    modlmap = enmap.modlmap(shape, wcs)
    binner = stats.bin2D(modlmap, bin_edges)
    cents, p1d = binner.bin(p2d)

    return cents, p1d

def getspec2d(map1, map2 = None):
    
    if map2 is None:
        map2 = map1
    
    shape,wcs = map1.shape, map1.wcs
    
    fc = maps.FourierCalc(shape,wcs)
    p2d,kmap,_ = fc.power2d(map1, map2)
    
    return p2d, kmap     