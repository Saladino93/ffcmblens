from pixell import enmap, enplot
import pixell

import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rc('lines', linewidth = '4')

matplotlib.rcParams['savefig.dpi'] = 300

matplotlib.rcParams["figure.dpi"] = 100

import seaborn as sns


from orphics import io,catalogs as cats,stats

import astropy.io.fits as fits

import scipy
from scipy import optimize
import scipy.stats


#Some inspired from / Many use orphics


def _save(directory, name, array):
    enmap.write_map(directory+name+'.fits', array)
    
def save(directory, name, i, array):
    _save(directory, name+'_'+str(i), array)


def _read(directory, name):
    return enmap.read_map(directory+name+'.fits')
def read(directory, name, i):
    return _read(directory, name+f'_{i}')


class write_read():
    def __init__(self, directory):
        self.directory = directory
    def set_directory(self, new_directory):
        self.directory = new_directory
    def get_directory(self):
        return self.directory

    def _save(self, directory, name, array):
        enmap.write_map(directory+name+'.fits', array)
    def save(self, name, i, array):
        directory = self.get_directory()
        _save(directory, name+'_'+str(i), array)

    def _read(self, directory, name):
        return enmap.read_map(directory+name+'.fits')
    def read(self, name, i):
        directory = self.get_directory()
        return _read(directory, name+f'_{i}')


def show(mappa):
    enplot.pshow(mappa)


def plot2dbinned(p2d, lmin = 100, lmax = 4000, deltal = 100, label = None, marker = None, color = None):
    
    el, cl = getspec2dbinned(p2d, lmin = lmin, lmax = lmax, deltal = deltal)
    plt.plot(el, cl, label = label, marker = marker, color = color)

    if label is not None:
        plt.legend()

def plot(el, cl, label = None, marker = None, color = None):

    plt.plot(el, cl, label = label, marker = marker, color = color)

    if label is not None:
        plt.legend()

def set_labels(ylabel = '$C_L$', xlabel = '$L$'):
    
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)


#NOTE to be DEPRECATED
import symlens as s
def interpolate(l, cl, modlmap):
        return  s.interp(l, cl)(modlmap)


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

#this is from orphics
def f2power(kmap1, kmap2, pixel_units = False):
        """Similar to power2d, but assumes both maps are already FFTed """
        shape, wcs = kmap1.shape, kmap1.wcs
        normfact = enmap.area(shape, wcs )/ np.prod(shape[-2:])**2.
        norm = 1. if pixel_units else normfact
        res = np.real(np.conjugate(kmap1)*kmap2)*norm
        return res

def ifft(mappa, wcs):
    return enmap.enmap(pixell.fft.ifft(mappa, axes = [-2,-1], normalize = True), wcs)

def fft(mappa):
    return enmap.samewcs(enmap.fft(mappa, normalize = False), mappa)