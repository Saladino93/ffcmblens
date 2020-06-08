from pixell import enmap, lensing

import numpy as np


#Lots of credits to orphics library by Mat

def set_seed(seed):
    np.random.seed(seed)

def get_total_map(shape, wcs, seed = 1):

    return 0


def get_lensed_cmb(shape, wcs, seed = 1, unlensed_cl = None):
    
    return 0

def get_unlensed_cmb(shape, wcs, seed = 1, cl = None, real_space = False):

    set_seed(seed)

    cl2dsqrt = enmap.spec2flat(shape, wcs, cl, 0.5, mode = "constant", smooth = "auto")
    #cl2d = 
    #cl2dsqrt = np.sqrt(cl2d)

    random_map = enmap.fft(enmap.rand_gauss(shape, wcs))
    data = enmap.map_mul(cl2dsqrt, random_map)
    kmap = enmap.ndmap(data, wcs)

    if real_space:
        return enmap.ifft(kmap).real
    else:
        return  kmap

def get_noise(shape, wcs):
    return 0


def get_foregrounds(shape, wcs, seed = 1, foreground_types: dict = {}):
    '''
    foreground_types dictionary with foreground entries.
    e.g.
        foreground_types['point_sources'] = {'number':, 'amplitude': }
    '''

    return 0

def get_point_sources(shape, wcs, number, amplitude):

    result = enmap.empty(shape, wcs)
    poisson = np.random.poisson(number, size = result.size)
    mappa = np.reshape(poisson, shape)
    
    return 0

