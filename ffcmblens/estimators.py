import itertools

import sympy as sp

import symlens as s

from pixell import enmap

import numpy as np

import matplotlib.pyplot as plt


### UTILITY FUNCTIONS FOR ESTIMATORS ###

def get_predefined_estimators(estimator_name, XY = 'TT', field_names = None):
    '''
    Get predefined filter.
    '''

    #For now just use symlens
    #Reimplement in future
    f, F, Fr = s.get_mc_expressions(estimator = estimator_name, XY = XY, field_names = field_names)
    return f, F, Fr


def multiple_bh(filter_vec, responses_matrix):
    inverse_responses_matrix = responses_matrix.inv()
    result = inverse_responses_matrix*filter_vec
    return result

def bh_for_g(n, filter_vec, responses_matrix):
    '''
    n is the index of the component of interest
    '''
    result = multiple_bh(filter_vec, responses_matrix)
    return result[n]


### OBJECTS ###

class BasicLensingEstimator():
    
    def __init__(self, shape, wcs, key, field_names, unnormfilter, unnormfilter_swapped, modecoupling, feed_dict, lmin, lmax, lxcut, lycut):
        
        self.shape = shape
        self.wcs = wcs
                
        self.feed_dict = feed_dict 
        
        if field_names is None:
            field_names = (None, None)
        self.field_names = field_names
            
        self.key = key
        self.estimator = self._add_estimator(unnormfilter, unnormfilter_swapped, modecoupling, self.feed_dict, lmin, lmax, lxcut, lycut)
                   
        self.factor = self._get_factor(shape, wcs)
        
        modlmap = enmap.modlmap(shape, wcs)
        self.modlmap = modlmap
        
    def _get_factor(self, shape, wcs):
        return enmap.pixsize(shape,wcs)**0.5 / (np.prod(shape[-2:])**0.5)
    
    def _add_estimator(self, unnormfilter, unnormfilter_swapped, modecoupling, feed_dict, lmin, lmax, lxcut, lycut):
        estimator = {'unnorm_filter': unnormfilter,
                     'unnormfilter_swapped': unnormfilter_swapped,
                     'modecoupling': modecoupling,
                                'feed_dict': feed_dict,
                                'lmin': lmin, 
                                'lmax': lmax, 
                                'lxcut': lxcut, 
                                'lycut': lycut}
        return estimator
        
        
    def _interpolate(self, l, cl, modlmap):
        return  s.interp(l, cl)(modlmap)
    
    def _integrate(self, integrand, feed_dict, xmask, ymask, groups = None):
        return s.integrate(self.shape, self.wcs, feed_dict, integrand, xmask = xmask, ymask = ymask, physical_units = False, groups = groups).real * self.factor

    def get_estimator(self):
        
        return self.estimator
    
    def get_field_names(self):
        
        return self.field_names
    
    def get_filter(self, unnorm = True, swapped = False):
        
        unnormfilter = self.estimator['unnorm_filter'] if not swapped else self.estimator['unnormfilter_swapped']
        
        if unnorm:  
            return unnormfilter
        else:
            return unnormfilter*self.AL
        
    def get_modecoupling(self):
        
        return self.estimator['modecoupling']
        
        
    def get_feed_dict(self):
        
        return self.estimator['feed_dict']
    
    def get_ells(self):
        
        lmin, lmax = self.estimator['lmin'], self.estimator['lmax']
        lxcut, lycut =  self.estimator['lxcut'], self.estimator['lycut']
        
        return lmin, lmax, lxcut, lycut
    
    def get_masks(self):
        
        lmin, lmax, lxcut, lycut = self.get_ells()
        
        mask = s.mask_kspace(self.shape, self.wcs,lmin = lmin, lmax = lmax, lxcut = lxcut, lycut = lycut)
        
        #for now just return two equal masks
        
        return mask, mask
        
    def get_norm(self, calculate = False):
        
        if not calculate:
            if not 'AL_'+self.key in self.estimator['feed_dict'].keys(): 
                print('Normalisation still not calculated!')
                calculate = True
            else:
                #exit function
                return self.estimator['feed_dict']['AL_'+self.key] 
            
        print('Getting normalisation of the estimator.')
        integrand = self.get_filter(unnorm = True)*self.get_modecoupling()
        feed_dict = self.get_feed_dict()
        xmask, ymask = self.get_masks()
        integration = self._integrate(integrand, feed_dict, xmask, ymask)
        result = self.modlmap**2./integration
        
        self.estimator['feed_dict']['AL'+self.key] = result
        self.AL = sp.Symbol('AL'+self.key)
            
        return result
    
    
    def response_to(self, external_modecoupling):
        
        integrand = self.get_filter(unnorm = False)*external_modecoupling
        feed_dict = self.get_feed_dict()
        xmask, ymask = self.get_masks()
        integration = self._integrate(integrand, feed_dict, xmask, ymask)
        result = integration
        
        return result
        
        
    
    
    def get_optimalnoise(self, calculate = False):
        
        return self.get_norm(calculate = False)*self.modlmap**2./4.
    
    
    def _get_map(self, F):
        
        xmask, ymask = self.get_masks()
        
        if self.field_names == (None, None):
            xname = 'X'
            yname = 'Y'
        else:
            xname, yname = self.field_names
        
        return s.unnormalized_quadratic_estimator_custom(self.shape, self.wcs, self.feed_dict, F, 
                                                xname = xname, yname = yname, xmask = xmask, ymask = ymask,
                                                groups= None, physical_units = True)
    
    
    def get_map(self, norm = True):
        
        F = self.get_filter(unnorm = not norm)

        self._get_map(F)
        
    
    


class LensingEstimator():
    def __init__(self, shape, wcs):
        
        self.shape = shape
        self.wcs = wcs
                
        self.keys = []
        self.estimators = {}
        
        self.colors = {}
                
        self.factor = self._get_factor(shape, wcs)
        
    def _get_factor(self, shape, wcs):
        return enmap.pixsize(shape,wcs)**0.5 / (np.prod(shape[-2:])**0.5)
    
    def _add_estimator(self, estimatorobject):
        key = estimatorobject.key
        self.estimators[key] = estimatorobject
        self.colors[key] = 'C'+int(len(self.estimators.keys()))
        self.keys = self.estimators.keys()
                                
    def _interpolate(self, l, cl, modlmap):
        return  s.interp(l, cl)(modlmap)
    
    def _integrate(self, integrand, feed_dict, xmask, ymask):
        return s.integrate(self.shape, self.wcs, feed_dict, integrand, xmask = xmask, ymask = ymask, physical_units = False).real * self.factor

    
    def _cross_noise(self, feed_dict, alpha_XY, beta_XY, Falpha, Fbeta, Fbeta_rev,
                      xmask = None, ymask = None,
                      field_names_alpha = None, field_names_beta = None,
                      falpha = None, fbeta = None, Aalpha = None, Abeta = None,
                     groups = None, kmask = None, power_name = "t"):
    
        return s.N_l_cross_custom(self.shape, self.wcs, feed_dict, alpha_XY, beta_XY, Falpha, Fbeta, Fbeta_rev,
                      xmask = xmask, ymask = ymask,
                      field_names_alpha = field_names_alpha, field_names_beta = field_names_beta,
                      falpha = falpha, fbeta = fbeta, Aalpha = Aalpha, Abeta = Abeta,
                      groups = groups, kmask = kmask, power_name = power_name)
    
    
    def get_estimatorobject(self, key):
        
        return self.estimators[key]
    
    def get_filter(self, key, unnorm = True, swapped = False):
        
        return self.get_estimatorobject(key).get_filter(unnorm = unnorm, swapped = swapped)
    
    def get_field_names(self, key):
        
        return self.get_estimatorobject(key).get_field_names()
        
    def get_feed_dict(self, key):
        
        return self.get_estimatorobject(key).get_feed_dict()
    
    def get_modecoupling(self, key):
        return self.get_estimatorobject(key).get_modecoupling()
    
    def get_ells(self, key):
        
        lmin, lmax, lxcut, lycut = self.get_estimatorobject(key).get_ells()
        
        return lmin, lmax, lxcut, lycut
    
    def get_masks(self, key):
        
        mask1, mask2 = self.get_estimatorobject(key).get_masks()
        
        return mask1, mask2
        
    def get_norm(self, key, calculate = False):
        
        result = self.get_estimatorobject(key).get_norm(calculate = calculate)
            
        return result
    
    def get_optimalnoise(self, key, calculate = False):
        
        result = self.get_estimatorobject(key).get_optimalnoise(calculate = calculate)
            
        return result
    
    
    def add_estimator(self, estimatorobject, getnorm = True):
        key = estimatorobject.key
        if key in self.estimators.keys():
            print(f'Estimator {key} already present!')
        else:
            self._add_estimator(estimatorobject)
            if getnorm:
                self.get_norm(key, calculate = True)
                            
    
    def gaussian_cov(self, key1, key2, calculate = False):
        
        if calculate:
            print(f'Getting Gaussian covariance for {key1} and {key2}')
            
            filter1 = self.get_filter(key1, unnorm = True)
            filter2 = self.get_filter(key2, unnorm = True)
            filter2swapped = self.get_filter(key2, unnorm = True, swapped = True)
            
            modecoupling1 = self.get_modecoupling(key1)
            modecoupling2 = self.get_modecoupling(key2)
            
            norm1 = self.get_norm(key1)
            norm2 = self.get_norm(key2)
            
            xmask1, ymask1 = self.get_masks(key1)
            xmask2, ymask2 = self.get_masks(key2)
            
            xmask = xmask1*xmask2
            ymask = ymask1*ymask2
            
            new_feed_dict = {**self.get_feed_dict(key1), **self.get_feed_dict(key2)}
                        
            result = self._cross_noise(new_feed_dict, alpha_XY = 'TT', beta_XY = 'TT',
                             Falpha = filter1, Fbeta = filter2, Fbeta_rev = filter2swapped,
                             xmask = xmask, ymask = ymask, 
                             falpha = modecoupling1, fbeta = modecoupling2,
                             Aalpha = norm1, Abeta = norm2, kmask = None)
            return result
            
            
            

            
    def generate_gaussian_cov(self, key_list = None, verbose = True):
        
        if key_list is None:
            lista = self.keys
        else:
            lista = key_list
            
        if verbose:
            print('Keys are, ', self.keys)
            
        listKeys = list(itertools.combinations_with_replacement(list(lista), 2))
        
        result = {}
            
        for a, b in listKeys:
            print(f'{a}-{b}')
            Cov = self.gaussian_cov(a, b)
            result[a+","+b] = Cov
            
    