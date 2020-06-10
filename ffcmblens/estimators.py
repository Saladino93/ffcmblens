import itertools

import sympy as sp

import symlens as s

from pixell import enmap

import numpy as np

import utils


### UTILITY FUNCTIONS FOR ESTIMATORS ###

def get_predefined_estimators(estimator_name, XY = 'TT', field_names = None):
    '''
    Get predefined filter.
    '''

    #For now just use symlens
    #Reimplement in future
    f, F, Fr = s.get_mc_expressions(estimator = estimator_name, XY = XY, field_names = field_names)

    if estimator_name in ['hu_ok', 'hdv', 'shear']:
        kappa_corr = 2/s.L**2
        f *= kappa_corr
        F *= kappa_corr
        Fr *= kappa_corr
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


#NOTE: to update
def get_bias_hardened_with_filters(filter_vec, filter_vec_r, n):
    
    #generalise to NxN
    RAB, RBA = sp.symbols('RAB RBA')
    responses_matrix = sp.Matrix([[1, RAB], [RBA, 1]])
    
    bhQ_wrtP = bh_for_g(n = n, filter_vec = filter_vec, responses_matrix = responses_matrix)
    bhQ_wrtPr = bh_for_g(n = n, filter_vec = filter_vec_r, responses_matrix = responses_matrix)
    
    det = responses_matrix.det()
    determinant = sp.Symbol('D')
    
    FBHQE, FrBHQE = sp.cancel(bhQ_wrtP*det)*(1), sp.cancel(bhQ_wrtPr*det)*(1)

    return FBHQE, FrBHQE

def get_bias_hardened(estimator1, estimator2):

    f1, f2 = estimator1.get_modecoupling(), estimator2.get_modecoupling()

    F1, F1r = estimator1.get_filter(unnorm = True), estimator1.get_filter(unnorm = True, swapped = True)
    F2, F2r = estimator2.get_filter(unnorm = True), estimator2.get_filter(unnorm = True, swapped = True)

    R1_2 = estimator1.response_to(f2, withoutnorm = False)
    R2_1 = estimator2.response_to(f1, withoutnorm = False)

    filter_vec = sp.Matrix([F1*estimator1.AL, F2*estimator2.AL])

    filter_vec_r = sp.Matrix([F1r*estimator1.AL, F2r*estimator2.AL])

    RAB, RBA = sp.symbols('RAB RBA')

    responses_matrix = sp.Matrix([[1, RAB], [RBA, 1]])

    Fbh1_wrt2 = bh_for_g(n = 0, filter_vec = filter_vec, responses_matrix = responses_matrix)

    Fbh1_wrt2r = bh_for_g(n = 0, filter_vec = filter_vec_r, responses_matrix = responses_matrix)

    det = responses_matrix.det()

    determinant = sp.Symbol('D')

    #I do this det/determinant for some reason that symlens does not allow fast simplification
    fBHQE, FBHQE, FrBHQE = f1, sp.cancel(Fbh1_wrt2*det)*(1/determinant), sp.cancel(Fbh1_wrt2r*det)*(1/determinant)

    new_feed_dict = {**estimator1.feed_dict, **estimator2.feed_dict}

    new_feed_dict['D'] = 1-R1_2*R2_1

    new_feed_dict[str(RAB)] = R1_2
    new_feed_dict[str(RBA)] = R2_1
    
    new_feed_dict[str(estimator1.AL)] = estimator1.get_norm()
    new_feed_dict[str(estimator2.AL)] = estimator2.get_norm()

    return new_feed_dict, fBHQE, FBHQE, FrBHQE 

    

### OBJECTS ###

class BasicLensingEstimator():
    
    def __init__(self, shape, wcs, key, field_names, unnormfilter, unnormfilter_swapped, modecoupling, feed_dict, lmin, lmax, lxcut, lycut):
        
        self.shape = shape
        self.wcs = wcs
                
        self.feed_dict = feed_dict.copy()
        
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
            try:
                return unnormfilter*self.AL
            except:
                print('Maybe you did not precalculate AL')
                self.get_norm(calculate = True)
                return unnormfilter*self.AL
        
    def get_modecoupling(self):
        
        return self.estimator['modecoupling']
        
        
    def get_feed_dict(self):
        
        return self.estimator['feed_dict']
    
    def get_ells(self):
        
        lmin, lmax = self.estimator['lmin'], self.estimator['lmax']
        lxcut, lycut =  self.estimator['lxcut'], self.estimator['lycut']
        
        return lmin, lmax, lxcut, lycut

    def set_ells(self, lmin = None, lmax = None, lxcut = None, lycut = None):
        
        if lmin is not None:
            self.estimator['lmin'] = lmin
        if lmax is not None:
            self.estimator['lmax'] = lmax
        if lxcut is not None:
            self.estimator['lxcut'] = lxcut
        if lycut is not None:
            self.estimator['lycut'] = lycut
        
        
    
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
        result = 1/integration
        
        self.estimator['feed_dict']['AL'+self.key] = result
        self.AL = sp.Symbol('AL'+self.key)
            
        return result
    
    
    def response_to(self, external_modecoupling, withoutnorm = False):

        unnorm = withoutnorm
        
        integrand = self.get_filter(unnorm = unnorm)*external_modecoupling
        feed_dict = self.get_feed_dict()
        xmask, ymask = self.get_masks()
        integration = self._integrate(integrand, feed_dict, xmask, ymask)
        result = integration
        
        return result

    
    def get_fieldnames(self, forxnames = False):

        if self.field_names == (None, None):
            xname = 'X'
            yname = 'Y'
        else:
            xname, yname = self.field_names

        if forxnames:
            xname = xname+'_l1'
            yname = yname+'_l2'

        return [xname, yname]   
        
    
    
    def get_optimalnoise(self, calculate = False):
        
        return self.get_norm(calculate = False)#*self.modlmap**2./4.
    
    
    def _get_map(self, F):
        
        xmask, ymask = self.get_masks()
        
        xname, yname = self.get_fieldnames(forxnames = True)

        
        kappa = s.integrate(self.shape, self.wcs, self.feed_dict,
                          s.e(xname)*s.e(yname)*F, xmask = xmask, ymask = ymask,
                          groups = None, physical_units = True)
        
        return kappa

    def get_map(self, norm = True, calculate = False):

        try:
            if self.lensing_rec is not None:
                return self.lensing_rec
            if calculate:
                raise Exception
        except:
            if not calculate:    
                print('You do not have lensing map.')

        print('Getting lensing')
        
        F = self.get_filter(unnorm = not norm)

        self.lensing_rec = self._get_map(F)
        
    
    


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
        self.colors[key] = 'C'+str(len(self.estimators.keys()))
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
        #4 to undo a 4 in symlens code as I am using different convention
        return 4*s.N_l_cross_custom(self.shape, self.wcs, feed_dict, alpha_XY, beta_XY, Falpha, Fbeta, Fbeta_rev,
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

    def set_ells(self, key, lmin = None, lmax = None, lxcut = None, lycut = None):

        self.get_estimatorobject(key).set_ells(lmin, lmax, lxcut, lycut)
    
    def get_masks(self, key):
        
        mask1, mask2 = self.get_estimatorobject(key).get_masks()
        
        return mask1, mask2
        
    def get_norm(self, key, calculate = False):
        
        result = self.get_estimatorobject(key).get_norm(calculate = calculate)
            
        return result
    
    def get_optimalnoise(self, key, calculate = False):
        
        result = self.get_estimatorobject(key).get_optimalnoise(calculate = calculate)
            
        return result


    def get_map(self, key, norm = True):

        return self.get_estimatorobject(key).get_map(norm)
    
    
    def add_estimator(self, estimatorobject, getnorm = True):
        key = estimatorobject.key
        if key in self.estimators.keys():
            print(f'Estimator {key} already present!')
        else:
            self._add_estimator(estimatorobject)
            if getnorm:
                self.get_norm(key, calculate = True)

    def get_fieldnames(self, key):

        return self.get_estimatorobject(key).get_fieldnames()
                            
    
    def gaussian_cov(self, key1, key2 = None, calculate = True, normalize_filter = True):

        if key2 is None:
            key2 = key1
        
        if calculate:
            print(f'Getting Gaussian covariance for {key1} and {key2}')
            
            filter1 = self.get_filter(key1, unnorm = True)
            filter2 = self.get_filter(key2, unnorm = True)
            filter2swapped = self.get_filter(key2, unnorm = True, swapped = True)
            
            modecoupling1 = self.get_modecoupling(key1)
            modecoupling2 = self.get_modecoupling(key2)
            
            if normalize_filter:
                norm1 = self.get_norm(key1)
                norm2 = self.get_norm(key2)
            else:
                norm1 = 1.
                norm2 = 1.
            
            xmask1, ymask1 = self.get_masks(key1)
            xmask2, ymask2 = self.get_masks(key2)
            
            xmask = xmask1*xmask2
            ymask = ymask1*ymask2
            
            new_feed_dict = {**self.get_feed_dict(key1), **self.get_feed_dict(key2)}

            fieldnames1 = self.get_fieldnames(key1)
            fieldnames2 = self.get_fieldnames(key2)

            result = self._cross_noise(new_feed_dict, alpha_XY = 'TT', beta_XY = 'TT',
                             Falpha = filter1, Fbeta = filter2, Fbeta_rev = filter2swapped,
                             xmask = xmask, ymask = ymask, field_names_alpha = fieldnames1, field_names_beta = fieldnames2,
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

        self.lensing_gaussian_cov_dict = result


    def lensing_gaussian_cov(self, key1, key2):

        if key2 is None:
            key2 = key1

        return self.lensing_gaussian_cov_dict[key1+","+key2]

    
    def trispectrum_from_source(self, key1, key2, S):
        estimator1 = self.get_estimatorobject(key1)
        estimator2 = self.get_estimatorobject(key2)

        filter1 = estimator1.get_filter(unnorm = False)
        filter2 = estimator2.get_filter(unnorm = False)
        
        feed_dict1 = estimator1.feed_dict
        feed_dict2 = estimator2.feed_dict

        xmask1, ymask1 = estimator1.get_masks()
        xmask2, ymask2 = estimator2.get_masks()

        result1 = s.integrate(self.shape, self.wcs, feed_dict1, filter1, xmask = xmask1, ymask = ymask1, physical_units = True).real
        result2 = s.integrate(self.shape, self.wcs, feed_dict2, filter2, xmask = xmask2, ymask = ymask2, physical_units = True).real

        T = result1*result2*S

        return T

    def generate_trispectra_from_source(self, S, key_list = None, verbose = True):
        
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
            T = self.trispectrum_from_source(key1 = a, key2 = b)
            result[a+","+b] = Cov

        self.trispectra_from_source = result
        

    def plot_binned(self, key1, key2 = None, mappa = False, apply_w4 = 1, lmin = 100, lmax = 4000, deltal = 100, cls = True):

        if not mappa:
            power2d = self.lensing_gaussian_cov(key1, key2)
        else:
            power2d = utils.f2power(self.get_map(key1), self.get_map(key2))/apply_w4

        el, cl = utils.getspec2dbinned(power2d, lmin = lmin, lmax = lmax, deltal = deltal)

        if key2 is None:
            stringa = key1
        else:
            stringa = key1+'-'+key2

        utils.plot(el, cl, label = '$'+stringa+'$')
        utils.set_labels()

        if cls:
            return el, cl


        
            
    