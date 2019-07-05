#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Module to do simple Hubble Diagram fit on Type Ia Supernovae redshift-luminosity relation """

import warnings
import numpy             as np

# - Astropy
import astropy
from astropy import constants
from astropy.cosmology import Planck15 as DEFAULT_COSMO

# - modefit 
from modefit.baseobjects import BaseModel, BaseFitter
    

__all__ = ["get_hubblefit"]

    
CLIGHT_km_s = constants.c.to("km/s").value
PECULIAR_VELOCITY = 300 # in km/s

###############################
#                             #
#   Main Tools                #
#                             #
###############################
def get_hubblefit(data, corr=["x1","c"], **kwargs):
    """ get an object that allows you to do Hubble fit (hubblizer)

    Parameters
    ----------
    data: [dict]
       This dictionary should have this format:
       {name: {k: value1,  k.err: err1,
               k2: value2, k2.err: err2,
               k3: value3,
               cov_kk2: cov_between_k_and_k2,
               cov_k2k: cov_between_k_and_k2,
               etc.
               }}
        empty entries will be assumed to be 0

        Requested parameters:
         - mag
         - zcmb

        Any list of k, k1,k2 could be used for standardization except
        the requested parameters (mag, zcmb)

    corr: [list of string] -optional-
        List and k-values (see data) that you want to use for standardization.

    Return
    ------
    HubbleFit object
    """
    return HubbleFit(data, corr=corr, **kwargs)

def standardization_model(corr):
    """ 
    This function builds and returns the Model used to fit the
    Hubble Data with the defined standardization parameter.
    
    Returns
    -------
    Child of ModelStandardization (with set  STANDARDIZATION)
    """
    class ModelStandardization_mystand( ModelStandardization ):
        STANDARDIZATION = corr

    return ModelStandardization_mystand()

###############################
#                             #
#   Main CLASSES              #
#                             #
###############################
class HubbleFit( BaseFitter ):
    """ """
    PROPERTIES         = ["model", "data", "hdkeys"]
    SIDE_PROPERTIES    = ["pec_velocity"]
    DERIVED_PROPERTIES = ["datafitted","covmatrix","loopcount"]

    # =============== #
    #  Main Methods   #
    # =============== #
    def __init__(self, data, corr, empty=False,
                 add_zerror=True, add_lenserr=True, build=False):
        """  low-level class to enable to fit a bimodal model on data
        given a probability of each point to belong to a group or the other.

        Parameters
        ----------
        data: [array]
            This dictionary should have this format:
            {name: {k: value1, k.err: err1,
                    k2: value2, k2.err, err2,
                    k3: value3,
                    cov_kk2: cov_between_k_and_k2,
                    cov_k2k: cov_between_k_and_k2,
                    etc.
                   }}
            empty entries will be assumed to be 0

            Requested parameters:
            - mag
            - zcmb

            Any list of k, k1,k2 could be used for standardization except
            the requested parameters (mag, zcmb)

        corr: [list of string] -optional-
            List and k-values (see data) that you want to use for standardization.
            
        add_lenserr,add_zerror: [bool] -optional-
            Include the peciluar dispersion/redshift error (add_zerror)
            and lensing error (add_zerror) on the covariance matrix.
            NB: lensing error is set to 0.055*z (Conley et al. 2011, Betoule et al. 2014)
            
        Return
        -------
        Defines the object
        """
        self.set_data(data)
        # -- for the fit
        # use_minuit has a setter
        self.set_model(standardization_model(corr))
        if build:
            self.build_sndata(add_zerror=add_zerror, add_lenserr=add_lenserr)

    # ---------- #
    #  SETTER    #
    # ---------- #
    def set_data(self, data):
        """ """
        self._properties["data"] = data

    def set_zcmb_keys(self, key, keyerr):
        """ """
        self._hdkeys["zcmb"] = key
        self._hdkeys["zcmb.err"] = keyerr

    def set_mb_keys(self, key, keyerr):
        """ """
        self._hdkeys["mb"] = key
        self._hdkeys["mb.err"] = keyerr
        
        
    def build_sndata(self, add_zerror=True, add_lenserr=True, maskout=None):
        """ """
        sndata = self.data[[self._hdkeys["zcmb"],self._hdkeys["zcmb.err"]] +
                           [self._hdkeys["mb"], self._hdkeys["mb.err"]]    +
                           [k for k in self.standardized_by] + [ k+".err" for k in self.standardized_by if k+".err" in self.data]
                          ]
        npoints = len(sndata)

        # Build covariance matrix
        covmat_init = np.zeros((npoints, self.model.nstandardization_coef, self.model.nstandardization_coef))
        
        # Diagonal terms (errors => Variance)
        for i,name in enumerate([self._hdkeys["mb"]]+self.standardized_by):
            covmat_init[:npoints,i,i] = np.asarray(self.get("%s.err"%name, default=0)) **2
            
        # Off diagonal terms (covariance as cov_p1_p2)
        for i, name1 in enumerate([self._hdkeys["mb"]]+self.standardized_by):
            for j,name2 in enumerate([self._hdkeys["mb"]]+self.standardized_by):
                if j>=i:
                    continue
                # cov_ab is the same as cov_ba. Which one was set?
                cov_a_b = self.get("cov_%s_%s"%(name1,name2), default=0)
                if np.all(cov_a_b==0): 
                    cov_a_b = self.get("cov_%s_%s"%(name2,name1), default=0)

                covmat_init[:npoints,i,j] = covmat_init[:npoints,j,i] = cov_a_b

        # Update the covariance matrix with various diagonal terms
        if add_zerror:
            self._has_zerror = True
            covmat_init[:,0,0] += self.get_systerror_redshift_doppler()**2
        else:
            self._has_zerror = False
            
        if add_lenserr:
            self._has_lesserr = True
            covmat_init[:,0,0] += (0.055*self.get(self._hdkeys["zcmb"]))**2
        else:
            self._has_lesserr = False

        # Record it
        self._derived_properties["datafitted"] = sndata
        self._derived_properties["covmatrix"]  = covmat_init
        
    # ---------- #
    #  GETTER    #
    # ---------- #
    def get(self, key, names=None, default=None):
        """ """
        if key in self.data.columns:
            return (self.data.get(key, default=default) if names is None else self.data.loc[names].get(key, default=default)).values
        if names is None:
            names = self.names
        return [default for i in range(len(np.atleast_1d(names)))]
        
    def get_distmod(self, which="corr"):
        """ This distance modulus are as if SN_M0 = 0"""
        if which in ["cosmo","model"]:
            return self.model.cosmo.distmod( self.get(self._hdkeys["zcmb"]) ).value + self.fitvalues["M0"], 0
        
        if which in ["obs","observed", "uncorr","uncorrected"]:
            return self.get(self._hdkeys["mb"]), self.get(self._hdkeys["mb.err"])
        
        if which in ["corr","corrected", "standardized"]:
            return self.get_distmod("obs")[0] - self.model.get_mcorr(self._standard_corrections), np.sqrt(self.model.get_variance(self.covmatrix))

    def get_hubbleres(self, corrected=True):
        """ """
        mag, dmag = self.get_distmod("corr" if corrected else "obs")
        return mag - self.get_distmod("model")[0], dmag

    def get_systerror_redshift_doppler(self, **kwargs):
        """ systematic magnitude error caused by errors on redshift and galaxy peculiar motion """
        dmp = self.get(self._hdkeys["zcmb.err"], **kwargs)**2 + (self.peculiar_velocity/CLIGHT_km_s)**2
        return  5/np.log(10) * np.sqrt(dmp)/self.get(self._hdkeys["zcmb"],**kwargs)

    # -------- #
    #  FITTER  #
    # -------- #
    # This is only there for the intrinsic stuff
    def fit(self, verbose=True, intrinsic=0.0, names=None,
            seek_chi2dof_1=True, chi2dof_margin=0.01,
            use_minuit=None, kfold=None, nsamples=1000, **kwargs):
        """ fit the data on the model

        *Important* minuit fit requires a chi2/dof=1 for accurate error estimate.
        
        Parameters:
        -----------
        intrinsic: [float/array] -optional-
            Intrinsic dispersion added in quadrature to the variances of the
            datapoint (estimated through the covariance matrix).

        verbose: [bool] -optional-
            Have printed information about the fit (chi2 and intrinsic dispersion )
            
        // Fitter to use
        
        use_minuit: [bool/None] -optional-
            If None, this will use the object's current *use_minuit* value.
            If bool, this will set the technique used to fit the *model*
            to the *data* and will thus overwrite the existing
            *self.use_minuit* value.
            Minuit is the iminuit library.
            Scipy.minimize is the alterntive technique.
            The internal functions are made such that none of the other function
            the user have access to depend on the choice of fitter technique.
            For instance the fixing value concept (see set_guesses) remains with
            scipy.

        // K Folding

        kfold: [int, None] -optional-
        
        nsamples: [int]

        // Kwargs
        
        **kwargs parameter associated values with the shape:
            'parametername'_guess, 'parametername'_fixed, 'parametername'_boundaries 

        Returns:
        --------
        Void, create output model values.
        """
        from modefit.utils import kwargs_update
        
        # count the lopping to avoid infinit loops
        self._increment_loop_()
        # set the intrinsic dispersion.
        
        self.model.set_intrinsic(intrinsic)
        # - Default fit values. from numpy.polyfit. You can overright it using kwargs
        proptofit = kwargs_update(self.default_guesses, **kwargs)

        # = The Fit = #
        output = super(HubbleFit,self).fit(**proptofit)

        
        #  Check intrinsic
        # ----------------
        if np.abs(self.fitvalues["chi2"]/self.dof - 1)>chi2dof_margin and seek_chi2dof_1 and self._loopcount<20:
            # => Intrinsic to be added
            if verbose:
                print(" Look for intrinsic dispersion: current chi2 %.2f for %d dof"%(self.fitvalues["chi2"],self.dof))
            return self.fit(intrinsic=self.fit_intrinsic(),verbose=verbose,
                            seek_chi2dof_1=True, chi2dof_margin=chi2dof_margin,
                            use_minuit=use_minuit, kfold=kfold, nsamples=nsamples,**kwargs)
        else:
            # => No intrinsic to be added
            if verbose:
                print(" Used intrinsic dispersion %.3f: chi2 %.2f for %d dof"%(self.model.intrinsic_dispersion, self.fitvalues["chi2"], self.dof))
            return output
        
    def fit_intrinsic(self, intrinsic_guess=0.1):
        """ Get the most optimal intrinsic dispersion given the current fitted standardization parameters. 
        
        The optimal intrinsic magnitude dispersion is the value that has to be added in quadrature to 
        the magnitude errors such that the chi2/dof is 1.

        Returns
        -------
        float (intrinsic magnitude dispersion)
        """
        from scipy import optimize
        def get_intrinsic_chi2dof(intrinsic):
            self.model.set_intrinsic(intrinsic)
            return np.abs(self.get_modelchi2(self._fitparams) / self.dof -1)
        
        return optimize.fmin(get_intrinsic_chi2dof,
                             intrinsic_guess, disp=0)[0]

    # -- Default guesses
    @property
    def default_guesses(self):
        """ On the flight simple linear fit between correction parameter and uncorrected magnitude residual. """
        return {"%s_guess"%param: np.polyfit(self.get(corr),
                                            self.get(self._hdkeys["mb"]) - self.model.get_model(self.get(self._hdkeys["zcmb"]), None), 1)[0]
                    for param,corr in zip(self.model.FREEPARAMETERS_STD, self.model.STANDARDIZATION) }
        
    # ---------- #
    # PLOTTER    #
    # ---------- #
    def show(self, figsize=[6,5],show_res=True, color="C0",
                 ax=None,
                 show_uncorr=False, show_legend = True, show_label = True,
                 propmodel={}, propuncorr={}, **kwargs):
        """ """
        import matplotlib.pyplot as mpl
        if ax is None:
            fig = mpl.figure(figsize=figsize)
        if not show_res:
            if ax is None:
                ax    = fig.add_axes(111)
            else:
                fig = ax.figure
        else:
            if ax is None:
                ax    = fig.add_axes([0.12,0.4,0.78,0.55])
                axres = fig.add_axes([0.12,0.1,0.78,0.27])
            else:
                ax, axres = ax
                fig = ax.figure

        prop = {**dict(ls="None", marker="o", ecolor="0.7", lw=1, alpha=1), **kwargs}
        propmodel = {**dict( ls="-", color="k", zorder=1), **propmodel}
        
        zcmb    = self.get(self._hdkeys["zcmb"])
        zcmberr = self.get(self._hdkeys["zcmb.err"])
        flag_used = np.in1d(self.names, self.names_fitted)
        #
        # Hubble Diagram
        #
        
        ## Uncorrected Hubble
        if show_uncorr:
            mag, magerr    = self.get_distmod("obs")
            ax.errorbar(zcmb, mag, xerr=zcmberr, yerr=magerr,
                    mfc="None", mec="C1", mew=1., **{**prop,**propuncorr})

        ## Corrected
        magc,magcerr  = self.get_distmod("corr")
        #### not used:
        if np.any(~flag_used):
            ax.errorbar(zcmb[~flag_used], magc[~flag_used], xerr=zcmberr[~flag_used], yerr=magcerr[~flag_used],
                            mfc="None", mec=color, mew=1.,
                            label=" [no used (%d)]"%len(flag_used[~flag_used]),
                            **prop)
        #### used:
        if np.any(flag_used):
            ax.errorbar(zcmb[flag_used], magc[flag_used], xerr=zcmberr[flag_used], yerr=magcerr[flag_used],
                            label="corrected [used (%d)]"%len(flag_used[flag_used]),
                            mfc=color, mec="None", **prop)

        # Model
        zz = np.linspace(0.001,0.12,200)
        ax.plot(zz, self.model.cosmo.distmod(zz).value + self.fitvalues["M0"],
                    label=self.model.cosmo.name, scalex=False, scaley=False,
                    **propmodel)
        #
        # Hubble Residual
        #
        if show_res:
            hr, dhr = self.get_hubbleres(corrected=True)
            #### not used:
            if np.any(~flag_used):
                axres.errorbar(zcmb[~flag_used], hr[~flag_used], xerr=zcmberr[~flag_used], yerr=dhr[~flag_used],
                                mfc="None", mec=color,mew=1.,
                                   **prop)
            #### used:
            if np.any(flag_used):
                axres.errorbar(zcmb[flag_used], hr[flag_used], xerr=zcmberr[flag_used], yerr=dhr[flag_used],
                            mfc=color, mec="None", **prop)
            
            axres.axhline(0, **propmodel)
            ax.set_xticklabels(["" for _ in ax.get_xticklabels()])
            axres.set_xlim(*ax.get_xlim())
        if show_legend:
            ax.legend(loc="best")

        if show_label:
            fig.axes[-1].set_xlabel("redshift", fontsize="medium")
            fig.axes[0].set_ylabel(r"distance modulus + SN$_{M}$", fontsize="medium")


        return fig
    
    # -------- #
    #  GETTER  #
    # -------- #
    def _get_model_args_(self):
        """ see model.get_loglikelihood"""
        # corresponding data entry:
        return self._zcmb, self._mb, self._standard_corrections, self.covmatrix

    @property
    def _zcmb(self):
        """ """
        return self.datafitted[self._hdkeys["zcmb"]]

    @property
    def _mb(self):
        """ """
        return self.datafitted[self._hdkeys["mb"]]

    @property
    def _standard_corrections(self):
        """ """
        return self.datafitted[self.standardized_by].values.T

    @property
    def covmatrix(self):
        """ """
        return self._derived_properties["covmatrix"]

    @property
    def datafitted(self):
        """ """
        if self._derived_properties["datafitted"] is None:
            return self.data
        return self._derived_properties["datafitted"]
    
    # -------- #
    #  SETTER  #
    # -------- #    
    # - Peculiar Velocity
    def set_peculiar_velocity(self, velocity_km_s):
        """ Set the peculiar velocity that should be used
        for the data.
        
        Parameters
        ----------
        velocity_km_s:
            None:  this will use the default PECULIAR_VELOCITY
            float: this will be the same for every objects (could be 0)
            array: this array must have the size of the data. Each point
                   could then have its own peculiar velocity
                   
        Returns
        -------
        Void
        """
        if velocity_km_s is None:
            velocity_km_s = PECULIAR_VELOCITY
            
        # - Set it: Float or numpy array
        self._side_properties["pec_velocity"] = velocity_km_s

    # --------- #
    #  PLOT     #
    # --------- #

    # =============== #
    #  Properties     #
    # =============== #
    @property
    def data(self):
        """ """
        return self._properties["data"]
    
    @property
    def names(self):
        """ """
        return self.data.index.values

    @property
    def names_fitted(self):
        """ """
        return self.datafitted.index.values

    
    @property
    def npoints(self):
        """ """
        return len(self.names)

    @property
    def standardized_by(self):
        """ """
        return self.model.STANDARDIZATION

    @property
    def _hdkeys(self):
        """ """
        if self._properties["hdkeys"] is None:
            self._properties["hdkeys"] = {}
            self.set_zcmb_keys("zcmb","zcmb.err")
            self.set_mb_keys("mb","mb.err")
            
        return self._properties["hdkeys"]
    # ======== #
    @property
    def peculiar_velocity(self):
        """ Peculiar velocity of galaxy to be added to the magnitude errors """
        if self._side_properties["pec_velocity"] is None:
            self.set_peculiar_velocity(None)
            
        return self._side_properties["pec_velocity"]

    # - Derived Properties
    @property
    def _loopcount(self):
        """ Number of time the looping is called"""
        if self._derived_properties["loopcount"] is None:
            self._derived_properties["loopcount"] = 0
        return self._derived_properties["loopcount"]
    
    def _increment_loop_(self):
        self._derived_properties["loopcount"] = self._loopcount + 1

# ========================= #
#                           #
#     Hubblizer Model       #
#                           #
# ========================= #

class ModelStandardization( BaseModel ):
    """ Virtual Class able to handle any standardization
    See the standardization_model() function that returns
    the model actually used (defining STANDARDIZATION)
    """
    STANDARDIZATION = []
    # FREEPARAMETERS defined on the flight though __new__
    
    PROPERTIES         = ["cosmo","standard_coef"]
    SIDE_PROPERTIES    = ["sigma_int"]
    DERIVED_PROPERTIES = []

    M0_guess=-19.
    # ================ #
    #  Main Method     #
    # ================ #
    def __new__(cls,*arg,**kwarg):
        """ Black Magic allowing generalization of standardization models """
        
        cls.FREEPARAMETERS_STD = ["a%d"%(i+1) for i in range(len(cls.STANDARDIZATION))]
        cls.FREEPARAMETERS     = ["M0"]+cls.FREEPARAMETERS_STD
        return super(ModelStandardization,cls).__new__(cls)
        
    # -------------------- #
    #  Modefit Generic     #
    # -------------------- #        
    def setup(self, parameters):
        """ fill the standardization_coef property that will be used for the standardization """
        for name,v in zip(self.FREEPARAMETERS, parameters):
            self.standardization_coef[name] = v

    def get_model(self, z, corrections):
        """ the magnitude that should be compared to the observed one:
        m_model = cosmology's distance-modulus + M_0 + standardization
        
        Returns
        -------
        Array (m_model)
        """
        # - model
        return self.cosmo.distmod(z).value + self.standardization_coef.get("M0",-19) + self.get_mcorr(corrections)

    def get_mcorr(self, corrections):
        """ magnitude correction caused by the standardization 
        e.g alpha*stretch + beta*color # beta will be negative
        """
        return np.sum([ self.standardization_coef[alpha]*coef
                    for alpha, coef in zip(self.FREEPARAMETERS_STD, corrections)],
                axis=0) if corrections is not None else 0
    
    def get_loglikelihood(self, z, mag, corrections, covmatrix):
        """ The loglikelihood (-0.5*chi2) of the data given the model

        for N data with M correction (i.e. if x_1 and c standardization, M=2 )
        
        Parameters
        ----------
        z, mag: [N-array,N-array]
            redshift and observed magnitude of the supernovae

        correction: [NxM array]
            correction parameter for each SNe

        covmatrix: [NxM+1xM+1 matrix]
            +1 because of M0
            The full covariance matrix between the standardization parameters and M0

        // Change the model

        parameters: [array] -optional-
            Change the current model with this parameter setup
        
        Returns
        -------
        float (-0.5*chi2)
        """
        res = mag - self.get_model(z, corrections)
        var = self.get_variance(covmatrix)
        if np.any(var<0):
            return -np.inf
        return -0.5 * np.sum(res**2 / var )

    def lnprior(self,parameter):
        """ flat prior """
        for name_param,p in zip(self.FREEPARAMETERS, parameter):
            if "sigma" in name_param and p<0:
                return -np.inf
        return 0
    
    # -------------------- #
    #  Model Special       #
    # -------------------- #
    def set_intrinsic(self, intrinsic_disp):
        """ defines the intrinsic dispersion of the model.
        The intrinsic dispersion is added in quadrature to the variance,
        which is estimated from the covariance matrix.
        
        Returns
        -------
        Void
        """
        if intrinsic_disp<0:
            raise ValueError("intrinsic_disp have to be positive or null")
        
        self._side_properties["sigma_int"] = intrinsic_disp

    def get_variance(self, covmatrix):
        """ returns the variance estimated from the covariance matrix.
        
        It takes into account the current standardization coef. (alpha, beta etc).
        It also includes the model intrinsic dispersion added in quadrature.
        
        Returns
        -------
        array (variances)
        """
        a_ = np.matrix(np.concatenate([[1.0],[self.standardization_coef[k]
                                      for k in self.FREEPARAMETERS_STD]]))
        
        return np.array([np.dot(a_, np.dot(c, a_.T)).sum() for c in covmatrix]) + self.intrinsic_dispersion**2 

    def set_cosmo(self, cosmo):
        """ """
        if astropy.cosmology.core.Cosmology not in cosmo.__class__.__mro__:
            raise TypeError("Only Astropy Cosmology object supported")
        
        self._properties["cosmo"] = cosmo
        
    # =============== #
    #  Properties     #
    # =============== #
    @property
    def intrinsic_dispersion(self):
        """ Intrinsic dispersion added in quadrature to the variance. """
        if self._side_properties["sigma_int"] is None:
            self.set_intrinsic(0)
        return self._side_properties["sigma_int"]
    
    @property
    def standardization_coef(self):
        """ dictionary containing the names and value of the standardization parameters """
        if self._properties["standard_coef"] is None:
            self._properties["standard_coef"] = {}
        return self._properties["standard_coef"]

    @property
    def nstandardization_coef(self):
        """ Number of standardization parameter +1 (the magnitude of the SN)"""
        return len(self.STANDARDIZATION) + 1
        
    @property
    def cosmo(self):
        """ """
        if self._properties["cosmo"] is None:
            warnings.warn("Using Default Cosmology")
            self.set_cosmo(DEFAULT_COSMO)
            
        return self._properties["cosmo"]
        
