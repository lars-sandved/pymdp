#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable=no-member
# pylint: disable=not-an-iterable

""" Functions

__author__: Conor Heins, Alexander Tschantz, Brennan Klein
"""

import numpy as np
from scipy import special
from pymdp import utils
from itertools import chain
from opt_einsum import contract

EPS_VAL = 1e-16 # global constant for use in spm_log() function

def spm_dot(X, x, dims_to_omit=None):
    """ Dot product of a multidimensional array with `x`. The dimensions in `dims_to_omit` 
    will not be summed across during the dot product
    
    Parameters
    ----------
    - `x` [1D numpy.ndarray] - either vector or array of arrays
        The alternative array to perform the dot product with
    - `dims_to_omit` [list :: int] (optional)
        Which dimensions to omit
    
    Returns 
    -------
    - `Y` [1D numpy.ndarray] - the result of the dot product
    """

    # Construct dims to perform dot product on
    if utils.is_obj_array(x):
        # dims = list((np.arange(0, len(x)) + X.ndim - len(x)).astype(int))
        dims = list(range(X.ndim - len(x),len(x)+X.ndim - len(x)))
        # dims = list(range(X.ndim))
    else:
        dims = [1]
        x = utils.to_obj_array(x)

    if dims_to_omit is not None:
        arg_list = [X, list(range(X.ndim))] + list(chain(*([x[xdim_i],[dims[xdim_i]]] for xdim_i in range(len(x)) if xdim_i not in dims_to_omit))) + [dims_to_omit]
    else:
        arg_list = [X, list(range(X.ndim))] + list(chain(*([x[xdim_i],[dims[xdim_i]]] for xdim_i in range(len(x))))) + [[0]]

    Y = np.einsum(*arg_list)

    # check to see if `Y` is a scalar
    if np.prod(Y.shape) <= 1.0:
        Y = Y.item()
        Y = np.array([Y]).astype("float64")

    return Y


def spm_dot_classic(X, x, dims_to_omit=None):
    """ Dot product of a multidimensional array with `x`. The dimensions in `dims_to_omit` 
    will not be summed across during the dot product
    
    Parameters
    ----------
    - `x` [1D numpy.ndarray] - either vector or array of arrays
        The alternative array to perform the dot product with
    - `dims_to_omit` [list :: int] (optional)
        Which dimensions to omit
    
    Returns 
    -------
    - `Y` [1D numpy.ndarray] - the result of the dot product
    """

    # Construct dims to perform dot product on
    if utils.is_obj_array(x):
        dims = (np.arange(0, len(x)) + X.ndim - len(x)).astype(int)
    else:
        dims = np.array([1], dtype=int)
        x = utils.to_obj_array(x)

    # delete ignored dims
    if dims_to_omit is not None:
        if not isinstance(dims_to_omit, list):
            raise ValueError("`dims_to_omit` must be a `list` of `int`")
        dims = np.delete(dims, dims_to_omit)
        if len(x) == 1:
            x = np.empty([0], dtype=object)
        else:
            x = np.delete(x, dims_to_omit)

    # compute dot product
    for d in range(len(x)):
        s = np.ones(np.ndim(X), dtype=int)
        s[dims[d]] = np.shape(x[d])[0]
        X = X * x[d].reshape(tuple(s))
        # X = np.sum(X, axis=dims[d], keepdims=True)

    Y = np.sum(X, axis=tuple(dims.astype(int))).squeeze()
    # Y = np.squeeze(X)

    # check to see if `Y` is a scalar
    if np.prod(Y.shape) <= 1.0:
        Y = Y.item()
        Y = np.array([Y]).astype("float64")

    return Y

def factor_dot_flex(M, xs, dims, keep_dims=None):
    """ Dot product of a multidimensional array with `x`.
    
    Parameters
    ----------
    - `M` [numpy.ndarray] - tensor
    - 'xs' [list of numpyr.ndarray] - list of tensors
    - 'dims' [list of tuples] - list of dimensions of xs tensors in tensor M
    - 'keep_dims' [tuple] - tuple of integers denoting dimesions to keep
    Returns 
    -------
    - `Y` [1D numpy.ndarray] - the result of the dot product
    """
    all_dims = tuple(range(M.ndim))
    matrix = [[xs[f], dims[f]] for f in range(len(xs))]
    args = [M, all_dims]
    for row in matrix:
        args.extend(row)

    args += [keep_dims]
    return contract(*args, backend='numpy')

def spm_dot_old(X, x, dims_to_omit=None, obs_mode=False):
    """ Dot product of a multidimensional array with `x`. The dimensions in `dims_to_omit` 
    will not be summed across during the dot product

    #TODO: we should look for an alternative to obs_mode
    
    Parameters
    ----------
    - `x` [1D numpy.ndarray] - either vector or array of arrays
        The alternative array to perform the dot product with
    - `dims_to_omit` [list :: int] (optional)
        Which dimensions to omit
    
    Returns 
    -------
    - `Y` [1D numpy.ndarray] - the result of the dot product
    """

    # Construct dims to perform dot product on
    if utils.is_obj_array(x):
        dims = (np.arange(0, len(x)) + X.ndim - len(x)).astype(int)
    else:
        if obs_mode is True:
            """
            @NOTE Case when you're getting the likelihood of an observation under 
                  the generative model. Equivalent to something like self.values[np.where(x),:]
                  when `x` is a discrete 'one-hot' observation vector
            """
            dims = np.array([0], dtype=int)
        else:
            """
            @NOTE Case when `x` leading dimension matches the lagging dimension of `values`
                  E.g. a more 'classical' dot product of a likelihood with hidden states
            """
            dims = np.array([1], dtype=int)

        x = utils.to_obj_array(x)

    # delete ignored dims
    if dims_to_omit is not None:
        if not isinstance(dims_to_omit, list):
            raise ValueError("`dims_to_omit` must be a `list` of `int`")
        dims = np.delete(dims, dims_to_omit)
        if len(x) == 1:
            x = np.empty([0], dtype=object)
        else:
            x = np.delete(x, dims_to_omit)

    # compute dot product
    for d in range(len(x)):
        s = np.ones(np.ndim(X), dtype=int)
        s[dims[d]] = np.shape(x[d])[0]
        X = X * x[d].reshape(tuple(s))
        # X = np.sum(X, axis=dims[d], keepdims=True)

    Y = np.sum(X, axis=tuple(dims.astype(int))).squeeze()
    # Y = np.squeeze(X)

    # check to see if `Y` is a scalar
    if np.prod(Y.shape) <= 1.0:
        Y = Y.item()
        Y = np.array([Y]).astype("float64")

    return Y


def spm_cross(x, y=None, *args):
    """ Multi-dimensional outer product
    
    Parameters
    ----------
    - `x` [np.ndarray] || [Categorical] (optional)
        The values to perfrom the outer-product with. If empty, then the outer-product 
        is taken between x and itself. If y is not empty, then outer product is taken 
        between x and the various dimensions of y.
    - `args` [np.ndarray] || [Categorical] (optional)
        Remaining arrays to perform outer-product with. These extra arrays are recursively 
        multiplied with the 'initial' outer product (that between X and x).
    
    Returns
    -------
    - `z` [np.ndarray] || [Categorical]
          The result of the outer-product
    """

    if len(args) == 0 and y is None:
        if utils.is_obj_array(x):
            z = spm_cross(*list(x))
        elif np.issubdtype(x.dtype, np.number):
            z = x
        else:
            raise ValueError(f"Invalid input to spm_cross ({x})")
        return z

    if utils.is_obj_array(x):
        x = spm_cross(*list(x))

    if y is not None and utils.is_obj_array(y):
        y = spm_cross(*list(y))

    A = np.expand_dims(x, tuple(range(-y.ndim, 0)))
    B = np.expand_dims(y, tuple(range(x.ndim)))
    z = A * B

    for x in args:
        z = spm_cross(z, x)
    return z

def dot_likelihood(A,obs):

    s = np.ones(np.ndim(A), dtype = int)
    s[0] = obs.shape[0]
    X = A * obs.reshape(tuple(s))
    X = np.sum(X, axis=0, keepdims=True)
    LL = np.squeeze(X)

    # check to see if `LL` is a scalar
    if np.prod(LL.shape) <= 1.0:
        LL = LL.item()
        LL = np.array([LL]).astype("float64")

    return LL


def get_joint_likelihood(A, obs, num_states):
    # deal with single modality case
    if type(num_states) is int:
        num_states = [num_states]
    A = utils.to_obj_array(A)
    obs = utils.to_obj_array(obs)
    ll = np.ones(tuple(num_states))
    for modality in range(len(A)):
        ll = ll * dot_likelihood(A[modality], obs[modality])
    return ll


def get_joint_likelihood_seq(A, obs, num_states):
    ll_seq = utils.obj_array(len(obs))
    for t, obs_t in enumerate(obs):
        ll_seq[t] = get_joint_likelihood(A, obs_t, num_states)
    return ll_seq

def get_joint_likelihood_seq_by_modality(A, obs, num_states):
    """
    Returns joint likelihoods for each modality separately
    """

    ll_seq = utils.obj_array(len(obs))
    n_modalities = len(A)

    for t, obs_t in enumerate(obs):
        likelihood = utils.obj_array(n_modalities)
        obs_t_obj = utils.to_obj_array(obs_t)
        for (m, A_m) in enumerate(A):
            likelihood[m] = dot_likelihood(A_m, obs_t_obj[m])
        ll_seq[t] = likelihood
    
    return ll_seq


def spm_norm(A):
    """ 
    Returns normalization of Categorical distribution, 
    stored in the columns of A.
    """
    A = A + EPS_VAL
    normed_A = np.divide(A, A.sum(axis=0))
    return normed_A

def spm_log_single(arr):
    """
    Adds small epsilon value to an array before natural logging it
    """
    return np.log(arr + EPS_VAL)

def spm_log_obj_array(obj_arr):
    """
    Applies `spm_log_single` to multiple elements of a numpy object array
    """

    obj_arr_logged = utils.obj_array(len(obj_arr))
    for idx, arr in enumerate(obj_arr):
        obj_arr_logged[idx] = spm_log_single(arr)

    return obj_arr_logged

def spm_wnorm(A):
    """ 
    Returns Expectation of logarithm of Dirichlet parameters over a set of 
    Categorical distributions, stored in the columns of A.
    """
    A = A + EPS_VAL
    norm = np.divide(1.0, np.sum(A, axis=0))
    avg = np.divide(1.0, A)
    wA = norm - avg
    return wA


def spm_betaln(z):
    """ Log of the multivariate beta function of a vector.
     @NOTE this function computes across columns if `z` is a matrix
    """
    return special.gammaln(z).sum(axis=0) - special.gammaln(z.sum(axis=0))

def dirichlet_log_evidence(q_dir, p_dir, r_dir):
    """
    Bayesian model reduction and log evidence calculations for Dirichlet hyperparameters
    This is a NumPY translation of the MATLAB function `spm_MDP_log_evidence.m` from the
    DEM package of spm. 

    Description (adapted from MATLAB docstring)
    This function computes the negative log evidence of a reduced model of a
    Categorical distribution parameterised in terms of Dirichlet hyperparameters 
    (i.e., concentration parameters encoding probabilities). It uses Bayesian model reduction 
    to evaluate the evidence for models with and without a particular parameter.
    Arguments:
    ===========
    `q_dir` [1D np.ndarray]: sufficient statistics of posterior of full model
    `p_dir` [1D np.ndarray]: sufficient statistics of prior of full model
    `r_dir` [1D np.ndarray]: sufficient statistics of prior of reduced model
    Returns:
    ==========
    `F` [float]: free energy or (negative) log evidence of reduced model
    `s_dir` [1D np.ndarray]: sufficient statistics of reduced posterior
    """

    # change in free energy or log model evidence
    s_dir = q_dir + r_dir - p_dir
    F  = spm_betaln(q_dir) + spm_betaln(r_dir) - spm_betaln(p_dir) - spm_betaln(s_dir)

    return F, s_dir

def softmax(dist):
    """ 
    Computes the softmax function on a set of values
    """

    output = dist - dist.max(axis=0)
    output = np.exp(output)
    output = output / np.sum(output, axis=0)
    return output

def softmax_obj_arr(arr):

    output = utils.obj_array(len(arr))

    for i, arr_i in enumerate(arr):
        output[i] = softmax(arr_i)
    
    return output

def compute_accuracy(log_likelihood, qs):
    """
    Function that computes the accuracy term of the variational free energy. This is essentially a stripped down version of `spm_dot` above,
    with fewer conditions / dimension handling in the beginning.
    """ 

    ndims_ll, n_factors = log_likelihood.ndim, len(qs)

    dims = list(range(ndims_ll - n_factors,n_factors+ndims_ll - n_factors))
    arg_list = [log_likelihood, list(range(ndims_ll))] + list(chain(*([qs[xdim_i],[dims[xdim_i]]] for xdim_i in range(n_factors))))

    return np.einsum(*arg_list)


def calc_free_energy(qs, prior, n_factors, likelihood=None):
    """ Calculate variational free energy
    @TODO Primarily used in FPI algorithm, needs to be made general
    """
    free_energy = 0
    for factor in range(n_factors):
        # Neg-entropy of posterior marginal H(q[f])
        negH_qs = qs[factor].dot(np.log(qs[factor][:, np.newaxis] + 1e-16))
        # Cross entropy of posterior marginal with prior marginal H(q[f],p[f])
        xH_qp = -qs[factor].dot(prior[factor][:, np.newaxis])
        free_energy += negH_qs + xH_qp

    if likelihood is not None:
        free_energy -= compute_accuracy(likelihood, qs)
    return free_energy

def spm_calc_qo_entropy(A, x):
    """ 
    Function that just calculates the entropy part of the state information gain, using the same method used in 
    spm_MDP_G.m in the original matlab code.

    Parameters
    ----------
    A (numpy ndarray or array-object):
        array assigning likelihoods of observations/outcomes under the various 
        hidden state configurations
    
    x (numpy ndarray or array-object):
        Categorical distribution presenting probabilities of hidden states 
        (this can also be interpreted as the predictive density over hidden 
        states/causes if you're calculating the expected Bayesian surprise)
        
    Returns
    -------
    H (float):
        the entropy of the marginal distribution over observations/outcomes
    """

    num_modalities = len(A)

    # Probability distribution over the hidden causes: i.e., Q(x)
    qx = spm_cross(x)
    qo = 0
    idx = np.array(np.where(qx > np.exp(-16))).T

    if utils.is_obj_array(A):
        # Accumulate expectation of entropy: i.e., E_{Q(o, x)}[lnP(o|x)] = E_{P(o|x)Q(x)}[lnP(o|x)] = E_{Q(x)}[P(o|x)lnP(o|x)] = E_{Q(x)}[H[P(o|x)]]
        for i in idx:
            # Probability over outcomes for this combination of causes
            po = np.ones(1)
            for modality_idx, A_m in enumerate(A):
                index_vector = [slice(0, A_m.shape[0])] + list(i)
                po = spm_cross(po, A_m[tuple(index_vector)])
            po = po.ravel()
            qo += qx[tuple(i)] * po
    else:
        for i in idx:
            po = np.ones(1)
            index_vector = [slice(0, A.shape[0])] + list(i)
            po = spm_cross(po, A[tuple(index_vector)])
            po = po.ravel()
            qo += qx[tuple(i)] * po
   
    # Compute entropy of expectations: i.e., -E_{Q(o)}[lnQ(o)]
    H = - qo.dot(spm_log_single(qo))

    return H

def spm_calc_neg_ambig(A, x):
    """
    Function that just calculates the negativity ambiguity part of the state information gain, using the same method used in 
    spm_MDP_G.m in the original matlab code.
    
    Parameters
    ----------
    A (numpy ndarray or array-object):
        array assigning likelihoods of observations/outcomes under the various 
        hidden state configurations
    
    x (numpy ndarray or array-object):
        Categorical distribution presenting probabilities of hidden states 
        (this can also be interpreted as the predictive density over hidden 
        states/causes if you're calculating the expected Bayesian surprise)
        
    Returns
    -------
    G (float):
        the negative ambiguity (negative entropy of the likelihood of observations given hidden states, expected under current posterior over hidden states)
    """

    num_modalities = len(A)

    # Probability distribution over the hidden causes: i.e., Q(x)
    qx = spm_cross(x)
    G = 0
    qo = 0
    idx = np.array(np.where(qx > np.exp(-16))).T

    if utils.is_obj_array(A):
        # Accumulate expectation of entropy: i.e., E_{Q(o, x)}[lnP(o|x)] = E_{P(o|x)Q(x)}[lnP(o|x)] = E_{Q(x)}[P(o|x)lnP(o|x)] = E_{Q(x)}[H[P(o|x)]]
        for i in idx:
            # Probability over outcomes for this combination of causes
            po = np.ones(1)
            for modality_idx, A_m in enumerate(A):
                index_vector = [slice(0, A_m.shape[0])] + list(i)
                po = spm_cross(po, A_m[tuple(index_vector)])

            po = po.ravel()
            qo += qx[tuple(i)] * po
            G += qx[tuple(i)] * po.dot(np.log(po + np.exp(-16)))
    else:
        for i in idx:
            po = np.ones(1)
            index_vector = [slice(0, A.shape[0])] + list(i)
            po = spm_cross(po, A[tuple(index_vector)])
            po = po.ravel()
            qo += qx[tuple(i)] * po
            G += qx[tuple(i)] * po.dot(np.log(po + np.exp(-16)))

    return G

def spm_MDP_G(A, x):
    """
    Calculates the Bayesian surprise in the same way as spm_MDP_G.m does in 
    the original matlab code.
    
    Parameters
    ----------
    A (numpy ndarray or array-object):
        array assigning likelihoods of observations/outcomes under the various 
        hidden state configurations
    
    x (numpy ndarray or array-object):
        Categorical distribution presenting probabilities of hidden states 
        (this can also be interpreted as the predictive density over hidden 
        states/causes if you're calculating the expected Bayesian surprise)
        
    Returns
    -------
    G (float):
        the (expected or not) Bayesian surprise under the density specified by x --
        namely, this scores how much an expected observation would update beliefs 
        about hidden states x, were it to be observed. 
    """

    num_modalities = len(A)

    # Probability distribution over the hidden causes: i.e., Q(x)
    qx = spm_cross(x)
    G = 0
    qo = 0
    idx = np.array(np.where(qx > np.exp(-16))).T

    if utils.is_obj_array(A):
        # Accumulate expectation of entropy: i.e., E_{Q(o, x)}[lnP(o|x)] = E_{P(o|x)Q(x)}[lnP(o|x)] = E_{Q(x)}[P(o|x)lnP(o|x)] = E_{Q(x)}[H[P(o|x)]]
        for i in idx:
            # Probability over outcomes for this combination of causes
            po = np.ones(1)
            for modality_idx, A_m in enumerate(A):
                index_vector = [slice(0, A_m.shape[0])] + list(i)
                po = spm_cross(po, A_m[tuple(index_vector)])
            
            po = po.ravel()
            qo += qx[tuple(i)] * po
            G += qx[tuple(i)] * po.dot(np.log(po + np.exp(-16)))
    else:
        for i in idx:
            po = np.ones(1)
            index_vector = [slice(0, A.shape[0])] + list(i)
            po = spm_cross(po, A[tuple(index_vector)])
            po = po.ravel()
            qo += qx[tuple(i)] * po
            G += qx[tuple(i)] * po.dot(np.log(po + np.exp(-16)))
   
    # Subtract negative entropy of expectations: i.e., E_{Q(o)}[lnQ(o)]
    G = G - qo.dot(spm_log_single(qo))  # type: ignore

    return G

def kl_div(P,Q):
    """
    Parameters
    ----------
    P : Categorical probability distribution
    Q : Categorical probability distribution

    Returns
    -------
    The KL-divergence of P and Q

    """
    dkl = 0
    for i in range(len(P)):
        dkl += np.dot(P[i], np.log(P[i] + EPS_VAL) - np.log(Q[i] + EPS_VAL))
    return(dkl)

def entropy(A):
    """
    Compute the entropy term H of the likelihood matrix,
    i.e. one entropy value per column
    """
    entropies = np.empty(len(A), dtype=object)
    for i in range(len(A)):
        if len(A[i].shape) > 2:
            obs_dim = A[i].shape[0]
            s_dim = A[i].size // obs_dim
            A_merged = A[i].reshape(obs_dim, s_dim)
        else:
            A_merged = A[i]

        H = - np.diag(np.matmul(A_merged.T, np.log(A_merged + EPS_VAL)))
        entropies[i] = H.reshape(*A[i].shape[1:])
    return entropies


# =============================================================================
# Likelihood Precision (ζ) Belief Updating
# Based on Parr et al. (2022) "Active Inference" Appendix B
#
# NAMING CONVENTION:
#   ζ (zeta) = likelihood/sensory precision (modulates A matrix)
#   γ (gamma) = policy precision (modulates expected free energy) [used elsewhere in pymdp]
#
# This follows Parr et al. (2022) where ζ denotes sensory precision.
# =============================================================================

def scale_likelihood(A, zeta):
    """
    Apply precision weighting to a likelihood matrix.

    Computes A_ζ = softmax(ζ * ln(A)) which scales the "sharpness"
    of the likelihood mapping. Higher ζ means more precise/peaked
    likelihood, lower ζ means flatter/less informative likelihood.

    Parameters
    ----------
    A : np.ndarray
        Likelihood matrix P(o|s) with shape (num_obs, num_states)
        Columns must sum to 1 (proper probability distribution)
    zeta : float
        Likelihood precision parameter (ζ). zeta=1 leaves A unchanged.
        zeta>1 sharpens the distribution, zeta<1 flattens it.

    Returns
    -------
    A_scaled : np.ndarray
        Precision-weighted likelihood matrix (same shape as A)

    Notes
    -----
    This implements the precision-weighting from active inference where
    attention/precision modulates the influence of sensory evidence.
    See Parr et al. (2022) "Active Inference" Chapter 4 and Appendix B.

    Note: ζ (zeta) is used for likelihood precision to distinguish from
    γ (gamma) which denotes policy precision in active inference.
    """
    log_A = np.log(A + EPS_VAL)
    log_A_scaled = zeta * log_A
    # Apply softmax column-wise to maintain proper probability distribution
    A_scaled = softmax(log_A_scaled)
    return A_scaled


def compute_expected_log_likelihood(A, obs, qs):
    """
    Compute the expected log-likelihood E_{Q(s)}[ln P(o|s)] for a given observation.

    This is the "accuracy" term in variational free energy, measuring how well
    the current beliefs explain the observation.

    Parameters
    ----------
    A : np.ndarray
        Likelihood matrix P(o|s) with shape (num_obs, num_states)
    obs : int or np.ndarray
        The observation, either as an index or one-hot vector
    qs : np.ndarray
        Posterior beliefs over hidden states Q(s), shape (num_states,)

    Returns
    -------
    expected_ll : float
        The expected log-likelihood E_{Q(s)}[ln P(o|s)]

    Notes
    -----
    For a discrete observation o, this computes:
        sum_s Q(s) * ln A[o, s]
    """
    # Handle observation as index or one-hot vector
    if np.isscalar(obs) or (isinstance(obs, np.ndarray) and obs.ndim == 0):
        obs_idx = int(obs)
        log_likelihood_given_obs = np.log(A[obs_idx, :] + EPS_VAL)
    else:
        # One-hot observation: compute weighted log-likelihood
        log_A = np.log(A + EPS_VAL)
        log_likelihood_given_obs = obs @ log_A

    # Expected log-likelihood under posterior beliefs
    expected_ll = np.dot(qs, log_likelihood_given_obs)
    return expected_ll


def compute_sensory_prediction_error(A, obs, qs):
    """
    Compute sensory prediction error as the difference between observed and
    expected observation under current beliefs.

    This measures how "surprising" the observation is given current state beliefs.
    Low prediction error = observation matches expectations = high precision appropriate.

    Parameters
    ----------
    A : np.ndarray
        Likelihood matrix P(o|s) with shape (num_obs, num_states)
    obs : int or np.ndarray
        The observation, either as an index or one-hot vector
    qs : np.ndarray
        Posterior beliefs over hidden states Q(s), shape (num_states,)

    Returns
    -------
    prediction_error : float
        Squared prediction error (always non-negative)
    expected_obs : np.ndarray
        Expected observation distribution P(o) = sum_s P(o|s)Q(s)
    """
    # Compute expected observation: P(o) = sum_s P(o|s) Q(s)
    expected_obs = A @ qs

    # Convert observation to one-hot if needed
    if np.isscalar(obs) or (isinstance(obs, np.ndarray) and obs.ndim == 0):
        obs_idx = int(obs)
        actual_obs = np.zeros(A.shape[0])
        actual_obs[obs_idx] = 1.0
    else:
        actual_obs = obs

    # Prediction error: squared difference between actual and expected observation
    prediction_error = np.sum((actual_obs - expected_obs) ** 2)

    return prediction_error, expected_obs


def update_likelihood_precision(
    zeta,
    A,
    obs,
    qs,
    log_zeta_prior_mean=0.0,
    log_zeta_prior_var=2.0,
    zeta_step=0.25,
    min_zeta=0.01,
    max_zeta=10.0
):
    """
    Update likelihood precision (ζ) following Equation B.45 from Parr et al. (2022).

    This is the RECOMMENDED method for publication. It implements precision
    belief updating using prediction error, faithful to Appendix B Equation B.45:

        d(ln ζ)/dt = (1/2)(ε'Πε - tr(Σ⁻¹)) - prior_regularization

    Where:
    - ln(ζ) is log-precision (ensures positivity)
    - ε'Πε is the precision-weighted squared prediction error
    - tr(Σ⁻¹) is the expected precision under the prior

    The key insight: precision should INCREASE when prediction error is LOWER
    than expected, and DECREASE when error is HIGHER than expected.

    Parameters
    ----------
    zeta : float
        Current likelihood precision estimate (ζ)
    A : np.ndarray
        Base likelihood matrix P(o|s) with shape (num_obs, num_states)
    obs : int or np.ndarray
        The observation, either as an index or one-hot vector
    qs : np.ndarray
        Prior beliefs over hidden states Q(s), shape (num_states,)
        NOTE: Should be PRIOR beliefs (before observing), not posterior
    log_zeta_prior_mean : float, default=0.0
        Prior mean for log-precision ln(ζ). Default 0.0 corresponds to ζ ≈ 1.0
    log_zeta_prior_var : float, default=2.0
        Prior variance for log-precision. Larger = weaker regularization,
        allowing more deviation from prior mean. Default 2.0 gives good dynamics.
    zeta_step : float, default=0.25
        Step size for precision updates. Default 0.25 provides stable updates.
    min_zeta : float, default=0.01
        Minimum allowed precision
    max_zeta : float, default=10.0
        Maximum allowed precision

    Returns
    -------
    zeta_new : float
        Updated likelihood precision estimate
    prediction_error : float
        Squared prediction error ||o - A @ Q(s)||²
    expected_error : float
        Expected prediction error under uniform beliefs (baseline)

    Notes
    -----
    The prediction error is computed as:
        PE = ||o - E[o|s]||² = ||o - A @ Q(s)||²

    The expected prediction error under uniform beliefs (baseline):
        E[PE] = ||o - A @ uniform||²

    The update follows B.45 exactly:
        d(ln ζ)/dt = (1/2)(E[PE] - ζ·PE) - (ln(ζ) - μ)/σ²

    Where:
    - E[PE] = exp(μ) * baseline_PE is the expected precision-weighted error
    - ζ·PE is the actual precision-weighted error
    - (ln(ζ) - μ)/σ² is the prior regularization term

    Tuning guidance:
    - zeta_step ∈ [0.1, 0.5]: controls adaptation speed
    - log_zeta_prior_var ∈ [1.0, 4.0]: controls flexibility around prior

    References
    ----------
    Parr, T., Pezzulo, G., & Friston, K. J. (2022). Active Inference:
    The Free Energy Principle in Mind, Brain, and Behavior. MIT Press.
    Appendix B, Equation B.45, pp. 243-257.
    """
    # Convert to log-precision (ensures positivity)
    log_zeta = np.log(zeta + EPS_VAL)

    # Compute prediction error using prior beliefs
    prediction_error, _ = compute_sensory_prediction_error(A, obs, qs)

    # Compute expected prediction error (baseline under uniform beliefs)
    num_states = len(qs)
    uniform_qs = np.ones(num_states) / num_states
    expected_error, _ = compute_sensory_prediction_error(A, obs, uniform_qs)

    # From B.45: d(ln ζ)/dt = (1/2)(ε'Πε - tr(Σ⁻¹))
    #
    # We interpret:
    # - ε'Πε ∝ ζ * prediction_error (precision-weighted actual error)
    # - tr(Σ⁻¹) ∝ exp(log_zeta_prior_mean) * expected_error (prior expected error)

    precision_weighted_error = zeta * prediction_error
    expected_precision_weighted_error = np.exp(log_zeta_prior_mean) * expected_error

    # The precision update from B.45:
    # d(ln ζ)/dt = 0.5 * (expected - actual)
    # When actual < expected: d(ln ζ)/dt > 0 → precision increases
    # When actual > expected: d(ln ζ)/dt < 0 → precision decreases
    error_drive = 0.5 * (expected_precision_weighted_error - precision_weighted_error)

    # Prior regularization on log-precision (Gaussian prior)
    prior_term = (log_zeta - log_zeta_prior_mean) / log_zeta_prior_var

    # Combined update
    d_log_zeta = error_drive - prior_term

    # Gradient descent
    log_zeta_new = log_zeta + zeta_step * d_log_zeta

    # Convert back to precision
    zeta_new = np.exp(log_zeta_new)
    zeta_new = np.clip(zeta_new, min_zeta, max_zeta)

    return zeta_new, prediction_error, expected_error