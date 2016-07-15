# -*- coding: utf-8 -*-

from __future__ import division, print_function

import numpy as np

def h_fpp(H):
    return np.exp(-0.398405*H)
def log10_h_fpp(H):
    return -0.398405*H/np.log(10)
def empirical_fourier(phases, probs=None, n=20):
    if probs is not None:
        coeff = np.sum(
            probs*np.exp(-2.j*np.pi*(np.arange(n+1)[:,None])*phases), axis=1)
    else:
        coeff = np.sum(
            np.exp(-2.j*np.pi*(np.arange(n+1)[:,None])*phases), axis=1)
    return coeff
def reconstruct(efc, xs):
    efcp = np.asarray(efc, dtype=np.complex)
    efcp[1:] *= 2
    return np.sum(efc*np.exp(2.j*np.pi*xs[:,None]*np.arange(len(efc))),
                      axis=1).real
def h_raw(efc, probs=None):
    coeff = efc[1:]
    if probs is not None:
        tpsq = np.sum(probs**2)
    else:
        tpsq = 1
    k = 2*np.abs(coeff)**2/tpsq
    hs = np.cumsum(k)-4*np.arange(len(k))
    ix = np.argmax(hs)
    return hs[ix], ix+1
def h(phases, probs=None, n=20):
    return log10_h_fpp(h_raw(empirical_fourier(phases, probs, n),probs)[0])

def weighted_histogram_errors(phases, probs=None, bins=20, middle=0):
    if probs is None:
        probs = np.ones_like(phases)
    phases = (phases-(middle-0.5)) % 1 + (middle-0.5)
    prob, be = np.histogram(phases, bins=bins, range=(middle-0.5,middle+0.5),
                             weights=probs)
    prob_sq, be = np.histogram(phases, bins=bins, range=(middle-0.5,middle+0.5),
                             weights=probs**2)
    prob_uncert = np.sqrt(prob_sq)

    return prob, prob_uncert, be

def background_level(probs):
    """Number of background photons that appear to be foreground

    Given the probability weights specified in probs, there will be some
    background photons contributing to the total probabilty coming from
    the source. The number returned from this function is the total
    probability contributed by background photons. When plotting a pulse
    phase histogram in rates, for example, this level (converted to rate)
    should be subtracted from all bins.
    """
    if probs is not None:
        bglevel = np.sum(probs*(1-probs))
    else:
        bglevel = 0
    return bglevel

def foreground_scale(probs):
    """Scale by which apparent foreground rates should be amplified

    Given the photon probabilities specified in probs, once the background
    is subtracted, the foreground total probability is underestimated; scale
    by this factor to recover true (estimated) photon counts.
    """
    if probs is not None:
        fgscale = np.sum(probs)/np.sum(probs**2)
        # fgscale = (1-background_level(probs)/np.sum(probs))**(-1)
    else:
        fgscale = 1
    return fgscale

def plot_result(phases, probs=None, middle=0,
                nharm='best', bins='best',
                time=1.):
    import matplotlib.pyplot as plt
    efc = empirical_fourier(phases, probs)
    hr, nc = h_raw(efc, probs)
    H = log10_h_fpp(hr)
    if nharm is None or nharm=='best':
        nharm = nc
    if bins is None or bins=='best':
        bins = 4*nharm
    bglevel = background_level(probs)/time
    fgscale = foreground_scale(probs)
    prob, prob_uncert, be = weighted_histogram_errors(phases, probs, bins,
                                                          middle=middle)
    plt.plot(np.repeat(be,2)[1:-1],
                fgscale*(np.repeat(prob,2)*bins/time-bglevel))
    plt.errorbar((be[1:]+be[:-1])/2,
                     fgscale*(prob*bins/time-bglevel),
                     fgscale*(prob_uncert*bins/time),
                     linestyle='none')

    xs = np.linspace(middle-0.5,middle+0.5,16*nharm+1)
    ys = reconstruct(efc[:nharm+1], xs)
    plt.plot(xs,fgscale*(ys/time-bglevel))

    plt.xlim(middle-0.5, middle+0.5)

    return H, bins

