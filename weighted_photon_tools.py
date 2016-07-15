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
    phases = warp(phases,middle)
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

class Profile(object):
    def __init__(self, coefficients, uncertainties):
        self.coefficients = np.array(coefficients, dtype=np.complex)
        if isinstance(uncertainties, float) or isinstance(uncertainties, int):
            self.uncertainties = uncertainties*np.ones(len(self.coefficients),dtype=np.float)
        else:
            self.uncertainties = np.array(uncertainties)
        n, = self.coefficients.shape
        if self.uncertainties.shape != (n,):
            raise ValueError("Uncertainties array is not the same shape as coefficients (%s != %s)" 
                             % (self.uncertainties.shape, self.coefficients.shape))
    def scale(self, factor):
        self.coefficients *= factor
        self.uncertainties *= factor
    def add(self, amount):
        self.coefficients[0] += amount

    def H_internal(self):
        coeff = self.coefficients[1:]/(self.uncertainties[1:]*np.sqrt(2))
        total_prob_sq = 1
        k = 2*np.abs(coeff)**2/total_prob_sq
        a = np.cumsum(k)-4*np.arange(len(k))
        h, nbest = np.amax(a), np.argmax(a)+1
        return h, nbest
    def H(self):
        return -0.398405*self.H_internal()[0]

    def grid_values(self, n_coefficients=None, n=None, turns=1):
        if n_coefficients is None:
            n_coefficients = len(self.coefficients)-1
        elif n_coefficients>len(self.coefficients)-1:
            raise ValueError("%d coefficients requested but only %d available" 
                             % (n_coefficients, len(self.coefficients)-1))
        if n is None:
            nn = 2*n_coefficients*16
            n = 1
            while n<nn:
                n *= 2
        c = self.coefficients[:n_coefficients+1].copy()
        c[1::2] *= -1
        return (np.linspace(-0.5,turns-0.5,turns*n+1),
                np.concatenate(
                    [np.fft.irfft(c.conj(),n=n)*n]*turns))

    def shift(self, phase=None):
        if phase is None:
            phase = -np.angle(self.coefficients[1])/(2*np.pi)
        self.coefficients *= np.exp(2.j*np.pi*phase*np.arange(len(self.coefficients)))

    def compute_shift(self, template, n_coefficients=None):
        if n_coefficients is None:
            n_coefficients = min(len(self.coefficients)-1,
                                 len(template.coefficients))
        c1 = np.zeros(n_coefficients+1, dtype=np.complex)
        c1n = min(len(c1),len(self.coefficients))
        c1[:c1n] = self.coefficients[:c1n]
        c2 = np.zeros(len(c1), dtype=np.complex)
        c2n = min(len(c2),len(template.coefficients))
        c2[:c2n] = template.coefficients[:c2n]
        c1[0] = 0
        c = c1*c2.conj()
        n = 1
        while n<16*n_coefficients:
            n *= 2
        r = np.fft.irfft(c, n=n)
        p = np.argmax(r)/n
        def cross(p):
            return -np.sum(
                    c1*c2.conj()*np.exp(2.j*np.pi*p*np.arange(len(c1)))
                    ).real
        #return r, p, [cross(pp) for pp in np.linspace(0,1,len(r), endpoint=False)] 
        #print cross(p-1/n), cross(p), cross(p+1/n)
        pmax = scipy.optimize.brent(cross,
                                    brack=(p-1/n, p, p+1/n))
        return pmax

    def trim(self, n_coefficients):
        self.coefficients = self.coefficients[:n_coefficients+1]
        self.uncertainties = self.uncertainties[:n_coefficients+1]

    def generate_fake_phases(self, n, n_coefficients=None):
        p, v = self.grid_values(n_coefficients=n_coefficients)
        cs = np.cumsum(v)
        r = p[np.searchsorted(cs,np.random.uniform(high=cs[-1], size=n))]
        r += np.random.uniform(high=p[1]-p[0],size=n)
        return r

    def generate_fake_profile(self, n_coefficients=None, uncertainties=None):
        if n_coefficients is None:
            n_coefficients = len(self.coefficients)-1
        if uncertainties is None:
            uncertainties = self.uncertainties
        c = np.zeros(n_coefficients+1, dtype=complex)
        nc = min(n_coefficients+1, len(self.coefficients))
        c[:nc] = self.coefficients[:nc]
        c += uncertainties*(np.random.randn(len(c))
                            +1.j*np.random.randn(len(c)))
        return Profile(c,uncertainties)

    def copy(self):
        return Profile(self.coefficients.copy(),
                       self.uncertainties.copy())

    def rms(self, n_coefficients=None):
        if n_coefficients is None:
            n_coefficients = len(self.coefficients)-1
        elif n_coefficients>len(self.coefficients)-1:
            raise ValueError("%d coefficients requested but only %d available" 
                             % (n_coefficients, len(self.coefficients)-1))
        s = np.sum(np.abs(self.coefficients[1:n_coefficients+1])**2
                   -2*self.uncertainties[1:n_coefficients+1]**2)
        #s = np.sum(np.abs(self.coefficients[1:n_coefficients+1])**2)
        if s<0:
            s=0
        return np.sqrt(2*s), np.sqrt(2*np.mean(self.uncertainties[1:n_coefficients+1]**2))

def circmean(phases):
    return np.angle(np.sum(np.exp(2.j*np.pi*phases)))/(2*np.pi)

def fold_phases(phases, probs=None, n=20):
    c = np.zeros(n+1, dtype=np.complex)
    if probs is None:
        for i in range(n+1):
            c[i] = np.sum(np.exp(2.j*np.pi*i*phases))

        return Profile(c, np.sqrt(len(phases)/2.))
    else:
        for i in range(n+1):
            c[i] = np.sum(probs*np.exp(2.j*np.pi*i*phases))

        return Profile(c, np.sqrt(np.sum(probs**2)/2.))


def wrap(x,middle=0):
    return (x+(0.5-middle)) % 1 - (0.5-middle)

def center_wrap(x, axis=None, weights=None):
    circ_mean = np.angle(np.average(np.exp(2.j*np.pi*x),
                                    weights=weights,
                                    axis=axis))/(2*np.pi)
    x = wrap(x-circ_mean)+circ_mean
    return x

def std_wrap(x):
    return np.std(center_wrap(x))
