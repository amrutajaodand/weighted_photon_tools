# -*- coding: utf-8 -*-

from __future__ import division, print_function

import os
import subprocess
import sys
import shutil
import pickle
import time
from logging import info, debug

from astropy.coordinates import get_icrs_coordinates

import numpy as np

Fermi_MJDREFf = 51910
Fermi_MJDREFi = 7.428703703703703*10**-4
FERMI_MJDREF = Fermi_MJDREFf + Fermi_MJDREFi

def Fermi_MET_to_MJD(x):
    y = x/86400.0 + FERMI_MJDREF
    return y

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
    """Object representing a pulse profile in Fourier space

    This represents a periodic function (with period 1) estimated from
    a finite data set; it therefore records a finite number of
    coefficients with uncertainties on each. These objects allow
    shifting, finding the optimal shift to match another, estimation
    of pulsed significance, estimation of the optimal number of harmonics
    for representation, and estimation of RMS amplitude with uncertainty.
    """
    def __init__(self, coefficients, uncertainties):
        self.coefficients = np.array(coefficients, dtype=np.complex)
        if isinstance(uncertainties, float) or isinstance(uncertainties, int):
            self.uncertainties = uncertainties*np.ones(len(self.coefficients),
                                                           dtype=np.float)
        else:
            self.uncertainties = np.array(uncertainties)
        n, = self.coefficients.shape
        if self.uncertainties.shape != (n,):
            raise ValueError("Uncertainties array is not the same shape "
                                 "as coefficients (%s != %s)"
                             % (self.uncertainties.shape,
                                    self.coefficients.shape))
    def scale(self, factor):
        self.coefficients *= factor
        self.uncertainties *= factor
    def add(self, other):
        if isinstance(other, float):
            self.coefficients[0] += amount
        else:
            raise ValueError("Adding %s to Profile unimplemented" % other)

    def H_internal(self):
        """Compute best number of harmonics and significance

        Returns the optimal number of harmonics for representing this Profile
        and the raw H score (which is proportional to the logarithm of the false
        positive probability). This number of harmonics is roughlt the number of
        statistically significant harmonics, and is often appropriate
        for graphical representation or cross-correlation.
        """
        coeff = self.coefficients[1:]/(self.uncertainties[1:]*np.sqrt(2))
        total_prob_sq = 1
        k = 2*np.abs(coeff)**2/total_prob_sq
        a = np.cumsum(k)-4*np.arange(len(k))
        h, nbest = np.amax(a), np.argmax(a)+1
        return h, nbest
    def H(self):
        """Test for significance of pulsations

        This function applies the H test to the Profile and returns the
        logarithm of the probability that pure noise would have given rise
        to a Profile this strongly modulated.
        """
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
        # FIXME: explain direction
        if phase is None:
            phase = -np.angle(self.coefficients[1])/(2*np.pi)
        self.coefficients *= np.exp(2.j*np.pi*phase
                                        *np.arange(len(self.coefficients)))

    def compute_shift(self, template, n_coefficients=None):
        # FIXME: explain direction
        # FIXME: use uncertainties sensibly
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
        pmax = scipy.optimize.brent(cross,
                                    brack=(p-1/n, p, p+1/n))
        return pmax

    def trim(self, n_coefficients):
        """Discard excess coefficients

        Typically the higher Fourier coefficients will be more seriously
        contaminated by noise. This function discards them. Use H_internal
        to obtain a recommendation for how many are worth keeping.
        """
        self.coefficients = self.coefficients[:n_coefficients+1]
        self.uncertainties = self.uncertainties[:n_coefficients+1]

    def generate_fake_phases(self, n, n_coefficients=None):
        """Generate a set of phases drawn from this distribution"""
        p, v = self.grid_values(n_coefficients=n_coefficients)
        cs = np.cumsum(v)
        r = p[np.searchsorted(cs,np.random.uniform(high=cs[-1], size=n))]
        r += np.random.uniform(high=p[1]-p[0],size=n)
        return r

    def generate_fake_profile(self, n_coefficients=None, uncertainties=None):
        """Generate a profile similar to this by applying noise

        The amount of noise is set by the uncertainties in the Fourier
        coefficients of this Profile. This may be useful for testing the
        correctness of statistical tests.
        """
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
        """Return an estimate of the RMS amplitude of this profile"""
        if n_coefficients is None:
            n_coefficients = len(self.coefficients)-1
        elif n_coefficients>len(self.coefficients)-1:
            raise ValueError("%d coefficients requested but only %d available"
                             % (n_coefficients, len(self.coefficients)-1))
        s = np.sum(np.abs(self.coefficients[1:n_coefficients+1])**2
                   -2*self.uncertainties[1:n_coefficients+1]**2)
        if s<0:
            s=0
        return (np.sqrt(2*s),
                np.sqrt(2*np.mean(self.uncertainties[1:n_coefficients+1]**2)))

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



def ensure_list(l):
    if isinstance(l,basestring) or isinstance(l,int):
        l = [l]
    return l

def touch(f):
    if not os.path.exists(f):
        raise ValueError("File %s does not exist yet" % f)
    subprocess.check_call(["touch",f])

def need_rerun(inputs, outputs):
    """Examine inputs and outputs and return whether a command should be rerun.

    If one of the outputs does not exist, or if the modification date of the
    newest input is newer than the oldest output, return True; else False. The
    idea is to allow make-like behaviour.
    """
    inputs = ensure_list(inputs)
    outputs = ensure_list(outputs)

    if len(outputs)==0:
        raise ValueError("No outputs specified")

    io = inputs
    inputs = []
    for i in io:
        if i.startswith("@"):
            for l in open(i[1:]).readlines():
                inputs.append(l.strip())
        else:
            inputs.append(i)

    oldest_out = np.inf
    oldest_out_name = None

    for o in outputs:
        if not os.path.exists(o):
            info("Output %s missing" % o)
            return True
        ot = os.path.getmtime(o)
        if ot<oldest_out:
            oldest_out = ot
            oldest_out_name = o

    for i in inputs:
        if os.path.getmtime(i) > oldest_out:
            info("Input %s newer than %s" % (i,oldest_out_name))
            debug("%s > %s" %
                      (time.ctime(os.path.getmtime(i)),
                        time.ctime(os.path.getmtime(oldest_out_name))))
            return True

    return False

class Command(object):
    """Run a command from the set of Fermi tools.

    Commands are automatically rerun only if necessary. Upon construction of
    the object, the input and output file arguments are listed; for keyword
    arguments, the name is given, while for positional arguments, the position.

    On calling this object, positional arguments appear in positions, and
    keyword arguments are appended in the form key=value. Two special keyword
    arguments are recognized:

    rerun determines whether to force a rerun of the command.
    True means always rerun, False means never, and None (the default)
    means rerun if necessary.

    call_id is a string describing this particular call. If provided,
    standard out and standard error are saved to files and can be displayed
    even if a rerun is not necessary. If not provided, they will be seen
    only if the command is actually run.
    """
    def __init__(self, command,
                     infiles=[], outfiles=[],
                     inplace={}):
        self.command = ensure_list(command)
        self.infiles = ensure_list(infiles)
        self.outfiles = ensure_list(outfiles)
        if len(self.outfiles)==0:
            raise ValueError("No output files specified")
        self.inplace = dict(inplace)
        for (k,v) in self.inplace.items():
            if k not in self.infiles:
                raise ValueError("Parameter %s to modify inplace not listed"
                                     "among input parameters: %s"
                                     % (k,self.infiles))
            if v not in self.outfiles:
                raise ValueError("Destination parameter %s for inplace"
                                     "modification not listed among output"
                                     "parameters: %s"
                                     % (v,self.outfiles))

    def format_kwargs(self, kwargs):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        rerun = kwargs.pop("rerun", None)
        call_id = kwargs.pop("call_id", None)

        infiles = [kwargs[f] for f in self.infiles]
        outfiles = [kwargs[f] for f in self.outfiles]
        stdout_name, stderr_name, args_name = [outfiles[0]+"."+s
                                                   for s in ["stdout",
                                                             "stderr",
                                                             "args"]]
        infiles.append(args_name)
        outfiles.extend((stderr_name, stdout_name))

        if os.path.exists(args_name):
            old_args = pickle.load(open(args_name,"r"))
        else:
            info("No old arguments on record")
            old_args = ([], {})
        new_args = (args, kwargs)
        if new_args != old_args:
            info("Arguments changed")
            debug("%s != %s" % (new_args, old_args))
            with open(args_name,"w") as f:
                pickle.dump(new_args, f)

        if rerun or (rerun is None and need_rerun(infiles, outfiles)):
            success = False
            try:
                if self.inplace:
                    for (k, v) in self.inplace.items():
                        shutil.copy(kwargs[k],kwargs[v])
                        kwargs[k] = kwargs[v]
                        del kwargs[v]
                with open(stdout_name,"w") as stdout, \
                  open(stderr_name, "w") as stderr:
                    P = subprocess.Popen(self.command
                                             +list(args)
                                             +self.format_kwargs(kwargs),
                                             stdout=stdout.fileno(),
                                             stderr=stderr.fileno())
                    P.communicate()
                stdout = open(stdout_name,"r").read()
                stderr = open(stderr_name,"r").read()
                if P.returncode:
                    raise ValueError("Command %s failed with return code %d.\n"
                                         "stdout:\n%s"
                                         "stderr:\n%s"
                                        % (" ".join(self.command
                                                        +list(args)
                                                        +self.format_kwargs(kwargs)),
                                               P.returncode,
                                               stdout,
                                               stderr))
                sys.stdout.write(stdout)
                sys.stderr.write(stderr)
                success = True
            finally:
                if not success:
                    for f in outfiles:
                        try:
                            os.unlink(f)
                        except OSError as e:
                            sys.stderr.write("Problem deleting %s: %s" % (f,e))
        else: # no need to rerun
            sys.stdout.write(open(stdout_name).read())
            sys.stderr.write(open(stderr_name).read())

class FermiCommand(Command):
    def format_kwargs(self, kwargs):
        fmtkwargs = []
        for (k,v) in kwargs.items():
            fmtkwargs.append("%s=%s" % (k,v))
        return fmtkwargs

class Tempo2Command(Command):
    def format_kwargs(self, kwargs):
        fmtkwargs = []
        for (k,v) in kwargs.items():
            fmtkwargs.append("-%s" % k)
            fmtkwargs.append(str(v))
        return fmtkwargs

tempo2 = Tempo2Command("tempo2", infiles=["f", "ft1", "ft2"],
                           outfiles=["outfile"],
                           inplace={"ft1":"outfile"})

def add_photon_phases(parfile, infile, scfile, outfile,
                          rerun=None, call_id=None,
                          orbital=False, barycol=None,
                          column_name=None,
                          t2command="tempo2"):
        t2args = ["-gr", "fermi", # make sure this is first just in case
                  "-phase"]
        t2kwargs = {}
        if orbital:
            t2args.append("-ophase")
        if barycol:
            t2kwargs["barycol"] = barycol
        if column_name:
            t2kwargs["colname"] = column_name

        tempo2(t2args, f=parfile, graph=0,
                       ft1=infile, ft2=scfile,
                       outfile=outfile,
                       **t2kwargs)


class Source(object):
    def __init__(self, name, directory, coords=None):
        self.name = name
        self.directory = directory
        if not os.path.exists(self.directory):
            os.makedirs(self.directory)
        if coords is None:
            coords = get_icrs_coordinates(self.name)
        self.coords = coords
        self.ra = self.coords.ra.degree
        self.dec = self.coords.dec.degree

    def file(self, filename):
        return os.path.join(self.directory, filename)


