import math
import numpy as np
from scipy.stats import uniform, gamma
import random
import heapq
import re
from collections import Counter
import numpy.random as rd
from scipy.signal import argrelmax
from .peptides import get_protein_formula
from warnings import warn
from copy import deepcopy

class Spectrum:
    def __init__(self, formula='', threshold=0.001, total_prob=None,
                 charge=1, adduct=None, confs=None, label=None, **other):
        """Initialize a Spectrum class.

        Initialization can be done by setting a peak list.

        The initialized spectrum is not normalised. In order to do this use
        normalize method.

        Parameters
        ----------

        formula: str
            The chemical formula of the molecule. If present, used as label.
        threshold: float
            Lower threshold on the intensity of simulated peaks. Used when
            `formula` is not an empty string, ignored when `total_prob` is not
            None.
        total_prob: float
            Lower bound on the total probability of simulated peaks, i.e.
            fraction of all potential signal, which will be simulated. Used
            when `formula` is not an empty string. When not None, then
            `threshold` value is ignored.
        charge: int
            A charge of the ion.
        adduct: str
            The ionizing element. When not None, then formula is updated
            with `charge` number of adduct atoms.
        confs: list
            A list of tuples of chemical shift and intensity. Confs contains peaks of an
            initialized spectrum. 
        label: str
            An optional spectrum label.
        """
        ### TODO2: seprarate subclasses for centroid & profile spectra
        self.formula = formula
        self.empty = False

        if label is None:
            self.label = formula
        else:
            self.label = label

        self.charge = charge

        if confs is not None:
            self.set_confs(confs)
        else:
            self.empty = True
            self.confs = []

    @classmethod
    def new_from_fasta(cls, fasta, threshold=0.001, total_prob=None, intensity=1.0,
                       empty=False, charge=1, label=None):
        return cls(get_protein_formula(fasta), threshold=threshold,
                        total_prob=total_prob, intensity=intensity,
                        empty=empty, charge=charge, label=label)

    @classmethod
    def new_from_csv(cls, filename, delimiter=","):
        spectrum = cls(label=filename)

        with open(filename, "r") as infile:
            header = next(infile)
            for line in infile:
                if line[0] == '#':
                    continue
                line = line.strip()
                line = line.split(delimiter)
                spectrum.confs.append(tuple(map(float, line)))
        spectrum.sort_confs()
        spectrum.merge_confs()
        return spectrum

    @classmethod
    def new_random(cls, domain=(0.0, 1.0), peaks=10):
        ret = cls()
        confs = []
        for _ in range(peaks):
            confs.append((random.uniform(*domain), random.uniform(0.0, 1.0)))
        ret.set_confs(confs)
        return ret

    def copy(self):
        """
        Return a (deep) copy of self
        """
        return deepcopy(self)

    def get_modal_peak(self):
        """
        Returns the peak with the highest intensity.
        """
        return max(self.confs, key=lambda x: x[1])

    def sort_confs(self):
        """
        Sorts configurations by their chemical shift.
        """
        self.confs.sort(key = lambda x: x[0])

    def merge_confs(self):
        """
        Merges configurations with an identical chemical shift, summing their intensities.
        """
        if not self.empty:
            cppm = self.confs[0][0]
            cprob = 0.0
            ret = []
            for ppm, prob in self.confs + [(-1, 0)]:
                if ppm != cppm:
                    ret.append((cppm, cprob))
                    cppm = ppm
                    cprob = 0.0
                cprob += prob
            ### TODO3: for profile spectra, set a margin of max. 5 zero intensities
            ### around any observed intensity to preserve peak shape
            ### For centroid spectra, remove all zero intensities.
            #self.confs = [x for x in ret if x[1] > 1e-12]
            self.confs = ret

    def set_confs(self, confs):
        self.confs = confs
        if len(self.confs) > 0:
            self.sort_confs()
            self.merge_confs()
        else:
            self.empty = True

    def __add__(self, other):
        res = self.__class__()
        res.confs = self.confs + other.confs
        res.sort_confs()
        res.merge_confs()
        res.label = self.label + ' + ' + other.label
        return res

    def __mul__(self, number):
        res = self.__class__()
        res.set_confs([(x[0], number*x[1]) for x in self.confs])
        res.label = self.label
        return res

    def __rmul__(self, number):
        # Here * is commutative
        return self * number

    def __len__(self):
        return len(self.confs)

    @staticmethod
    def ScalarProduct(spectra, weights):
        ret = spectra[0].__class__()
        Q = [(spectra[i].confs[0], i, 0) for i in range(len(spectra))]
        heapq.heapify(Q)
        while Q != []:
            conf, spectre_no, conf_idx = heapq.heappop(Q)
            ret.confs.append((conf[0], conf[1] * weights[spectre_no]))
            conf_idx += 1
            if conf_idx < len(spectra[spectre_no]):
                heapq.heappush(Q, (spectra[spectre_no].confs[conf_idx], spectre_no, conf_idx))
        ret.merge_confs()
        return ret

    def normalize(self, target_value = 1.0):
        """
        Normalize the intensity values so that they sum up to the target value.
        """
        x = target_value/math.fsum(v[1] for v in self.confs)
        self.confs = [(v[0], v[1]*x) for v in self.confs]

    def WSDistanceMoves(self, other):
        """
        Return the optimal transport plan between self and other.
        """
        try:
            ii = 0
            leftoverprob = other.confs[0][1]
            for ppm, prob in self.confs:
                while leftoverprob <= prob:
                    yield (other.confs[ii][0], ppm, leftoverprob)
                    prob -= leftoverprob
                    ii += 1
                    leftoverprob = other.confs[ii][1]
                yield (other.confs[ii][0], ppm, prob)
                leftoverprob -= prob
        except IndexError:
            return

    def WSDistance(self, other):
        if not np.isclose(sum(x[1] for x in self.confs), 1.):
            raise ValueError('Self is not normalized.')
        if not np.isclose(sum(x[1] for x in other.confs), 1.):
            raise ValueError('Other is not normalized.')
        return math.fsum(abs(x[0]-x[1])*x[2] for x in self.WSDistanceMoves(other))

    def explained_intensity(self,other):
        """
        Returns the amount of mutual intensity between self and other,
        defined as sum of minima of intensities, chemical-shift-wise.
        """
        e = 0
        for i in range(len(self.confs)):
            e += min(self.confs[i][1],other.confs[i][1])
        return e

    def coarse_bin(self, nb_of_digits):
        """
        Rounds the chemical shift to a given number of decimal digits
        """
        self.confs = [(round(x[0], nb_of_digits), x[1]) for x in self.confs]
        self.merge_confs()

    def add_chemical_noise(self, nb_of_noise_peaks, noise_fraction, span=1.2):
        """
        Add additional peaks that simulate chemical noise.

        The method adds additional peaks with uniform distribution in the frequency domain
        domain and gamma distribution in the intensity domain. The spectrum
        does NOT need to be normalized. Accordingly, the method does not
        normalize the intensity afterwards! Works in-situ (on self).

        Parameters
        ----------
        nb_of_noise_peaks : int
            The number of added peaks.
        noise_fraction : float
            The amount of noise signal in the spectrum, >= 0 and <= 1.
        span: float or 2-tuple of floats
           If float, then `span` specifies a factor by which the chemical shift range is
           increased. If 2-tuple, then `span` specifies chemical shift range, which is
           noised.

        Returns
        -------
        None
        """
        if isinstance(span, (float, int)):
            span_increase = span
            prev_span = (min(x[0] for x in self.confs),
                         max(x[0] for x in self.confs))
            span_move = 0.5 * (span_increase - 1) * (prev_span[1] - prev_span[0])
            span = (max(prev_span[0] - span_move, 0),
                    prev_span[1] + span_move)
        noisex = uniform.rvs(loc=span[0], scale=span[1]-span[0],
                             size=nb_of_noise_peaks)
        noisey = gamma.rvs(a=2, scale=2, size=nb_of_noise_peaks)
        noisey /= sum(noisey)
        signal = sum(x[1] for x in self.confs)
        noisey *=  signal * noise_fraction / (1 - noise_fraction)
        noise = [(x, y) for x,y in zip(noisex, noisey)]
        self.confs.extend(noise)
        self.sort_confs()
        self.merge_confs()

    def add_gaussian_noise(self, sd):
        """
        Adds gaussian noise to each peak, simulating
        electronic noise.
        """
        noised = rd.normal([y for x,y in self.confs], sd)
        # noised = noised - min(noised)
        self.confs = [(x[0], y) for x, y in zip(self.confs, noised) if y > 0]

    def distort_intensity(self, N, gain, sd):
        """
        Distorts the intensity measurement in a mutiplicative noise model - i.e.
        assumes that each ion yields a random amount of signal.
        Assumes the molecule is composed of one element, so it's
        an approximation for normal molecules.
        The resulting spectrum is not normalized.
        Works in situ (modifies self).
        N: int
            number of ions
        gain: float
            mean amount of signal of one ion
        sd: float
            standard deviation of one ion's signal

        Return: np.array
            The applied deviations.
        """
        p = np.array([x[1] for x in self.confs])
        assert np.isclose(sum(p), 1), 'Spectrum needs to be normalized prior to distortion'
        X = [(x[0], N*gain*x[1]) for x in self.confs]  # average signal
        peakSD = np.sqrt(N*sd**2*p + N*gain**2*p*(1-p))
        U = rd.normal(0, 1, len(X))
        U *= peakSD
        X = [(x[0], max(x[1] + u, 0.)) for x, u in zip(X, U)]
        self.confs = X
        return U

    def distort_ppm(self, mean, sd):
        """
        Distorts the chemical shift measurement by a normally distributed
        random variable with given mean and standard deviation.
        Use non-zero mean to approximate calibration error.
        Returns the applied shift.
        """
        N = rd.normal(mean, sd, len(self.confs))
        self.confs = [(x[0] + u, x[1]) for x, u in zip(self.confs, N)]
        self.sort_confs()
        self.merge_confs()
        return N

    @staticmethod
    def sample_multinomial(reference, N, gain, sd):
        """
        Samples a spectrum of N molecules based on peak probabilities
        from the reference spectrum. Simulates both isotope composition
        and amplifier randomness.
        The returned spectrum is not normalized.
        N: int
            number of ions in the spectrum
        gain: float
            The gain of the amplifier, i.e. average signal from one ion
        sd: float
            Standard deviation of one ion's signal
        """
        p = [x[1] for x in reference.confs]
        assert np.isclose(sum(p), 1), 'Spectrum needs to be normalized prior to sampling'
        U = rd.multinomial(N, p)
        U = rd.normal(U*gain, np.sqrt(U*sd**2))
        retSp = Spectrum('', empty=True, label='Sampled ' + reference.label)
        retSp.set_confs([(x[0], max(u, 0.)) for x, u in zip(reference.confs, U)])
        return retSp

    def find_peaks(self):
        """
        Returns a list of local maxima.
        Each maximum is reported as a tuple of ppm and intensity.
        The last and final configuration is never reported as a maximum.
        Note that this function should only be applied to profile spectra - the result
        does not make sense for centroided spectrum.
        Applying a gaussian or Savitzky-Golay filter prior to peak picking
        is advised in order to avoid detection of noise.
        """
        diffs = [n[1]-p[1] for n,p in zip(self.confs[1:], self.confs[:-1])]
        is_max = [nd <0 and pd > 0 for nd, pd in zip(diffs[1:], diffs[:-1])]
        peaks = [x for x, p in zip(self.confs[1:-1], is_max) if p]
        return peaks

    def trim_negative_intensities(self):
        """
        Detects negative intensity measurements and sets them to 0.
        """
        self.confs = [(ppm, intsy if intsy >= 0 else 0.) for ppm, intsy in self.confs]

    def centroid(self, max_width, peak_height_fraction=0.5):
        """Return confs of a centroided spectrum.

        The function identifies local maxima of intensity and integrates peaks in the regions
        delimited by peak_height_fraction of the apex intensity.
        By default, for each peak the function will integrate the region delimited by the full width at half maximum.
        If the detected region is wider than max_width, the peak is considered as noise and discarded.
        Small values of max_width tend to miss peaks, while large ones increase computational complexity
        and may lead to false positives.

        Note that this function should only be applied to profile spectra - the result
        does not make sense for centroided spectrum.
        Applying a gaussian or Savitzky-Golay filter prior to peak picking
        is advised in order to avoid detection of noise.

        Returns
        -----------------
            A tuple of two peak lists that can be used to construct a new Spectrum object.
            The first list contains configurations of centroids (i.e. centers of mass and areas of peaks).
            The second list contains configurations of peak apices corresponding to the centroids
            (i.e. locations and heights of the local maxima of intensity.)
        """
        ### TODO: change max_width to be in ppm?
        # Validate the input:
        if any(intsy < 0 for ppm, intsy in self.confs):
            warn("""
                 The spectrum contains negative intensities!
                 It is advised to use Spectrum.trim_negative_intensities() before any processing
                 (unless you know what you're doing).
                 """)

        # Transpose the confs list to get an array of ppms and an array of intensities:
        ppm, intsy = np.array(self.confs).T

        # Find the local maxima of intensity:
        peak_indices = argrelmax(intsy)[0]

        peak_ppm = []
        peak_intensity = []
        centroid_ppm = []
        centroid_intensity = []
        max_dist = max_width/2.
        n = len(ppm)
        for p in peak_indices:
            current_ppm = ppm[p]
            current_intsy = intsy[p]
            # Compute peak centroids:
            target_intsy = peak_height_fraction*current_intsy
            right_shift = 1
            left_shift = 1
            # Get the ppm points bounding the peak fragment to integrate.
            # First, go to the right from the detected apex until one of the four conditions are met:
            # 1. we exceed the ppm range of the spectrum
            # 2. we exceed the maximum distance from the apex given by max_dist
            # 3. the intensity exceeds the apex intensity (meaning that we've reached another peak)
            # 4. we go below the threshold intensity (the desired stopping condition)
            # Note: in step 3, an alternative is to check if the intensity simply starts to increase w.r.t. the previous inspected point.
            # Such an approach may give less false positive peaks, but is very sensitive to electronic noise and to overlapping peaks.
            # When we check if the intensity has not exceeded the apex intensity, and we encounter a cluster of overlapping peaks,
            # then we will effectively consider the highest one as the true apex of the cluster and integrate the whole cluster only once.
            while p + right_shift < n-1 and ppm[p+right_shift] - ppm[p] < max_dist and intsy[p+right_shift] <= current_intsy and intsy[p+right_shift] > target_intsy:
                right_shift += 1
            # Get the ppm values of points around left ppm value of the peak boundary (which will be interpolated):
            rx1, rx2 = ppm[p+right_shift-1], ppm[p+right_shift]
            ry1, ry2 = intsy[p+right_shift-1], intsy[p+right_shift]
            if not ry1 >= target_intsy >= ry2:
                # warn('Failed to find the right boundary of the peak at %f (probably found an overlapping peak)' % current_ppm)
                continue
            # Find the left boundary of the peak:
            while p - left_shift > 1 and ppm[p] - ppm[p-left_shift] < max_dist and intsy[p-left_shift] <= current_intsy and intsy[p-left_shift] > target_intsy:
                left_shift += 1
            lx1, lx2 = ppm[p-left_shift], ppm[p-left_shift+1]
            ly1, ly2 = intsy[p-left_shift], intsy[p-left_shift+1]
            if not ly1 <= target_intsy <= ly2:
                # warn('Failed to find the left boundary of the peak at %f (probably found an overlapping peak)' % current_ppm)
                continue
            # Interpolate the ppm values actually corresponding to peak_height_fraction*current_intsy:
            lx = (target_intsy-ly1)*(lx2-lx1)/(ly2-ly1) + lx1
            if not lx1 <= lx <= lx2:
                raise RuntimeError('Failed to interpolate the left boundary ppm value of the peak at %f' % current_ppm)
            rx = (target_intsy-ry1)*(rx2-rx1)/(ry2-ry1) + rx1
            if not rx1 <= rx <= rx2:
                raise RuntimeError('Failed to interpolate the right boundary ppm value of the peak at %f' % current_ppm)
            # Join the interpolated boundary with the actual measurements:
            x = np.hstack((lx, ppm[(p-left_shift+1):(p+right_shift)], rx))
            y = np.hstack((target_intsy, intsy[(p-left_shift+1):(p+right_shift)], target_intsy))
            # Integrate the area:
            cint = np.trapz(y, x)
            cppm = np.trapz(y*x, x)/cint
            if cppm not in centroid_ppm:  # intensity errors may introduce artificial peaks
                centroid_ppm.append(cppm)
                centroid_intensity.append(cint)
                # Store the apex data:
                peak_ppm.append(current_ppm)
                peak_intensity.append(current_intsy)
        return(list(zip(centroid_ppm, centroid_intensity)), list(zip(peak_ppm, peak_intensity)))

    def resample(self, target_ppm, ppm_distance_threshold=0.05):
        """
        Returns a resampled spectrum with intensity values approximated
        at points given by a sorted iterable target_ppm.
        The approximation is performed by a piecewise linear interpolation
        of the spectrum intensities. The spectrum needs to be in profile mode
        in order for this procedure to work properly.
        The spectrum is interpolated only if two target ppm values closest to a
        given target ppm are closer than the specified threshold
        This is done in order to interpolate the intensity only within peaks, not between them.
        If the surrounding ppm values are further away than the threshold,
        it is assumed that the given target ppm corresponds to the background and
        there is no intensity at that point.
        A rule-of-thumb is to set threshold as twice the distance between
        neighboring ppm measurements.
        Large thresholds may lead to non-zero resampled intensity in the background,
        low thresholds might cause bad interpolation due to missing intensity values.
        """
        ppm = [ppm for ppm, intsy in self.confs]
        intsy = [intsy for ppm, intsy in self.confs]
        x = target_ppm[0]
        for m in target_ppm:
            assert m >= x, "The target_ppm list is not sorted!"
            x = m
        lenx = len(target_ppm)
        lent = len(ppm)
        qi = 0  # query (x) index
        ti = 0  # target index - the first index s.t. ppm[ti] >= x[qi]
        y = [0.]*lenx  # resampled intensities
        y0, y1 = intsy[0], intsy[0]  # intensities of target spectrum around the point target_ppm[qi]
        x0, x1 = ppm[0], ppm[0]  # ppm around the point target_ppm[qi]
        # before ppm starts, the intensity is zero:
        while target_ppm[qi] < ppm[0]:
            qi += 1
        # interpolating:
        while ti < lent-1:
            ti += 1
            y0 = y1
            y1 = intsy[ti]
            x0 = x1
            x1 = ppm[ti]
            while qi < lenx and target_ppm[qi] <= ppm[ti]:
                # note: maybe in this case set one of the values to zero to get a better interpolation of edges
                if x1-x0 < ppm_distance_threshold:
                    y[qi] = y1 + (target_ppm[qi]-x1)*(y0-y1)/(x0-x1)
                qi += 1
        return self.__class__(confs = list(zip(target_ppm, y)))


    def fuzzify_peaks(self, sd, step):
        """
        LEGACY FUNCTION. USE SELF.GAUSSIAN_SMOOTING INSTEAD.   
        Applies a gaussian filter to the peaks, effectively broadening them
        and simulating low resolution. Works in place, modifying self.
        The parameter step gives the distance between samples in chemical shift axis.
        After the filtering, the area below curve (not the sum of intensities!)
        is equal to the sum of the input peak intensities.
        """
        new_ppm = np.arange(self.confs[0][0] - 4*sd, self.confs[-1][0] + 4*sd, step)
        A = new_ppm[:,np.newaxis] - np.array([m for m,i in self.confs])
        # we don't need to evaluate gaussians to far from their mean,
        # from our perspective 4 standard deviations from the mean is the same
        # as the infinity; this allows to avoid overflow as well:
        A[np.abs(A) > 4*sd] = np.inf
        A **= 2
        A /= (-2*sd**2)
        A = np.exp(A)
        new_intensity = A @ np.array([i for m,i in self.confs])  # matrix multiplication
        new_intensity /= (np.sqrt(2*np.pi)*sd)
        self.set_confs(list(zip(new_ppm, new_intensity)))


    def gaussian_smoothing(self, sd=0.01, new_ppm=0.01):
        """
        Applies a gaussian filter to the spectrum in order to smooth
        it out and decrease the electronic noise.
        Technically, each intensity measurement is replaced by a Gaussian weighted average
        of the neighbouring intensities.  
        As a consequence, the resolution gets decreased.
        Parameter sd (float) controls the width of the gaussian filter.
        Parameter new_ppm (float or np.array) is the chemical shift axis of the resulting smoothed spectrum.
        Setting it to float generates an equally spaced chemical shift axis with new_ppm being the step length.
        Setting it to np.array sets it as the resulting chemical shift axis.  
        Note that after filtering, the area below curve (not the sum of intensities!)
        is equal to the area of the original spectrum in profile mode,
        or the sum of the input peak intensities in centroid mode.
        """
        if isinstance(new_ppm, float):
            new_ppm = np.arange(self.confs[0][0] - 4*sd, self.confs[-1][0] + 4*sd, new_ppm)
        assert np.all(new_ppm[1:] >= new_ppm[:-1]), 'The new ppm axis needs to be sorted!'
        smooth_intensity = np.zeros(new_ppm.shape)
        for ppm, intsy in self.confs:
            # smooth_intensity += intsy*np.exp(-(ppm - new_ppm)**2)**(1/(2*sd**2))
            lpid, rpid = np.searchsorted(new_ppm, (ppm - 4*sd, ppm + 4*sd))
            peak_ppm = new_ppm[lpid:rpid]
            smooth_intensity[lpid:rpid] += intsy*np.exp(-(ppm - peak_ppm)**2)**(1/(2*sd**2))
        smooth_intensity /= np.sqrt(2*np.pi)*sd
        self.set_confs(list(zip(new_ppm, smooth_intensity)))


    def cut_smallest_peaks(self, removed_proportion=0.001):
        """
        Removes smallest peaks until the total removed intensity amounts
        to the given proportion of the total signal in the spectrum.
        """
        self.confs.sort(key = lambda x: x[1], reverse=True)
        threshold  = removed_proportion*sum(x[1] for x in self.confs)
        removed = 0
        while len(self.confs)>0 and removed + self.confs[-1][1] <= threshold:
            removed += self.confs.pop()[1]
        self.confs.sort(key = lambda x: x[0])


    def filter_against_other(self, others, margin=0.15):
        """
        Remove signal from the spectrum which is far from other spectra signal.

        This method removes peaks from self spectrum which position is outside the
        area of any peak of other spectra +/- margin. The method does not
        modify self spectrum and returns a new instance of the filtered
        spectrum.

        Parameters
        ----------
        self
            Spectrum to be filtered.
        others:
            One instance of the spectrum against self is filtered or iterable of
            instances of other spectra.
        margin
            chemical shift radius within signal should be left.

        Returns
        -------
        Spectrum
            A new spectrum with filtered out peaks.

        """
        try:
            other_confs = []
            for other_spectrum in others:
                other_confs.extend(other_spectrum.confs)
            other = self.__class__(confs=other_confs)
        except TypeError:
            other = others
        other_ppms = [i[0] for i in other.confs]


        result_confs = []
        index = 0
        for ppm, abund in self.confs:
            while (index + 1 < len(other_ppms) and
                   other_ppms[index + 1] < ppm):
                index += 1
            if abs(ppm - other_ppms[index]) <= margin or (
                    index + 1 < len(other_ppms) and
                    abs(ppm - other_ppms[index + 1]) <= margin):
                result_confs.append((ppm, abund))

        result_spectrum = self.__class__(confs=result_confs, label=self.label)
        return result_spectrum

    def plot(self, show = True, profile=False, linewidth=1, **plot_kwargs):
        """
        Plots the spectrum.
        The keyword argument show is retained for backwards compatibility.
        """
        import matplotlib.pyplot as plt
        if profile:
            plt.plot([x[0] for x in self.confs], [x[1] for x in self.confs],
                     linestyle='-', linewidth=linewidth, label=self.label, **plot_kwargs)
        else:
            plt.vlines([x[0] for x in self.confs], [0],
                       [x[1] for x in self.confs], label = self.label,
                       linewidth=linewidth, **plot_kwargs)
        if show:
            plt.show()

    @staticmethod
    def plot_all(spectra, show=True, profile=False, cmap=None, **plot_kwargs):
        """
        Shows the supplied list of spectra on a single plot.
        """
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
        if not cmap:
            colors = cm.rainbow(np.linspace(0, 1, len(spectra)))
            colors =  [[0, 0, 0, 0.8]] + [list(x[:3]) + [0.6] for x in colors]
        else:
            try:
                colors = [[0, 0, 0, 0.8]] + [cmap(x, alpha=1) for x in range(len(spectra))]
            except:
                colors = cmap
        i = 0
        for spectre in spectra:
            spectre.plot(show=False, profile=profile, color = colors[i],
                         **plot_kwargs)
            i += 1
        #plt.legend(loc=9, bbox_to_anchor=(0.5, -0.1), ncol=len(spectra))  # legend below plot
        plt.legend(loc=0, ncol=1)
        if show:
            plt.show()

