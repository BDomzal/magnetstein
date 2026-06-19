from .base_spectrum import *


class NMRSpectrum(BaseSpectrum):
    def __init__(self, confs=None, protons=None, label=None, **other):
        
        BaseSpectrum.__init__(self, confs=confs, label=label, **other)
        self.protons = protons


    def get_highest_peak(self):
        """
        Returns the peak with the highest intensity.
        """
        return max(self.confs, key=lambda x: x[1])


    def average_frequency(self):
        """
        Returns the average frequency.
        """
        norm = float(sum(x[1] for x in self.confs))
        return sum(x[0]*x[1]/norm for x in self.confs)


    def bin_to_nominal(self, nb_of_digits=0):
        """
        Rounds values on the horizontal axis to a given number of decimal digits.
        Works in situ, returns None.
        The default nb_of_digits is zero.
        """
        xcoord, ycoord = zip(*self.confs)
        xcoord = map(lambda x: x, xcoord)
        xcoord = (xcoord[0] + round(x-xcoord[0], nb_of_digits) for x in xcoord)
        xcoord = map(lambda x: x, xcoord)
        self.confs = list(zip(xcoord, ycoord))
        self.sort_confs()
        self.merge_confs()


    def coarse_bin(self, nb_of_digits):
        """
        Rounds the values on the horizontal axis to a given number of decimal digits
        """
        self.confs = [(round(x[0], nb_of_digits), x[1]) for x in self.confs]
        self.merge_confs()


    def distort_intensity(self, N, gain, sd):
        """
        Distorts the intensity measurement in a mutiplicative noise model - i.e.
        assumes that each molecule yields a random amount of signal.
        The resulting spectrum is not normalized.
        Works in situ (modifies self).
        N: int
            number of molecules
        gain: float
            mean amount of signal of one peak
        sd: float
            standard deviation of one peak

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


    def distort_horizontal_axis(self, mean, sd):
        """
        Distorts the measurement along horizontal axis by a normally distributed
        random variable with given mean and standard deviation.
        Use non-zero mean to approximate calibration error.
        Returns the applied shift.
        """
        N = rd.normal(mean, sd, len(self.confs))
        self.confs = [(x[0] + u, x[1]) for x, u in zip(self.confs, N)]
        self.sort_confs()
        self.merge_confs()
        return N


    def resample(self, target_location, distance_threshold=0.05):
        """
        Returns a resampled spectrum with intensity values approximated
        at points given by a sorted iterable target_location.
        The approximation is performed by a piecewise linear interpolation
        of the spectrum intensities. The spectrum needs to be in profile mode
        in order for this procedure to work properly.
        The spectrum is interpolated only if two target values closest to a
        given target_location are closer than the specified threshold
        This is done in order to interpolate the intensity only within peaks, not between them.
        If the surrounding values are further away than the threshold,
        it is assumed that the given target_location corresponds to the background and
        there is no intensity at that point.
        A rule-of-thumb is to set threshold as twice the distance between
        neighboring measurements along the horizontal axis.
        Large thresholds may lead to non-zero resampled intensity in the background,
        low thresholds might cause bad interpolation due to missing intensity values.
        """
        mz = [mz for mz, intsy in self.confs]
        intsy = [intsy for mz, intsy in self.confs]
        x = target_location[0]
        for m in target_location:
            assert m >= x, "The target_location list is not sorted!"
            x = m
        lenx = len(target_location)
        lent = len(mz)
        qi = 0  # query (x) index
        ti = 0  # target index - the first index s.t. mz[ti] >= x[qi]
        y = [0.]*lenx  # resampled intensities
        y0, y1 = intsy[0], intsy[0]  # intensities of target spectrum around the point target_location[qi]
        x0, x1 = mz[0], mz[0]  # mz around the point target_location[qi]
        # before mz starts, the intensity is zero:
        while target_location[qi] < mz[0]:
            qi += 1
        # interpolating:
        while ti < lent-1:
            ti += 1
            y0 = y1
            y1 = intsy[ti]
            x0 = x1
            x1 = mz[ti]
            while qi < lenx and target_location[qi] <= mz[ti]:
                # note: maybe in this case set one of the values to zero to get a better interpolation of edges
                if x1-x0 < distance_threshold:
                    y[qi] = y1 + (target_location[qi]-x1)*(y0-y1)/(x0-x1)
                qi += 1
        return self.__class__(confs = list(zip(target_location, y)))


    def gaussian_smoothing(self, sd=0.01, new_axis=0.01):
        """
        Applies a gaussian filter to the spectrum in order to smooth
        it out and decrease the electronic noise.
        Technically, each intensity measurement is replaced by a Gaussian weighted average
        of the neighbouring intensities.  
        As a consequence, the resolution gets decreased.
        Parameter sd (float) controls the width of the gaussian filter.
        Parameter new_axis (float or np.array) is the horizontal axis of the resulting smoothed spectrum.
        Setting it to float generates an equally spaced horizontal axis with new_axis being the step length.
        Setting it to np.array sets it as the resulting horizontal axis.  
        Note that after filtering, the area below curve (not the sum of intensities!)
        is equal to the area of the original spectrum in profile mode,
        or the sum of the input peak intensities in centroid mode.
        """
        if isinstance(new_axis, float):
            new_axis = np.arange(self.confs[0][0] - 4*sd, self.confs[-1][0] + 4*sd, new_axis)
        assert np.all(new_axis[1:] >= new_axis[:-1]), 'The new axis needs to be sorted!'
        smooth_intensity = np.zeros(new_axis.shape)
        for mz, intsy in self.confs:
            # smooth_intensity += intsy*np.exp(-(mz - new_axis)**2)**(1/(2*sd**2))
            lpid, rpid = np.searchsorted(new_axis, (mz - 4*sd, mz + 4*sd))
            peak_mz = new_axis[lpid:rpid]
            smooth_intensity[lpid:rpid] += intsy*np.exp(-(mz - peak_mz)**2)**(1/(2*sd**2))
        smooth_intensity /= np.sqrt(2*np.pi)*sd
        self.set_confs(list(zip(new_axis, smooth_intensity)))

   def plot(self, show = True, profile=True, linewidth=1, **plot_kwargs):
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
    def plot_all(spectra, show=True, profile=True, cmap=None, **plot_kwargs):
        """
        Shows the supplied list of spectra on a single plot.
        """
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm

        if not cmap:
            prop_cycle = plt.rcParams['axes.prop_cycle']
            colors = prop_cycle.by_key()['color']
        elif cmap == 'rainbow':
            colors = cm.rainbow(np.linspace(0, 1, len(spectra)))
            colors =  [[0, 0, 0, 0.8]] + [list(x[:3]) + [0.6] for x in colors]
        else:
            colors = [[0, 0, 0, 0.8]] + [cmap(x, alpha=1) for x in range(len(spectra))]

        for i, spectre in enumerate(spectra):
            spectre.plot(show=False, profile=profile, color = colors[i],
                         **plot_kwargs)

        #plt.legend(loc=9, bbox_to_anchor=(0.5, -0.1), ncol=len(spectra))  # legend below plot
        plt.legend(loc=0, ncol=1)
        if show:
            plt.show()