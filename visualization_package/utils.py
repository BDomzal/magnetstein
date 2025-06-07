import numpy as np
import pandas as pd
import pulp
from matplotlib.colors import LinearSegmentedColormap
from masserstein import NMRSpectrum, estimate_proportions

#Function written by Krzysztof Zakrzewski (B. Miasojedow's master's student)
def shift_and_mix(spectra, shifts, probs):
    """
    Shifts and scales individual spectra, then mixes them into a single NMR spectrum.

    Args:
        spectra (list): List of spectrum-like objects, each with a `.confs` attribute containing (ppm, intensity) tuples.
        shifts (list): List of float shifts to apply to each spectrum (e.g., in ppm).
        probs (list): List of float scaling factors (e.g., estimated proportions) to apply to intensities of each spectrum.

    Returns:
        NMRSpectrum: A normalized and trimmed mixed NMR spectrum.
    """
    # Extract .confs from each spectrum
    confs = [spectrum.confs for spectrum in spectra]
    confs_nz = [[] for _ in spectra]  # Store non-zero, shifted, and scaled data

    # Shift x-values and scale y-values by their proportions
    for i, conf in enumerate(confs):
        for point in conf:
            point_x = point[0] + shifts[i]
            point_y = point[1] * probs[i]
            if point_y > 0:
                confs_nz[i].append((point_x, point_y))

    # Define helper to merge two sorted spectra
    def mix_2(confs1, confs2):
        n1, i1 = len(confs1), 0
        n2, i2 = len(confs2), 0
        confs_2mixed = []

        while i1 < n1 and i2 < n2:
            if confs1[i1][0] == confs2[i2][0]:
                confs_2mixed.append((confs1[i1][0], confs1[i1][1] + confs2[i2][1]))
                i1 += 1
                i2 += 1
            elif confs1[i1][0] < confs2[i2][0]:
                confs_2mixed.append(confs1[i1])
                i1 += 1
            else:
                confs_2mixed.append(confs2[i2])
                i2 += 1

        # Append remaining points
        confs_2mixed.extend(confs1[i1:])
        confs_2mixed.extend(confs2[i2:])
        return confs_2mixed

    # Recursively combine all spectra using divide-and-conquer
    def mix_recursive(confs, left, right):
        if left == right:
            return confs[left]
        mid = left + (right - left) // 2
        return mix_2(
            mix_recursive(confs, left, mid),
            mix_recursive(confs, mid + 1, right)
        )

    confs_mixed = mix_recursive(confs_nz, 0, len(confs_nz) - 1)

    # Build and normalize resulting NMRSpectrum
    result = NMRSpectrum(confs=confs_mixed)
    result.trim_negative_intensities()
    result.normalize()
    return result


def retrieve_transport_plan(mix, spectra, interesting_region, kappa_mixture, kappa_components, solver=pulp.LpSolverDefault, log=True):
    """
    Computes the Wasserstein transport plan between a mixture and its estimated components,
    and returns the transport matrix within a region of interest.

    Args:
        mix (NMRSpectrum): The mixture spectrum.
        spectra (list of NMRSpectrum): Component spectra.
        interesting_region (tuple): (lower_ppm, upper_ppm) region to focus on.
        kappa_mixture (float): Regularization parameter for mixture noise.
        kappa_components (float): Regularization parameter for component noise.
        log (bool): If True, log-transform transport masses for better visualization.

    Returns:
        transport_df (pd.DataFrame): Transport matrix (from_ppm x to_ppm).
        mix_confs (np.ndarray): Mixture spectrum in the region of interest.
        wsom_confs (np.ndarray): Weighted sum of components in the region.
        distances (dict): Dictionary of transport distances with noise markers.
    """
    # Estimate proportions using optimal transport
    estimation = estimate_proportions(
        mix, spectra,
        MTD=kappa_mixture,
        MTD_th=kappa_components,
        verbose=True,
        solver=solver,
        what_to_compare='area'
    )

    common_horizontal_axis = estimation['common_horizontal_axis']
    updated = []

    # Pad missing ppm values in all spectra to match the common axis
    for sp in [mix] + spectra:
        if len(sp.confs) != len(common_horizontal_axis):
            missing = set(common_horizontal_axis) - {el[0] for el in sp.confs}
            new_confs = sp.confs + [(m, 0.0) for m in missing]
            updated.append(NMRSpectrum(confs=new_confs))
        else:
            updated.append(sp)

    mix = updated[0]
    spectra = updated[1:]

    # Normalize mixture to account for estimated noise
    p0 = 1 - sum(estimation['proportions'])
    p0_prime = estimation['proportion_of_noise_in_components']
    mix.normalize(target_value=1 - p0_prime)

    # Normalize each component according to its estimated proportion
    for i, sp in enumerate(spectra):
        sp.normalize(estimation['proportions'][i])

    # Combine all component intensities into one spectrum
    all_confs = [conf for sp in spectra for conf in sp.confs]
    weighted_sum_of_components = NMRSpectrum(confs=all_confs)

    # Restore component normalization for later reuse
    for sp in spectra:
        sp.normalize(1)

    # Subtract estimated noise from both mixture and components
    noise_in_mix = estimation['noise']
    noise_in_components = estimation['noise_in_components']
    mix_intensities = np.array([pt[1] for pt in mix.confs]) - np.array(noise_in_mix)
    mix = NMRSpectrum(confs=list(zip(common_horizontal_axis, mix_intensities)))
    wsom_intensities = np.array([pt[1] for pt in weighted_sum_of_components.confs]) - np.array(noise_in_components)
    weighted_sum_of_components = NMRSpectrum(confs=list(zip(common_horizontal_axis, wsom_intensities)))

    # Compute optimal transport plan between mixture and weighted sum of components
    transport_plan = list(mix.WSDistanceMoves(weighted_sum_of_components))

    # Extract all ppm values involved in the transport
    ppm_from = [f for f, _, _ in transport_plan]
    ppm_to = [t for _, t, _ in transport_plan]
    all_ppm = sorted(set(ppm_from + ppm_to), reverse=True)

    # Filter to only include points within the interesting region
    lower_lim, upper_lim = interesting_region
    in_region = [(lower_lim < ppm < upper_lim) for ppm in all_ppm]
    all_ppm = [ppm for ppm, keep in zip(all_ppm, in_region) if keep]

    # Initialize transport matrix
    transport_df = pd.DataFrame(0.0, index=all_ppm, columns=all_ppm)
    distances = {}

    # Fill transport matrix
    for from_ppm, to_ppm, mass in transport_plan:
        if mass > 0 and from_ppm in all_ppm and to_ppm in all_ppm:
            if log:
                value = -np.log10(mass)
            else:
                value = mass
            transport_df.loc[from_ppm, to_ppm] = value
            distances[abs(from_ppm - to_ppm)] = distances.get(abs(from_ppm - to_ppm), 0) + value

    # Add noise terms to distances
    if log:
        distances[kappa_mixture] = -np.log10(noise_in_mix)
        distances[kappa_components] = -np.log10(noise_in_components)
    else:
        distances[kappa_mixture] = noise_in_mix
        distances[kappa_components] = noise_in_components

    # Crop mixture and component spectra to region of interest
    mix_confs = np.array(mix.confs)
    mix_confs = mix_confs[(mix_confs[:, 0] > lower_lim) & (mix_confs[:, 0] < upper_lim)]
    mix_confs = np.flip(mix_confs, axis=0)

    wsom_confs = np.array(weighted_sum_of_components.confs)
    wsom_confs = wsom_confs[(wsom_confs[:, 0] > lower_lim) & (wsom_confs[:, 0] < upper_lim)]
    wsom_confs = np.flip(wsom_confs, axis=0)

    return transport_df, mix_confs, wsom_confs, distances