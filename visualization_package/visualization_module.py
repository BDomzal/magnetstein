import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from masserstein import NMRSpectrum, estimate_proportions

def visualize_transport_plan(
    transport_df,
    mix_confs,
    wsom_confs,
    experiment_name=None,
    lower_lim=None,
    upper_lim=None,
    figures_path=None,
    variant=None,
    cmap='hot_r',
    point_scaling=20,
    show_colorbar=True,
    save=True,
    figsize=(14, 12),
    title="Transport Plan",
    *args,
    **kwargs
):
    """
    Visualizes the transport plan using scatter and line plots.

    Args:
        transport_df (pd.DataFrame): Transport matrix (2D) to visualize.
        mix_confs (np.ndarray): Coordinates of mixture components.
        wsom_confs (np.ndarray): Coordinates of WSOM components.
        experiment_name (str, optional): Prefix for saving the plot.
        lower_lim (int, optional): Lower limit index for region label in filename.
        upper_lim (int, optional): Upper limit index for region label in filename.
        figures_path (str, optional): Directory to save the figure.
        variant (int, optional): Variant index to include in filename.
        cmap (str): Matplotlib colormap name.
        point_scaling (int): Scaling factor for point size in scatter plot.
        show_colorbar (bool): Whether to display the colorbar.
        save (bool): Whether to save the figure.
        figsize (tuple): Size of the entire figure.
        title (str): Title of the visualization.
        *args, **kwargs: Additional args passed to `scatter()`.

    Returns:
        None
    """
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(2, 3, width_ratios=[1, 5, 0.2], height_ratios=[1, 5],
                           wspace=0.1, hspace=0.1)

    base_cmap = plt.cm.get_cmap(cmap)
    colors = base_cmap(np.linspace(0, 1, 256))
    colors[0] = [1, 1, 1, 1]
    custom_cmap = LinearSegmentedColormap.from_list('custom_cmap', colors)

    # # Heatmap subplot
    ax_heatmap = plt.subplot(gs[1, 1])
    y_idx, x_idx = np.nonzero(transport_df.values)
    values = transport_df.values[y_idx, x_idx]


    sc = ax_heatmap.scatter(
        x_idx, y_idx,
        c=values,
        cmap=custom_cmap,
        s=values * point_scaling / values.max(),
        edgecolors='k',
        linewidths=0.0,
        *args, **kwargs
    )

    ax_heatmap.set_xlim(-0.5, transport_df.shape[1]-0.5)
    ax_heatmap.set_ylim(transport_df.shape[0]-0.5, -0.5)
    ax_heatmap.set_xticks([])
    ax_heatmap.set_yticks([])
    ax_heatmap.set_title(title)

    # Left subplot (WSOM)
    ax_left = plt.subplot(gs[1, 0])
    ax_left.plot(wsom_confs[:, 1], wsom_confs[:, 0], color='black', linewidth=1)
    ax_left.invert_xaxis()
    ax_left.set_xticks([])
    ax_left.set_ylabel('Combination of components', size=12)

    # Top subplot (Mixture)
    ax_top = plt.subplot(gs[0, 1])
    ax_top.plot(mix_confs[:, 0], mix_confs[:, 1], color='black', linewidth=1)
    ax_top.invert_xaxis()
    ax_top.xaxis.set_label_position('top')
    ax_top.xaxis.tick_top()
    ax_top.set_yticks([])
    ax_top.set_title('Mixture', size=12)

    # Align axes limits
    intensity_min = min(ax_left.get_xlim()[1], ax_top.get_ylim()[0])
    intensity_max = max(ax_left.get_xlim()[0], ax_top.get_ylim()[1])
    ax_left.set_xlim(intensity_max, intensity_min)
    ax_top.set_ylim(intensity_min, intensity_max)

    # Colorbar subplot
    if show_colorbar:
        ax_cbar = plt.subplot(gs[1, 2])
        cbar = plt.colorbar(sc, cax=ax_cbar)
        cbar.ax.get_yaxis().labelpad = 30
        cbar.set_ticks([])
        cbar.set_label('Amount of transport', rotation=270, size=12)

    # Save figure
    if save and figures_path is not None and experiment_name is not None and lower_lim is not None and upper_lim is not None:
        filename = f"{experiment_name}_region_{lower_lim}_{upper_lim}"
        if variant is not None:
            filename = f"{experiment_name}_variant_{variant+1}_region_{lower_lim}_{upper_lim}"
        plt.savefig(f"{figures_path}/{filename}.png", dpi=300, bbox_inches='tight')

    plt.show()


def visualize_transport_distance_distribution(
    distances,
    component_kappa=None,
    mixture_kappa=None,
    component_label="Kappa components",
    mixture_label="Kappa mixture",
    component_color="hotpink",
    mixture_color="cornflowerblue",
    bins=100,
    figsize=(8, 6),
    title="Transport Distance Distribution",
    save_path=None
):
    """
    Plots a histogram of transport distances with optional markers for specific kappa values.

    Args:
        distances (dict): Dictionary of transport distances with noise markers.
        component_kappa (float, optional): Value for a component kappa to highlight.
        mixture_kappa (float, optional): Value for a mixture kappa to highlight.
        component_label (str): Label for the component kappa line.
        mixture_label (str): Label for the mixture kappa line.
        component_color (str): Color for the component kappa markers. Color should be in Matplotlib CSS Colors domain.
        mixture_color (str): Color for the mixture kappa markers. Color should be in Matplotlib CSS Colors domain.
        bins (int): Number of histogram bins.
        figsize (tuple): Size of the figure.
        title (str): Title of the plot.
        save_path (str, optional): Path to save the plot. If None, plot is not saved.

    Returns:
        None
    """
    fig, ax = plt.subplots(figsize=figsize)

    if component_kappa is not None:
        y_comp = distances.get(component_kappa, 0)
        distances[component_kappa] = 0

    if mixture_kappa is not None:
        y_mix = distances.get(mixture_kappa, 0)
        distances[mixture_kappa] = 0

    counts, bins_, _ = ax.hist(distances, bins=bins, color='gray', edgecolor='black')

    bin_width = bins_[1] - bins_[0]

    # Highlight component kappa
    if component_kappa is not None:
        ax.bar(component_kappa, y_comp, width=bin_width,
               color=component_color, edgecolor=component_color, zorder=5)
        ax.axvline(component_kappa, color=component_color, linestyle='--', linewidth=1, zorder=6)
        ax.text(component_kappa, max(counts) * 0.5, component_label,
                rotation=90, color=component_color, va='bottom', ha='left', fontsize=12)

    # Highlight mixture kappa
    if mixture_kappa is not None:
        ax.bar(mixture_kappa, y_mix, width=bin_width,
               color=mixture_color, edgecolor=mixture_color, zorder=5)
        ax.axvline(mixture_kappa, color=mixture_color, linestyle='--', linewidth=1, zorder=6)
        ax.text(mixture_kappa, max(counts) * 0.5, mixture_label,
                rotation=90, color=mixture_color, va='bottom', ha='right', fontsize=12)

    # Axis and labels
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Transport distance, ppm", fontsize=14)
    ax.set_ylabel("Amount of signal transported", fontsize=14)
    ax.tick_params(labelsize=12)

    # Save or show
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()


def visualize_spectra(
    mixture,
    spectra_object,
    probs,
    components_names,
    window,
    shift=None,
    cumulate=True,
    figsize=(15, 9),
    title="Spectral Decomposition",
    save_path=None
):
    """
    Visualizes the spectral decomposition of a mixture into its component spectra.

    Supports both cumulative plotting using `fill_between` and standard stack plotting,
    with optional spectral shifting and moving average smoothing.

    Args:
        mixture (NMRSpectrum): An object representing the mixture spectrum, with a `.confs` attribute,
                 where each element is a (ppm, intensity) tuple.
        spectra_object (list of NMRSpectrum): List of component spectrum objects, each with `.confs`.
        probs (list of float): Scaling factors (e.g., probabilities or weights) for each component.
        components_names (list of str): Names of each component for the legend.
        window (int): Window size for moving average smoothing.
        shift (list of float, optional): Horizontal shift (e.g., ppm offset) for each component.
        cumulate (bool): If True, plot a cumulative filled spectrum using `fill_between`.
                         If False, plot standard stacked spectra using `stackplot`.
        figsize (tuple of int): Figure size in inches.
        title (str): Title of the plot.

    Returns:
        None
    """
    fig, ax = plt.subplots(figsize=figsize)

    if cumulate:
        cumulative = None
        for i, spectrum in enumerate(spectra_object):
            # Prepare x and y data
            x = np.array([pt[0] + shift[i] if shift else pt[0] for pt in spectrum.confs])
            y = np.array([probs[i] * pt[1] for pt in spectrum.confs])
            y_smoothed = np.convolve(y, np.ones(window) / window, mode='same')

            # Sort for proper plotting
            sort_idx = np.argsort(x)
            x_sorted = x[sort_idx]
            y_sorted = y_smoothed[sort_idx]

            base = np.zeros_like(y_sorted) if cumulative is None else cumulative
            ax.fill_between(x_sorted, base, base + y_sorted, label=components_names[i])
            cumulative = base + y_sorted

    else:
        for i, spectrum in enumerate(spectra_object):
            x = [pt[0] + shift[i] if shift else pt[0] for pt in spectrum.confs]
            y = [probs[i] * pt[1] for pt in spectrum.confs]
            y_smoothed = np.convolve(y, np.ones(window) / window, mode='same')
            ax.stackplot(x, y_smoothed, labels=[components_names[i]])

    # Plot the original mixture spectrum on top
    mix_x = np.array([pt[0] for pt in mixture.confs])
    mix_y = np.array([pt[1] for pt in mixture.confs])
    mix_y_smoothed = np.convolve(mix_y, np.ones(window) / window, mode='same')

    sort_idx = np.argsort(mix_x)
    ax.plot(mix_x[sort_idx], mix_y_smoothed[sort_idx], label="Mixture", color='black', linewidth=2)

    # Final plot formatting
    ax.legend()
    ax.set_title(title)
    ax.set_xlabel("Chemical shift (ppm)")
    ax.set_ylabel("Intensity")
    ax.invert_xaxis()

    # Save or show
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()


def visualize_stacked_spectra(spectra, colors=None, labels=None,
                         scale_factor=1.2, figsize=(10, 6), xlim=None, title=None):
    """
    Plot stacked NMR spectra with vertical offsets based on each spectrum's max intensity.
    Spectra must have a `.confs` attribute with (ppm, intensity) pairs.

    Args:
        spectra (list of NMRSpectrum): List of spectral objects with `.confs` attribute.
        colors (list of str, optional): Plot color per spectrum. Colors should be in Matplotlib CSS Colors domain.
        labels (list of str, optional): Labels for the legend.
        scale_factor (float): Vertical spacing multiplier between spectra.
        figsize (tuple): Size of the matplotlib figure.
        xlim (tuple, optional): x-axis (ppm) limits.
        title (str, optional): Plot title.

    Returns:
        None
    """
    num_spectra = len(spectra)
    if colors is None:
        colors = plt.cm.tab10.colors[:num_spectra]

    # Extract max intensities for spacing
    max_intensities = [max(pt[1] for pt in sp.confs) for sp in spectra]
    offsets = np.cumsum([0] + [m * scale_factor for m in max_intensities[:-1]])

    plt.figure(figsize=figsize)

    for i, sp in enumerate(spectra):
        # Extract and sort ppm and intensity
        ppm, intensity = zip(*sp.confs)
        ppm, intensity = np.array(ppm), np.array(intensity)
        sort_idx = np.argsort(ppm)
        ppm_sorted = ppm[sort_idx]
        intensity_sorted = intensity[sort_idx]

        plt.plot(ppm_sorted, intensity_sorted + offsets[i],
                 color=colors[i],
                 lw=1.5,
                 label=labels[i] if labels else None)

    plt.xlabel("Chemical shift (ppm)", fontsize=12)
    plt.yticks([])  # Hide y-axis ticks
    plt.gca().invert_xaxis()

    if xlim:
        plt.xlim(xlim)
    if labels:
        plt.legend(loc='upper right', fontsize=10)
    if title:
        plt.title(title, fontsize=14)

    plt.tight_layout()
    plt.show()

def plot_removed_noise(spectrum, noise, title="Noise Removed", color='gray', alpha=0.5, linewidth=0.75):
    """
    Plots the original spectrum and highlights the removed noise as a filled region.

    Args:
        spectrum (NMRSpectrum): Original spectrum object with `.confs` (ppm, intensity).
        noise (list or np.ndarray): Noise intensities removed (same length as spectrum).
        title (str): Plot title.
        color (str): Fill color for noise. Color should be in Matplotlib CSS Colors domain.
        alpha (float): Transparency for the noise fill.
        linewidth (float): Line width for the spectrum plot.

    Returns:
        None
    """

    x = [pt[0] for pt in spectrum.confs]
    y = [pt[1] for pt in spectrum.confs]

    fig, ax = plt.subplots(figsize=(10, 4))

    # Highlight removed noise
    ax.fill_between(
        x,
        [yi - ni for yi, ni in zip(y, noise)],
        y,
        color=color,
        alpha=alpha,
        label="Noise removed"
    )

    # Plot mixture
    ax.plot(
        x,
        y,
        linestyle="-",
        linewidth=linewidth,
        color="black",
        label="Original spectrum"
    )

    # NMR convention: x axis inverted
    ax.invert_xaxis()

    plt.xlabel("Chemical shift (ppm)")
    plt.ylabel("Intensity")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()