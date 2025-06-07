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

    # Heatmap subplot
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
    if save and figures_path and experiment_name and lower_lim is not None and upper_lim is not None:
        filename = f"{experiment_name}_region_{lower_lim}_{upper_lim}"
        if variant is not None:
            filename = f"{experiment_name}_variant_{variant+1}_region_{lower_lim}_{upper_lim}"
        plt.savefig(f"{figures_path}/{filename}.png", dpi=300, bbox_inches='tight')

    plt.show()


