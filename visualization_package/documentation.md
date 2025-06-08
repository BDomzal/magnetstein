# Documentation

NAME

    visualization_package.utils

FUNCTIONS
```
retrieve_transport_plan(mix, spectra, interesting_region, kappa_mixture, kappa_components, solver=pulp.LpSolverDefault, log=True)
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
```
```
shift_and_mix(spectra, shifts, probs)
        Shifts and scales individual spectra, then mixes them into a single NMR spectrum.

        Args:
            spectra (list): List of spectrum-like objects, each with a `.confs` attribute containing (ppm, intensity) tuples.
            shifts (list): List of float shifts to apply to each spectrum (e.g., in ppm).
            probs (list): List of float scaling factors (e.g., estimated proportions) to apply to intensities of each spectrum.

        Returns:
            NMRSpectrum: A normalized and trimmed mixed NMR spectrum.
```
NAME

    visualization_package.visualization_module

FUNCTIONS

```
visualize_spectra(mixture, spectra_object, probs, components_names, window, shift=None, cumulate=True, figsize=(15, 9), title='Spectral Decomposition', save_path=None)
        Visualizes the spectral decomposition of a mixture into its component spectra.

        Supports both cumulative plotting using `fill_between` and standard stack plotting,
        with optional spectral shifting and moving average smoothing.

        Args:
            mixture: An object representing the mixture spectrum, with a `.confs` attribute,
                     where each element is a (x, y) tuple.
            spectra_object (list): List of component spectrum objects, each with `.confs`.
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
```
```
visualize_stacked_spectra(spectra, colors=None, labels=None, scale_factor=1.2, figsize=(10, 6), xlim=None, title=None)
        Plot stacked NMR spectra with vertical offsets based on each spectrum's max intensity.
        Spectra must have a `.confs` attribute with (ppm, intensity) pairs.

        Args:
            spectra (list): List of spectral objects with `.confs` attribute.
            colors (list of str, optional): Plot color per spectrum.
            labels (list of str, optional): Labels for the legend.
            scale_factor (float): Vertical spacing multiplier between spectra.
            figsize (tuple): Size of the matplotlib figure.
            xlim (tuple, optional): x-axis (ppm) limits.
            title (str, optional): Plot title.

        Returns:
            None
```

```
visualize_transport_distance_distribution(distances, component_kappa=None, mixture_kappa=None, component_label='Kappa components', mixture_label='Kappa mixture', component_color='hotpink', mixture_color='cornflowerblue', bins=100, figsize=(8, 6), title='Transport Distance Distribution', save_path=None)
        Plots a histogram of transport distances with optional markers for specific kappa values.

        Args:
            distances (list or np.ndarray): List of transport distances.
            component_kappa (float, optional): Value for a component kappa to highlight.
            mixture_kappa (float, optional): Value for a mixture kappa to highlight.
            component_label (str): Label for the component kappa line.
            mixture_label (str): Label for the mixture kappa line.
            component_color (str): Color for the component kappa markers.
            mixture_color (str): Color for the mixture kappa markers.
            bins (int): Number of histogram bins.
            figsize (tuple): Size of the figure.
            title (str): Title of the plot.
            save_path (str, optional): Path to save the plot. If None, plot is not saved.

        Returns:
            None
```
```
visualize_transport_plan(transport_df, mix_confs, wsom_confs, experiment_name=None, lower_lim=None, upper_lim=None, figures_path=None, variant=None, cmap='hot_r', point_scaling=20, show_colorbar=True, save=True, figsize=(14, 12), title='Transport Plan', *args, **kwargs)
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
```