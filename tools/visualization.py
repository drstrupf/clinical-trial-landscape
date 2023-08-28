"""
Helper functions to plot heatmaps, worldmaps, and regression plots.

Note that some design and default argument choices are specific for
our analyses.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns

sns.set(color_codes=True)
sns.set_style("whitegrid", {"grid.color": "gainsboro"})


def plot_heatmap_per_phase(
    trial_data,
    index_column_name,
    value_column_name,
    annotation_format="g",
    log_scaled=False,
    linear_diverging=False,
    scale_factor=1,
    phase_column_name="phase",
    cbar=False,
    linear_palette="rocket",
    diverging_palette="vlag_r",
    ax=None,
):
    """Plot a heatmap over phase columns.

    Args:
        trial_data: a dataframe with at least one column for phase, an index column and a value column
        index_column_name: the name of the index column, e.g. continent
        value_column_name: the column with the values to plot, e.g. number of trials
        annotation_format: the annotation format, default is 'g'.
        log_scaled: display log10 of values
        linear_diverging: use a diverging palette with midpoint 1, default is False
        scale_factor: multiply values with a custom factor, default is 1
        phase_column_name: column name where phase is specified
        cbar: show colorbar, default is False because annotation is on
        standard_cmap: the colormap to use
        ax: ax

    Returns:
        a heatmap
    """
    if ax is None:
        ax = plt.gca()

    # Prepare data - get a pivoted dataframe with the relevant columns only.
    pivoted_data = trial_data[
        [index_column_name, phase_column_name, value_column_name]
    ].pivot(
        index=index_column_name, columns=phase_column_name, values=value_column_name
    )
    # This is very specific: rename the phase entries so that they look nice.
    pivoted_data = pivoted_data.rename(
        columns={
            colname: colname.replace("PHASE", "Phase ")
            for colname in pivoted_data.columns
        }
    )
    # If we use a log10-scale, the scale factor and linear diverging args are ignored.
    if log_scaled:
        scale_factor = 1
        linear_diverging = False

    pivoted_data = pivoted_data * scale_factor

    # Prepare colorbar arguments.
    if log_scaled:
        vmin = np.log10(pivoted_data[pivoted_data.gt(0)].min(axis=None))
        vmax = np.log10(pivoted_data[pivoted_data.gt(0)].max(axis=None))
        center = 0
        cmap = diverging_palette
    elif linear_diverging:
        vmin = pivoted_data.min(axis=None)
        vmax = pivoted_data.max(axis=None)
        center = 1
        cmap = diverging_palette
    else:
        vmin = None
        vmax = None
        center = None
        cmap = linear_palette

    if log_scaled:
        pivoted_data = np.log10(pivoted_data)

    g = sns.heatmap(
        data=pivoted_data,
        annot=True,
        fmt=annotation_format,
        robust=False,
        cbar=cbar,
        vmin=vmin,
        vmax=vmax,
        center=center,
        cmap=cmap,
        ax=ax,
    )
    ax.set_xlabel(None)
    ax.set_ylabel(None)
    # Since the grid might be visible in case of NaNs when
    # using log scale, we disable it.
    ax.grid(False)

    return g


class MidpointNormalizeFair(mcolors.Normalize):
    """Create a custom diverging colormap.

    See https://stackoverflow.com/a/55667609 and
    https://matplotlib.org/users/colormapnorms.html for
    more explanation.
    """

    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        mcolors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        result, is_scalar = self.process_value(value)
        self.autoscale_None(result)
        vlargest = max(abs(self.vmax - self.midpoint), abs(self.vmin - self.midpoint))
        x, y = [self.midpoint - vlargest, self.midpoint, self.midpoint + vlargest], [
            0,
            0.5,
            1,
        ]
        return np.ma.masked_array(np.interp(value, x, y))


# World map
def plot_choropleth_map_country_level(
    trial_data,
    column_to_plot,
    geometry_base_dataframe,
    log_scale=False,
    drop_zeros=True,
    log_scale_diverging_palette=False,
    show_colorbar=True,
    colormap_minimum_value=None,
    colormap_maximum_value=None,
    edgecolor=u"black",
    edges_linewidth=0.25,
    geometry_column_name="geometry",
    country_id_column_name="country_ISO",
    base_color="silver",
    base_edgecolor=u"white",
    linear_palette="rocket",
    diverging_palette="vlag_r",
    ax=None,
):
    """Generate a choropleth world map.

    Diverging colormap uses the midpoint normalizer above,
    linear uses the Normalize class from mcolors.

    See the following references for some input on
    customizing/normalizing colormaps:

    - https://gis.stackexchange.com/a/330175
    - https://stackoverflow.com/a/18195921
    - https://stackoverflow.com/a/67751829
    - https://stackoverflow.com/a/55667609

    Args:
        trial_data: a pandas dataframe with at least a column for country names and a value column
        column_to_plot: the column with the values to show, e.g. number of trials
        geometry_base_dataframe: a dataframe with geometry data for countries
        log_scale: apply log10 to values
        drop_zeros: drop zero values
        log_scale_diverging_palette: use a diverging palette for log plot
        show_colorbar: display colorbar
        colormap_minimum_value: minimum value anchor for colormap
        colormap_maximum_value: maximum value anchor for colormap
        edgecolor: edges of overlay countries
        edges_linewidth: width of overlay country edges
        geometry_column_name: column with geometry data
        country_id_column_name: column with country ID in df and geometry data
        base_color: color of base world map
        base_edgecolor: edges of base map countries
        linear_palette: palette for linear colormap
        diverging_palette: palette for diverging colormap
        ax: ax

    Returns:
        a worldmap with choropleth overlay
    """
    # If no ax provided, create one
    if ax is None:
        ax = plt.gca()

    # Plot the base world map
    geometry_base_dataframe.plot(
        edgecolor=base_edgecolor, color=base_color, linewidth=edges_linewidth, ax=ax
    )

    # Prepare plot data
    plot_data = pd.merge(
        left=geometry_base_dataframe,
        right=trial_data[[country_id_column_name, column_to_plot]],
        on=country_id_column_name,
        how="inner",
    )[[country_id_column_name, geometry_column_name, column_to_plot]]

    # Remove zeros (i.e. countries with no data will be left blank)
    if drop_zeros:
        plot_data = plot_data[plot_data[column_to_plot] > 0]

    if log_scale:
        plot_data[column_to_plot] = np.log10(plot_data[column_to_plot])

    if log_scale and log_scale_diverging_palette:
        colormap_normalizer = MidpointNormalizeFair(
            midpoint=0, vmin=colormap_minimum_value, vmax=colormap_maximum_value
        )
        palette = diverging_palette
    else:
        colormap_normalizer = mcolors.Normalize(
            vmin=colormap_minimum_value, vmax=colormap_maximum_value
        )
        palette = linear_palette

    # Plot the geography stuff
    plot_data.plot(
        column=column_to_plot,
        edgecolor=edgecolor,
        linewidth=edges_linewidth,
        legend=False,
        cmap=palette,
        norm=colormap_normalizer,
        ax=ax,
    )

    # Create and display colorbar if required
    if show_colorbar:

        # Create a normalized colorbar
        cbar = plt.cm.ScalarMappable(norm=colormap_normalizer, cmap=palette)

        # Create an axis on the right side of ax
        divider = make_axes_locatable(ax)
        colorbar_ax = divider.append_axes("right", size="1%", pad=0.05)

        # Display the colorbar without outline
        colbar = plt.colorbar(cbar, cax=colorbar_ax)
        colbar.outline.set_visible(False)

    # Make it look nice...
    ax.axis("scaled")
    ax.set_xlim((-190, 190))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.grid(False)
    sns.despine(bottom=True, top=True, left=True, right=True, ax=ax)

    return ax


# Regression plot with statsmodels. Seaborn includes a function to plot
# a linear regression, but it doesn't return the parameters or measures.
# This function basically replicates the regression in regplot, and also
# returns the fit results from statsmodels OLS.
def linear_regression_plot_with_statsmodels(x, y, n_points=100, ax=None):
    """Create a linear regression plot and return the fitting results.

    See https://stackoverflow.com/a/59756979 for more info, and
    https://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.OLS.html
    for statsmodels documentation.

    Args:
        x: an array with the explanatory variable's values
        y: an array with the response values
        n_points: the number of points to generate for the plot
        ax: plot ax

    Returns:
        g: the regression plot
        fit_results: the results of the fit
    """
    # If no ax provided, create one
    if ax is None:
        ax = plt.gca()

    # OLS for best fit
    x_with_constant = sm.add_constant(x)
    fit_results = sm.OLS(y, x_with_constant).fit()

    # Now compute the predicted y for some points
    x_for_plot = sm.add_constant(np.linspace(np.min(x), np.max(x), n_points))
    y_predicted_for_plot = fit_results.get_prediction(x_for_plot)

    # Now create the plot - mean line
    g = ax.plot(
        x_for_plot[:, 1],
        y_predicted_for_plot.predicted_mean,
        color="black",
        linewidth=0.75,
    )

    # 95% CI
    ax.fill_between(
        x_for_plot[:, 1],
        y_predicted_for_plot.predicted_mean - 2 * y_predicted_for_plot.se_mean,
        y_predicted_for_plot.predicted_mean + 2 * y_predicted_for_plot.se_mean,
        alpha=0.125,
        color="black",
        linewidth=0,
    )

    return g, fit_results


# Combined regression and scatter plot
def linear_regression_and_scatter_plot(
    data,
    x_column,
    y_column,
    scatter_palette="husl",
    scatter_hue_column=None,
    scatter_hue_order=None,
    scatter_style=None,
    scatter_style_order=None,
    scatter_markers=True,
    scatter_alpha=1,
    xlim=None,
    ylim=None,
    n_points=100,
    ax=None,
):
    """Create a linear regression plot with scatterplot overlay.

    Args:
        data: a pandas dataframe with at least an explanatory and target value column
        x_column: the column with the explanatory variable
        y_column: the column with the targets
        scatter_palette: palette for the scatterplot, ignored if no hue specified
        scatter_hue_column: the column with the hue variable
        scatter_hue_order: a list with the hue order
        scatter_style: set marker style based on column
        scatter_style_order: order of marker styles
        scatter_markers: list of marker styles
        scatter_alpha: alpha for scatterplot markers
        xlim: x limits of the plot
        ylim: y limits of the plot
        n_points: the number of points to generate for the regression plot
        ax: plot ax

    Returns:
        ax: ax with regression and scatter plot
    """
    # If no ax provided, create one
    if ax is None:
        ax = plt.gca()

    x_data = data[x_column]
    y_data = data[y_column]

    # Regression
    g, regression_fit_results = linear_regression_plot_with_statsmodels(
        x=x_data, y=y_data, n_points=n_points, ax=ax
    )

    # Scatterplot overlay
    if scatter_hue_column is None:
        scatter_palette = None
    g_overlay = sns.scatterplot(
        data=data,
        x=x_column,
        y=y_column,
        hue=scatter_hue_column,
        hue_order=scatter_hue_order,
        style=scatter_style,
        style_order=scatter_style_order,
        markers=scatter_markers,
        palette=scatter_palette,
        alpha=scatter_alpha,
        ax=ax,
    )

    # Add the R2 annotation
    if xlim is None:
        xposition = max(x_data) - 0.085
    else:
        xposition = xlim[1] - 0.085
    if ylim is None:
        yposition = min(y_data) + 0.5
    else:
        yposition = ylim[0] + 0.5

    props = dict(boxstyle="round", facecolor="white", alpha=1)
    ax.text(
        xposition,
        yposition,
        r"$R^2$" + ": " + str(np.round(regression_fit_results.rsquared, 3)),
        fontsize=12,
        verticalalignment="top",
        bbox=props,
    )

    # Niceify
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    sns.despine(bottom=True, top=True, left=True, right=True, ax=ax)

    return ax
