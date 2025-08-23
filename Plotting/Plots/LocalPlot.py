import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import re
from collections import defaultdict
import os
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import Normalize, LinearSegmentedColormap
import numbers
##############################################################################################################################LOCAL PLOT###############################################################################################################################
def is_number(x):
    return isinstance(x, numbers.Number)

def plot_dimension_comparison_grid_local_with_mse_bars(results, data_sets, save_name='grid_plot',
                                                       save_dir=r'C:\Users\User\PycharmProjects\pythonProject3\LIDstuff\saved_results\plots',
                                                       show=False):
    dataset_names = list(data_sets.keys())
    method_keys = list(results.keys())
    green_red_cmap = mcolors.LinearSegmentedColormap.from_list("GreenRed", ["green", "red"])
    num_methods = len(method_keys)
    num_datasets = len(dataset_names)
    # Create a wider figure to fit the spatial plot + MSE bar
    fig, axs = plt.subplots(num_methods, num_datasets,
                            figsize=(10 * num_datasets, 7 * num_methods),
                            dpi=400)
    if num_methods == 1:
        axs = np.expand_dims(axs, axis=0)
    if num_datasets == 1:
        axs = np.expand_dims(axs, axis=1)
    # --- Collect bias²/variance from results using advanced parsing ---
    mse_data = defaultdict(dict)
    global_mse_max = 0

    for method_key in method_keys:
        try:
            raw = method_key[len("mle_"):] if method_key.startswith("mle_") else method_key
            method_part, rest = raw.split('_k_', 1)
            method_type = method_part if method_part else 'mle'
            k = int(rest.split('_')[0])
        except Exception as e:
            print(f"[ERROR] Could not parse method_key: {method_key}")
            continue

        sr_match = re.search(r'sampling_rate_([0-9.]+)', method_key)
        sampling_rate = sr_match.group(1) if sr_match else None

        if method_type == 'mle':
            label = 'MLE'
        else:
            readable = method_type.replace('_', '-').title()
            label = f"{readable} (rate={sampling_rate})" if sampling_rate else readable

        _, result_metrics = results[method_key]
        for dataset, vals in result_metrics.items():
            if len(vals) < 4:
                continue
            bias2, var = vals[2], vals[3]
            mse = bias2 + var
            mse_data[dataset][label] = {'bias2': bias2, 'var': var, 'mse': mse}
            global_mse_max = max(global_mse_max, mse)

    global_mse_max = 0
    for dataset_dict in mse_data.values():
        for method_dict in dataset_dict.values():
            mse = method_dict['bias2'] + method_dict['var']
            global_mse_max = max(global_mse_max, mse)

    # --- Main Plot Grid ---
    for row_idx, method_key in enumerate(method_keys):
        est_dict, _ = results[method_key]

        raw = method_key[len("mle_"):] if method_key.startswith("mle_") else method_key
        method_part, rest = raw.split('_k_', 1)
        method_type = method_part if method_part else 'mle'
        k = int(rest.split('_')[0])
        sr_match = re.search(r'sampling_rate_([0-9.]+)', method_key)
        sampling_rate = sr_match.group(1) if sr_match else None
        label = 'MLE' if method_type == 'mle' else f"{method_type.replace('_', '-').title()} (rate={sampling_rate})" if sampling_rate else method_type.title()

        for col_idx, dataset_name in enumerate(dataset_names):
            ax = axs[row_idx, col_idx]
            if dataset_name not in est_dict:
                ax.axis("off")
                continue

            X, true_dims, _ = data_sets[dataset_name]
            estimated_dims = est_dict[dataset_name][0]
            x_2d = X[:, :2]

            #mask = (x_2d[:, 0] > 1) & (x_2d[:, 0] < 1.5) & (x_2d[:, 1] > 1) & (x_2d[:, 1] < 1.5)
            #if np.sum(mask) == 0:
            #    ax.axis("off")
            #    continue

            x_2d_filtered = x_2d
            true_filtered = true_dims
            est_filtered = estimated_dims

            abs_error = np.abs(est_filtered - true_filtered)
            normalized_error = np.where(true_filtered > 0, abs_error / true_filtered, abs_error)
            normalized_error_clipped = np.clip(normalized_error, 0, 1)

            sc = ax.scatter(x_2d_filtered[:, 0], x_2d_filtered[:, 1],
                            s=5,
                            c=normalized_error_clipped,
                            cmap=green_red_cmap,
                            vmin=0,
                            vmax=1,
                            alpha=0.8,
                            rasterized=True)

            if row_idx == 0:
                ax.set_title(f"{dataset_name}", fontsize=10)
            if col_idx == 0:
                ax.set_ylabel(label, fontsize=9)

            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect('equal')

            # --- Dynamic MSE bar as scaled axis height ---
            if dataset_name in mse_data and label in mse_data[dataset_name]:
                bias2 = mse_data[dataset_name][label]['bias2']
                var = mse_data[dataset_name][label]['var']
                mse = bias2 + var

                # Normalize height relative to global max
                norm_height = mse / global_mse_max
                height = 0.7 * norm_height
                bar_ax = ax.inset_axes([1.02, 0.15, 0.05, height])  # scale bar height dynamically
                bar_ax.axis("off")

                bar_ax.bar(0, bias2 / mse, width=1, color='green')
                bar_ax.bar(0, var / mse, width=1, bottom=bias2 / mse, color='red')

    # Colorbar
    cbar_ax = fig.add_axes([0.94, 0.15, 0.015, 0.7])
    sm = plt.cm.ScalarMappable(cmap=green_red_cmap, norm=plt.Normalize(vmin=0, vmax=1))
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label("Normalized Error (0 = green, ≥1 = red)")

    plt.tight_layout(rect=[0, 0, 0.93, 1])
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{save_name}.pdf")
    plt.savefig(save_path, dpi=400, bbox_inches='tight')
    if show:
        plt.show()

def plot_lid_results(datasets, results, r=1.0, figsize=(8,8), show=False, save_dir = r'C:\Users\User\PycharmProjects\pythonProject3\LIDstuff\saved_results\plots', save_name = '2d_plot', bounds=None, subset_key=None):
    """
    Plot LID estimation results.

    Parameters
    ----------
    datasets : dict
        Dict[dataset_key] = (X, lid, d, idx)
        X: np.ndarray, shape (n_points, d)
        lid: array of ground truth LID values, length n_points
    results : dict
        Dict[method_key] = [
            {dataset_key: (lid_estimates, avg_estimate)},
            {dataset_key: [total_avg, total_mse, total_bias2, total_var, subset_measures, subsets]}
        ]
    r : float
        Scaling parameter: absolute difference = r * ground truth corresponds to full red.
    figsize : tuple
        Size for each subplot
    """
    method_keys = list(results.keys())
    dataset_keys = list(datasets.keys())
    n_methods = len(method_keys)
    n_datasets = len(dataset_keys)

    # Precompute max mse per dataset for consistent bar scaling
    max_mse = {}
    for dkey in dataset_keys:
        ms = []
        for m in method_keys:
            measures = results[m][1][dkey]
            subset_measures = measures[4]
            if subset_key and subset_key in subset_measures:
                mse = subset_measures[subset_key][1]
            else:
                mse = measures[1]
            ms.append(mse)
        max_mse[dkey] = max(ms)

    # Setup figure
    fig, axes = plt.subplots(n_methods, n_datasets,
                             figsize=(figsize[0] * n_datasets + 1,
                                      figsize[1] * n_methods),
                             squeeze=False)

    # Colormap from green (zero error) to red (max error)
    cmap = LinearSegmentedColormap.from_list('error_rg', ['green', 'red'])

    # Loop through subplots
    for i, mkey in enumerate(method_keys):
        for j, dkey in enumerate(dataset_keys):
            ax = axes[i, j]

            # Data unpack
            X_full, lid_full, _, _ = datasets[dkey]
            ests_full, _ = results[mkey][0][dkey]
            measures = results[mkey][1][dkey]
            total_orig, total_mse, total_bias2, total_var, subset_measures, subsets = measures

            # Apply bounds mask if requested
            if bounds is not None:
                x1, x2 = bounds[0]
                y1, y2 = bounds[1]
                mask = (X_full[:,0] >= x1) & (X_full[:,0] <= x2) & \
                       (X_full[:,1] >= y1) & (X_full[:,1] <= y2)
                X = X_full[mask]
                lid_truth = lid_full[mask]
                ests = ests_full[mask]
            else:
                X, lid_truth, ests = X_full, lid_full, ests_full

            if subset_key and subset_key in subset_measures:
                avg_m, mse_m, bias2_m, var_m = subset_measures[subset_key][:4]
                title_items = subset_measures[subset_key][4]
                mse_plot, bias_plot, var_plot = mse_m, bias2_m, var_m
            else:
                avg_m = total_orig
                mse_plot, bias_plot, var_plot = total_mse, total_bias2, total_var
                title_items = subset_measures

            # Compute color values
            diff = np.abs(lid_truth - ests)
            norm_diff = np.clip(diff / (r * lid_truth), 0, 1)
            colors = cmap(norm_diff)

            # Scatter plot
            ax.scatter(X[:,0], X[:,1], c=colors, s=10)
            ax.set_aspect('equal', 'box')
            ax.set_xticks([])
            ax.set_yticks([])

            bias_frac = bias_plot / max_mse[dkey]
            var_frac = var_plot / max_mse[dkey]
            # Create bar axis to the right
            divider = make_axes_locatable(ax)
            ax_bar = divider.append_axes('right', size='20%', pad=0.1)
            ax_bar.bar(0, bias_frac, color='blue', width=0.6)
            ax_bar.bar(0, var_frac, bottom=bias_frac, color='yellow', width=0.6)
            ax_bar.set_ylim(0, 1)
            ax_bar.set_xticks([])
            ax_bar.set_yticks([])

            # Titles
            lines = []
            lines.append(
                f"Total: avg={avg_m:.3f}, mse={mse_plot:.3f}, var={var_plot:.3f}, bias²={bias_plot:.3f}"
            )
            for lk, meas in title_items.items():
                if is_number(lk):
                    aa, mm, bb2, vv = meas[0], meas[1], meas[2], meas[3]
                    lines.append(f"LID {lk}: avg={aa:.3f}, mse={mm:.3f}, var={vv:.3f}, bias²={bb2:.3f}")
            ax.set_title("\n".join(lines), fontsize=12)

    fig.subplots_adjust(left=0.18, right=0.85, top=0.9, bottom=0.1)

    # row labels
    for i, mkey in enumerate(method_keys):
        pos = axes[i,0].get_position()
        fig.text(0.03, pos.y0+pos.height/2, mkey,
                 va='center', ha='left', rotation='vertical', fontsize=12)
    # col labels
    for j, dkey in enumerate(dataset_keys):
        pos = axes[0,j].get_position()
        fig.text(pos.x0+pos.width/2, 0.95, dkey,
                 va='bottom', ha='center', fontsize=12)

    # colorbar
    cax = fig.add_axes([0.88,0.1,0.02,0.8])
    sm=plt.cm.ScalarMappable(cmap=cmap,norm=Normalize(0,1)); sm.set_array([])
    fig.colorbar(sm, cax=cax, label=f'1-0: {r*100:.0f}%-0% error')

    # save or show
    if save_dir and save_name:
        os.makedirs(save_dir, exist_ok=True)
        fig.savefig(os.path.join(save_dir,f"{save_name}.pdf"),
                    dpi=400, bbox_inches='tight')
    if show:
        plt.show()

"""
I want you to write a python plotting function using the followingly structured input. We have "datasets" and "results".

The "datasets" input is a dictionary ("dataset_key"-s are the same as before) with:

Dict["dataset_key"] = (X, lid, d, idx)

- X is a numpy array of datapoints with d long rows (for the current plot d = 2)
- d = 2 (for this plot)
- lid is the known ground truth local intrinsic dimension of the datapoints in X
- idx are manifold indices (these are essentially the same for us as lid, in terms of categorization, so we can disregard these)

The "results" input is a dictionary with:

Dict["method_key"][0]["dataset_key"] = (lid_estimates, avg_lid_Estimate) 
Dict["method_key"][1]["dataset_key"] = [total_avg, total_mse, total_bias2, total_var, subset_measures, subsets]

- lid_estimates are estimates of lid at the points of datasets["dataset_key"][0], using the "method_key" named method,. Therefore lid_estimates estimate datasets["dataset_key"][1]. 
- total_avg = avg_lid_Estimate, and are both disregarded
- total_mse, total_bias2, total_var are the sum total of estimator mean squared error, bias, and variance across different manifolds of the dataset
- subset_measures is a dictionary with keys that are np.unique(datasets["dataset_key"][1]) so np.unique(lid) the differeing ground truth LIDs, differentiating between manifolds. Where

subset_measures[lid_key] = [manifold_avg, manifold_mse, manifold_bias2, manifold_var]
here manifold_avg, manifold_mse, manifold_bias2, manifold_var are the sum total of estimator mean squared error, bias, and variance across the manifold(s) in the dataset with ground truth LID = lid_key.

- subsets is a similar dictionary, with subset_measures[lid_key] = [Xm, lidm] containing a sub-array of X, with LID = lid_key, and the array of LID for the corresponding points (which should always be = lid_key for all of them)
--------------------------------------------------------------------------------------------------------------

Here follows the plotting function you need to write:

A large plot of subplots with "method_key" as rows, and "dataset_key" as colums. Each subplot contains the same thing, just for a different method/dataset, as specified by the keys.

Plot the points of X (so, datasets["dataset_key"][0]) on a scatterplot with X[0,:] as the x and X[1,:] as the y coordinates.

The points are colored on a color scale from full red to full green. The color is defined based on the absolute difference between the lid ground truth of the point (from, datasets["dataset_key"][1]) and the estimated lid of the point (from, results["method_key"][0]["dataset_key"][0]). full red corresponds to an absolute difference of 100%*r compared to the lid ground truth, while full green corresponds to an absolute difference of 0%. Where r is a parameter of the plotting function.

The colorbar of red to green is only plotted once on the very right side of the whole plot (not for each subplot)

A bar is plotted on the right side of each subplot. This bar's height is based on total_mse for that subplot, in relation to the other subplots in the COLUMN (the largest one is the largest bar, but they remain untransformed so that an x times as large mse corresponds to x times as tall bar). The bottom part of the bar is colored up to the value of total_bias with blue and the rest of the way with yellow. It should be the case that total_bias + total_var = total_mse, and this relationship should remain untransformed in terms of the bar, the proportion of total_var and total_bias in total_mse is expressed by the bar.

The title of each subplot shows subset_measures[lid_key][0] = manifold_avg for each lid_key, as "lid_key: manifold_avg".

The rows are vertically labeled once on the very left side of the whole plot by the respective "method_key" and each column is labeled once at the top of the column by the respective "dataset_key".

Optional possibilities:

There is an added possibility, to plot only points falling inside a square region [x1,x2] x [y1, y2] for every subplot, specified by input to the plotting function, but otherwise the plot structure remains the same.

There is an added possibility to use a specific key for subset_measures, based on a bound (we assume measures for this region are precomputed). If we do this, the exact key has to be given as input, but every calculation (for the total_mse decomposition bar, and subplot titles "lid_key: manifold_avg") will be done using subset_measures[key]. In sense
subset_measures[key] functions as the whole of [total_avg, total_mse, total_bias2, total_var, subset_measures, subsets] for calculating measures for the blue-yellow bar and the titles. 
But for ease of code, the scatterplot data is still coming from the original dataset and results.
"""