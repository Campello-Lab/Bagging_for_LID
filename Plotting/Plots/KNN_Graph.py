import skdim
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.manifold import MDS
from pathlib import Path
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
import os
from LIDBagging.Helper.Other import split_long_name
from tqdm import tqdm
import umap.umap_ as umap
from sklearn.manifold import SpectralEmbedding
from sklearn.metrics import pairwise_distances
import matplotlib
matplotlib.use("Agg")
###############################################################################################################################KNN GRAPH###############################################################################################################################
def landmark_mds(X, n_landmarks=200, n_components=2, random_state=42):
    np.random.seed(random_state)
    n_samples = X.shape[0]
    landmarks_idx = np.random.choice(n_samples, size=min(n_landmarks, n_samples), replace=False)
    X_landmarks = X[landmarks_idx]

    # Step 1: Run classical MDS on landmarks
    D_landmarks = pairwise_distances(X_landmarks)
    mds = MDS(n_components=n_components, dissimilarity='precomputed', random_state=random_state)
    Y_landmarks = mds.fit_transform(D_landmarks)

    # Step 2: Compute distances from all points to landmarks
    D_all_to_landmarks = pairwise_distances(X, X_landmarks)

    # Step 3: Triangulate to get approximate coordinates
    # Using least squares projection to landmark embedding
    G = -0.5 * (D_all_to_landmarks ** 2)
    G_centered = G - G.mean(axis=0)  # Center each column
    pseudo_inv = np.linalg.pinv(Y_landmarks)
    Y_all = G_centered @ pseudo_inv.T  # Project non-landmarks into embedding

    return Y_all

def plot_all_knn_graphs(dataset_dict, k=5, output_path='all_knn_graphs.pdf', position_method = 'mds'):
    num_datasets = len(dataset_dict)
    fig_height = num_datasets * 3
    fig, axes = plt.subplots(num_datasets, 1, figsize=(8, fig_height))
    if num_datasets == 1:
        axes = [axes]

    for ax, (name, data) in zip(axes, dataset_dict.items()):
        x = np.asarray(data[0])
        lid_values = np.full(len(x), data[1]) if not isinstance(data[1], np.ndarray) else np.asarray(data[1])
        labels = np.asarray(data[3]) if len(data) >= 4 else np.zeros(len(x), dtype=int)
        n = len(x)

        # Compute kNN
        dists, neighbors = skdim._commonfuncs.get_nn(x, k=k, n_jobs=1)

        # Embed in 2D with MDS
        pairwise_dist = np.linalg.norm(x[:, None] - x[None, :], axis=2)
        if position_method == 'mds':
            pos = MDS(n_components=2, dissimilarity="precomputed", random_state=42).fit_transform(pairwise_dist)
        elif position_method == 'umap':
            pos = umap.UMAP(n_neighbors=k, min_dist=0.1, metric='euclidean').fit_transform(x)
        elif position_method == 'se':
            pos = SpectralEmbedding(n_components=2, n_neighbors=k).fit_transform(x)
        elif position_method == 'lmds':
            pos = landmark_mds(x, n_landmarks=200, n_components=2, random_state=42)
        pos_dict = {i: pos[i] for i in range(n)}

        # Build the graph
        G = nx.Graph()
        for i in range(n):
            G.add_node(i, label=labels[i])
            for j in neighbors[i]:
                color = 'lightgray' if labels[i] == labels[j] else 'dimgray'
                G.add_edge(i, j, color=color)

        edge_colors = [G[u][v]['color'] for u, v in G.edges()]
        node_labels = [G.nodes[i]['label'] for i in G.nodes]
        node_colors = plt.cm.tab10(np.array(node_labels) % 10)

        # Draw the graph
        nx.draw_networkx_edges(G, pos_dict, edge_color=edge_colors, width=0.5, alpha=0.6, ax=ax)
        nx.draw_networkx_nodes(G, pos_dict, node_color=node_colors, node_size=10, alpha=0.9, ax=ax)

        # Custom legend with color dots and LID values
        from matplotlib.lines import Line2D
        unique_labels = np.unique(labels)
        legend_elements = []
        for label in unique_labels:
            mask = labels == label
            avg_lid = lid_values[mask].mean()
            if avg_lid.is_integer():
                lid_info = f"{int(avg_lid)}"
            else:
                lid_info = f"{avg_lid:.5f}".rstrip("0").rstrip(".")
            color = plt.cm.tab10(label % 10)
            legend_elements.append(Line2D(
                [0], [0],
                marker='o',
                color='w',
                markerfacecolor=color,
                markersize=6,
                label=f"LID = {lid_info}"
            ))

        ax.legend(handles=legend_elements, loc="upper right", fontsize=8, frameon=False)
        ax.set_title(f'{name}\nEmbedding Dim: {int(data[2])}', fontsize=10)
        ax.axis("off")
        print(f"{name} is done. Remaining: {num_datasets - (axes.tolist().index(ax) + 1)}")

    plt.tight_layout()
    save_path = Path(r'C:\Users\User\PycharmProjects\pythonProject3\LIDstuff\saved_results\plots') / output_path
    fig.savefig(save_path, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved all plots to {save_path}")

def reduce_result(result):
    reduced_result = result.copy()
    for key, value in result.items():
        for key2 in value[0].keys():
            reduced_result[key][0][key2] = result[key][0][key2][0]
    return reduced_result
def plot_graph_comparison_grid(results, dataset_dict, save_name='graph_grid_error',
                               save_dir=r'C:\Users\User\PycharmProjects\pythonProject3\LIDstuff\saved_results\plots',
                               k=5, show=False, clip_threshold=1, position_method = 'mds'):
    results = reduce_result(results)
    method_keys = list(results.keys())
    dataset_names = list(dataset_dict.keys())
    green_red_cmap = LinearSegmentedColormap.from_list("GreenRed", ["green", "red"])

    num_methods = len(method_keys)
    num_datasets = len(dataset_names)
    fig, axs = plt.subplots(num_methods, num_datasets, figsize=(10 * num_datasets, 7 * num_methods), dpi=300)
    if num_methods == 1:
        axs = np.expand_dims(axs, axis=0)
    if num_datasets == 1:
        axs = np.expand_dims(axs, axis=1)

    # Compute per-column maximum MSE
    max_mse_per_dataset = {}
    for dataset_name in dataset_names:
        max_mse = 0
        for method_key in method_keys:
            try:
                mse = results[method_key][1][dataset_name][1]
                max_mse = max(max_mse, mse)
            except Exception:
                continue
        max_mse_per_dataset[dataset_name] = max_mse

    for row_idx, method_key in tqdm(enumerate(method_keys)):
        est_dict, metrics_dict = results[method_key]
        for col_idx, dataset_name in enumerate(dataset_names):
            ax = axs[row_idx, col_idx]
            if dataset_name not in est_dict:
                ax.axis("off")
                continue
            x = np.asarray(dataset_dict[dataset_name][0])
            true_lid = np.asarray(dataset_dict[dataset_name][1])
            if len(true_lid[np.isnan(est_dict[dataset_name])]) > 0:
                print(true_lid[np.isnan(est_dict[dataset_name])])
            est_lid = np.asarray(est_dict[dataset_name])
            labels = np.zeros(len(x), dtype=int)
            n = len(x)

            # kNN
            from skdim._commonfuncs import get_nn
            _, neighbors = get_nn(x, k=k, n_jobs=1)

            # MDS layout
            pairwise_dist = np.linalg.norm(x[:, None] - x[None, :], axis=2)
            if position_method == 'mds':
                pos = MDS(n_components=2, dissimilarity="precomputed", random_state=42).fit_transform(pairwise_dist)
            elif position_method == 'umap':
                pos = umap.UMAP(n_neighbors=k, min_dist=0.1, metric='euclidean').fit_transform(x)
            elif position_method == 'se':
                pos = SpectralEmbedding(n_components=2, n_neighbors=k).fit_transform(x)
            elif position_method == 'lmds':
                pos = landmark_mds(x, n_landmarks=200, n_components=2, random_state=42)
            pos_dict = {i: pos[i] for i in range(n)}

            # Graph
            G = nx.Graph()
            for i in range(n):
                G.add_node(i)
                for j in neighbors[i]:
                    G.add_edge(i, j, color='lightgray')

            error = np.abs(est_lid - true_lid)
            norm_error = np.where(true_lid > 0, error / (clip_threshold*true_lid), error)
            norm_error_clipped = np.clip(norm_error, 0, 1)
            node_colors = green_red_cmap(norm_error_clipped)

            edge_colors = [G[u][v]['color'] for u, v in G.edges()]

            # Draw
            nx.draw_networkx_edges(G, pos_dict, edge_color=edge_colors, width=0.5, alpha=0.6, ax=ax)
            nx.draw_networkx_nodes(G, pos_dict, node_color=node_colors, node_size=10, alpha=0.9, ax=ax)

            # Annotations
            if row_idx == 0:
                ax.set_title(dataset_name, fontsize=22)
            if col_idx == 0:
                namedrawn = split_long_name(method_key, max_length=16)
                ax.annotate(namedrawn, xy=(0, 0.5), xycoords='axes fraction',
                            fontsize=18, rotation=90, ha='right', va='center')
            ax.axis("off")
            # Bar inset for MSE, Bias², Var
            try:
                mse = metrics_dict[dataset_name][1]
                bias2 = metrics_dict[dataset_name][2]
                var = metrics_dict[dataset_name][3]
                max_mse = max_mse_per_dataset[dataset_name]

                norm_height = mse / max_mse if max_mse > 0 else 0
                height = 0.7 * norm_height
                bar_ax = ax.inset_axes([1.02, 0.15, 0.05, height])
                bar_ax.axis("off")
                bar_ax.bar(0, bias2 / mse, width=1, color='blue')
                bar_ax.bar(0, var / mse, width=1, bottom=bias2 / mse, color='yellow')
            except Exception:
                continue

    # After all subplots are drawn
    for row in range(num_methods):
        for col in range(num_datasets):
            ax = axs[row, col]
            # Skip if this subplot was turned off
            if not ax.get_visible():
                continue

            # Get subplot bounding box in figure coordinates
            bbox = ax.get_position()

            # Draw a rectangle around the subplot
            rect = plt.Rectangle(
                (bbox.x0, bbox.y0),
                bbox.width,
                bbox.height,
                transform=fig.transFigure,
                color='black',
                linewidth=0.5,
                fill=False,
                zorder=1000  # Ensure it's on top
            )
            fig.patches.append(rect)

    # Colorbar
    cbar_ax = fig.add_axes([0.94, 0.15, 0.015, 0.7])
    sm = plt.cm.ScalarMappable(cmap=green_red_cmap, norm=plt.Normalize(vmin=0, vmax=1))
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label(f"Normalized LID Error (0 = green, ≥{clip_threshold} = red)", size=20)

    #plt.tight_layout(rect=[0.05, 0, 0.93, 1])
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{save_name}.pdf")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    if show:
        plt.show()
    return save_path
