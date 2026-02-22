from copy import deepcopy
import re
from pathlib import Path
import json
import pathlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np
import pandas as pd
import seaborn as sns
import networkx as nx

import analysis
import coordinate
from plot_styler import BasePlotStyler, PlosStyler
import utils


class ScientificPlotter:
    """
    Helper class to generate complex scientific figures.
    """

    def __init__(self, journal_styler: BasePlotStyler, is_periodic=False):
        self.style = journal_styler
        self.fig_id = {
            "scatter": ["A", "B", "C", "D", "E"],
            "kde": ["F", "G", "H", "I", "J"],
            "bottom": [["K", "L"], ["M", "N"], ["O", "P"]],  # 3x2 list
            "fig2": [["A", "B"], ["C", "D"], ["E", "F"]],  # 3x2 list
        }
        self.is_periodic = is_periodic

    def k_load_or_process_data(
        self, graph_path, current_result_path, n_val, r_val, cache_read, cache_dump
    ):
        """
        Checks cache, runs aggregation for all bottom plots if needed, and dumps results.
        Returns the three aggregated data dictionaries.
        """

        # 1. Define Cache Path
        cache_dir = Path("./_plot_cache")
        cache_dir.mkdir(exist_ok=True)
        cache_key = f"data_r{r_val}_p{self.is_periodic}_n{n_val}.json"
        cache_file = cache_dir / cache_key

        # --- READ CACHE (Load all three dicts) ---
        if cache_read and cache_file.exists():
            print(f"✅ Loading cached data from {cache_file.name}")
            with open(cache_file, "r") as f:
                data = json.load(f)
                return (
                    data["nearest_neighbor"],
                    data["connection_dist_raw"],
                    data["wiring_cost_avg"],
                )

        # --- PROCESS DATA (If cache read is false or file doesn't exist) ---
        print("⏳ Processing data (No cache found or read=False)")

        # Define Subgroup Keys
        dkeys_nn = ["20", "50", "100", "200", "1000"]
        dkeys_cd = ["20", "50", "100", "200", "1000"]
        cost_fns = ["exponential", "lognormal"]

        nn_dict = {key: [] for key in dkeys_nn}  # Nearest Neighbor Data (Row 1)
        k_raw_dict_base = {sub_key: [] for sub_key in dkeys_cd}
        k_dict_combo = {key: deepcopy(k_raw_dict_base) for key in cost_fns}
        wiring_cost_dict_combo = deepcopy(k_dict_combo)

        neighbors_to_count = 50

        ext = f"*{str(n_val)}_{str(r_val)}*_2.csv"
        paths = utils.get_rfiles(current_result_path, ext)

        for p in paths:
            info = re.split("_ |_| ", p.stem)
            if "pl" in p.stem:
                try:
                    neighbors = int(info[5])
                except (IndexError, ValueError):
                    continue
            elif "uni" in p.stem:
                neighbors = 1000
            else:
                continue
            if str(neighbors) not in nn_dict:
                continue
            df_pos = pd.read_csv(p, sep=",", index_col=0)
            pos_arr = df_pos[["x", "y"]].to_numpy()

            nbrs = analysis.neuron_n_nearest_neighbors(
                pos_arr,
                neighbors_to_count,
                domain=(0, 0, 1, 1),
                periodic=self.is_periodic,
            )
            if len(nbrs) < 50:
                print(f"Length shorter than 50 for: {p}")
            nn_dict[str(neighbors)].append(nbrs)

        # --- 4. Run Aggregation for Connection Distances (GMLs) ---
        current_graph_path = (
            graph_path
            / f"periodic_{str(self.is_periodic)}"
            / "conformational"
            / str(r_val)
        )
        paths_gml = utils.get_rfiles(current_graph_path, "*.gml")

        for p in paths_gml:
            try:
                info = p.stem.split("_")
                cost_function = info[-1]
                subgroup_key = info[3]

                if (
                    cost_function in k_dict_combo
                    and subgroup_key in k_dict_combo[cost_function]
                ):
                    G = nx.read_gml(p)
                    # This function returns both the raw array and the scalar average
                    raw_dists, avg_cost = self._get_edge_stats(
                        G, domain=(0, 0, 1, 1), periodic=self.is_periodic
                    )

                    # Store raw distances
                    k_dict_combo[cost_function][subgroup_key].extend(raw_dists.tolist())

                    # Store average cost (the aggregation you wanted)
                    wiring_cost_dict_combo[cost_function][subgroup_key].append(avg_cost)

            except Exception as e:
                print(f"Error processing GML {p.name}: {e}")

        # --- DUMP CACHE ---
        if cache_dump:
            output_data = {
                "nearest_neighbor": nn_dict,
                "connection_dist_raw": k_dict_combo,
                "wiring_cost_avg": wiring_cost_dict_combo,
            }
            with open(cache_file, "w") as f:
                json.dump(output_data, f)
            print(f"💾 Data dumped to {cache_file.name}")

        return nn_dict, k_dict_combo, wiring_cost_dict_combo

    def _set_panel_label(self, ax, panel_id):
        """
        Sets the standard panel label (e.g., 'K', 'L') for a bottom subplot.
        This encapsulates font and position styling.
        """
        ax.set_title(
            panel_id,
            loc="left",
            fontsize=self.style.font_size + 2,
            fontweight="bold",
        )

    def _setup_layout(
        self,
        n_top_rows,
        n_top_cols,
        n_bottom_rows,
        n_bottom_cols,
        width_mm,
        height_mm,
        top_to_bottom_ratio,
    ):
        """
        Helper to create the complex GridSpec layout using nested grids.
        """
        figsize_inches = self.style.get_figsize_mm(width_mm, height_mm)

        fig = plt.figure(figsize=figsize_inches, constrained_layout=True)

        main_gs = fig.add_gridspec(nrows=2, ncols=1, height_ratios=top_to_bottom_ratio)

        # --- Top Grid ---
        gs_top = main_gs[0].subgridspec(n_top_rows, n_top_cols)
        axs_top = np.empty((n_top_rows, n_top_cols), dtype=object)
        for r in range(n_top_rows):
            for c in range(n_top_cols):
                axs_top[r, c] = fig.add_subplot(gs_top[r, c])

        # --- Bottom Grid ---
        gs_bottom = main_gs[1].subgridspec(n_bottom_rows, n_bottom_cols)

        # Create the 2x2 grid of axes
        axs_bottom = np.empty((n_bottom_rows, n_bottom_cols), dtype=object)
        for r in range(n_bottom_rows):
            for c in range(n_bottom_cols):
                axs_bottom[r, c] = fig.add_subplot(gs_bottom[r, c])

        # --- NEW: Create invisible axes for row titles ---
        axs_bottom_titles = []
        for r in range(n_bottom_rows):
            ax_title = fig.add_subplot(gs_bottom[r, :])
            ax_title.set_axis_off()
            axs_bottom_titles.append(ax_title)

        return fig, axs_top, axs_bottom, axs_bottom_titles

    def _get_edge_stats(self, G, domain=(0, 0, 1, 1), periodic=False):
        """
        Returns the raw edge distances and the average cost per edge.

        Returns
        -------
        tuple: (np.ndarray, float)
            - edge_distances: Array of all edge lengths.
            - avg_cost: The mean distance (0.0 if graph is empty or no edges).
        """

        # Check for empty graph or graph with no nodes
        if G.number_of_nodes() == 0:
            return np.array([]), 0.0

        # --- 1. Efficiently Extract Positions and Nodelist ---
        # G.nodes(data="pos") yields (node, pos_value) tuples in NetworkX's internal order.
        # The keys() and values() from this dict will maintain that order.
        try:
            pos_dict = dict(G.nodes(data="pos"))
        except Exception as e:
            print(f"Warning: Graph is missing the 'pos' attribute on nodes: {e}")
            return np.array([]), 0.0

        # Ensure all nodes have 'pos' data by checking if the lengths match.
        if len(pos_dict) != G.number_of_nodes():
            print("Warning: Some nodes are missing 'pos' data.")
            return np.array([]), 0.0

        # Use pos_dict.values() directly to create the array, avoiding N dictionary lookups.
        pos_array = np.array(list(pos_dict.values()))

        # The node order is now implicitly defined by list(pos_dict.keys())
        # and matches the order used by nx.to_numpy_array(G).

        # --- 2. Calculate full distance matrix ---
        dists = coordinate.periodic_dist(pos_array, domain=domain, periodic=periodic)

        # --- 3. Get Adjacency Matrix and Mask ---
        # By omitting the 'nodelist' argument, we rely on the default order,
        # which is list(G.nodes()) and must match the order of pos_array values.
        # We use bool dtype for minimal memory/faster mask operations.
        adj_bool = nx.to_numpy_array(G, dtype=bool)

        # If undirected, zero out the lower triangle (including diagonal)
        # to count each edge only once.
        if not G.is_directed():
            # np.triu(..., k=1) is the fastest way to get the upper triangle excluding the diagonal
            mask = np.triu(adj_bool, k=1)
        else:
            # For directed graphs, all entries are relevant (including diagonal for self-loops)
            mask = adj_bool

        # --- 4. Extract Distances and Calculate Average ---
        edge_distances = dists[mask]

        # Calculate the mean safely using np.mean() which handles empty arrays correctly (returns NaN).
        # We use a standard check for the final return.
        if edge_distances.size > 0:
            avg_cost = np.mean(edge_distances)
        else:
            avg_cost = 0.0

        return edge_distances, avg_cost

    def plot_spatial_setup(
        self, reverse_homes, n_val, r_val, current_result_path, width_type="max"
    ):
        """
        FIGURE 1: Plots the spatial arrangement (Scatter) and density (KDE).
        """
        n_cols = len(reverse_homes)
        n_rows = 2  # Scatter + KDE
        scatter_index = "1"

        if width_type == "single_col":
            width_mm = self.style.single_col_width_mm
        elif width_type == "max":
            width_mm = self.style.max_width_mm
        else:
            width_mm = width_type

        # Calculate height based on 1:1 aspect ratio constraint
        col_width = width_mm / n_cols
        # Height = (width of one plot * 2 rows) + extra for titles/cbar
        calculated_height = (col_width * 2) + 25

        figsize = self.style.get_figsize_mm(width_mm, calculated_height)
        fig = plt.figure(figsize=figsize, constrained_layout=True)

        # Simple 2xN grid
        gs = fig.add_gridspec(nrows=n_rows, ncols=n_cols, hspace=0.15)

        for i, h in enumerate(reverse_homes):
            ax_scatter = fig.add_subplot(gs[0, i])
            ax_kde = fig.add_subplot(gs[1, i])
            if "uni" in h:
                ext = f"{scatter_index}_{str(h)}_{str(n_val)}_{str(r_val)}_2.csv"
            else:
                ext = f"{scatter_index}*{str(n_val)}_{str(r_val)}_{str(h)}_2.csv"
            paths = utils.get_rfiles(current_result_path, ext)
            if not paths:
                print(f"Warning: No files found for {ext}. Skipping column {i}.")
                continue
            p = paths[0]
            info = re.split("_ |_| ", p.stem)
            if "pl" in p.stem:
                neighbors = int(info[5])
            elif "uni" in p.stem:
                neighbors = 0
            else:
                neighbors = 0
            df_pos = pd.read_csv(p, sep=",", index_col=0)
            pos_arr = df_pos[["x", "y"]].to_numpy()

            if "uni" in p.stem:
                ax_scatter.scatter(
                    pos_arr[:, 0],
                    pos_arr[:, 1],
                    s=10,
                    color="white",
                    alpha=1,
                    linewidth=0.5,
                    edgecolors="black",
                )

                title_text = rf"$N_{{\mathrm{{init}}}} = {1000}$"
            else:
                ax_scatter.scatter(
                    pos_arr[neighbors:, 0],
                    pos_arr[neighbors:, 1],
                    s=8,
                    color="grey",
                    alpha=1,
                    linewidth=0.5,
                    edgecolors="white",
                )
                ax_scatter.scatter(
                    pos_arr[:neighbors, 0],
                    pos_arr[:neighbors, 1],
                    s=8,
                    color="white",
                    alpha=1,
                    linewidth=0.5,
                    edgecolors="black",
                )
                title_text = f"$N_{{\mathrm{{init}}}} = {neighbors}$"

            ax_scatter.set_xlim((-0.0, 1.0))
            ax_scatter.set_ylim((-0.0, 1.0))
            ax_scatter.set_xticks((0.0, 1.0))
            ax_scatter.set_yticks((0.0, 1.0))
            ax_scatter.set_box_aspect(1)
            ax_scatter.grid(False)

            kde_plot = sns.kdeplot(
                df_pos,
                x="x",
                y="y",
                bw_adjust=0.2,
                fill=True,
                cbar=True,
                cbar_kws=dict(shrink=0.7, location="bottom"),
                levels=5,
                cmap="Greys",
                ax=ax_kde,
            )
            kde_plot.collections[0].colorbar.ax.tick_params(
                axis="x", rotation=90, labelsize=self.style.font_size - 2
            )
            ax_kde.set_xlim((-0.0, 1.0))
            ax_kde.set_ylim((-0.0, 1.0))
            ax_kde.set_xticks((0.0, 1.0))
            ax_kde.set_yticks((0.0, 1.0))
            ax_kde.set_box_aspect(1)
            ax_kde.grid(False)

            ax_scatter.set_title(
                title_text,
                loc="center",
                # fontsize=self.style.font_size,
                fontweight="normal",
            )

            ax_scatter.set_title(
                self.fig_id["scatter"][i],
                loc="left",
                fontsize=self.style.label_size,
                fontweight="bold",
            )

            ax_kde.set_title(
                self.fig_id["kde"][i],
                loc="left",
                fontsize=self.style.label_size,
                fontweight="bold",
            )

            ax_scatter.set_xlabel("")
            ax_kde.set_xlabel(r"$x$")

            if i == 0:
                ax_scatter.set_ylabel(r"$y$", rotation=0)
                ax_kde.set_ylabel(r"$y$", rotation=0)
            else:
                ax_scatter.set_yticklabels([])
                ax_scatter.set_ylabel("")
                ax_kde.set_yticklabels([])
                ax_kde.set_ylabel("")
            ax_scatter.set_xticklabels([])

        return fig

    def plot_connectivity_metrics(
        self,
        n_val,
        r_val,
        current_result_path,
        graph_path,
        width_type="max",
        height_mm=200,
        cache_read=True,
        cache_dump=False,
    ):
        """
        FIGURE 2: Plots statistics (NN, Wiring Cost, Connection Probabilities).
        """
        # --- 1. Load Data ---
        # Note: We don't need reverse_homes here, passing None or empty list if not used in cache key gen
        nn_dict, k_dict_combo, wiring_cost_dict_combo = self._load_or_process_data(
            graph_path=graph_path,
            current_result_path=current_result_path,
            n_val=n_val,
            r_val=r_val,
            cache_read=cache_read,
            cache_dump=cache_dump,
        )

        # --- 2. Setup Layout (3x2 Grid) ---
        if width_type == "single_col":
            width_mm = self.style.single_col_width_mm
        elif width_type == "max":
            width_mm = self.style.max_width_mm
        else:
            width_mm = width_type

        figsize = self.style.get_figsize_mm(width_mm, height_mm)
        fig = plt.figure(figsize=figsize, constrained_layout=True)

        # Create standard 3x2 grid
        gs = fig.add_gridspec(nrows=3, ncols=2)  # , hspace=0.15)

        # Create Axes Arrays
        axs = np.empty((3, 2), dtype=object)
        title_axs = []

        for r in range(3):
            # Create invisible title axis
            t_ax = fig.add_subplot(gs[r, :])
            t_ax.set_axis_off()
            title_axs.append(t_ax)
            for c in range(2):
                axs[r, c] = fig.add_subplot(gs[r, c])

        # --- 4. Plot Bottom Rows (Distance and CDF) ---
        ax_dist = axs[0, 0]  # Plot K
        ax_cum_dist = axs[0, 1]  # Plot L
        bins = np.linspace(0, 0.2, 21, endpoint=True)

        # Plotting for Row 1
        for key in nn_dict.keys():
            if not nn_dict[key]:
                continue
            neigh_dists = np.asarray(nn_dict[key]).flatten()
            ax_dist.hist(
                neigh_dists,
                bins=bins,
                density=True,
                align="mid",
                histtype="step",
                linewidth=2,
                label=str(key),
            )
            sns.ecdfplot(neigh_dists, ax=ax_cum_dist, label=str(key))  # linewidth=2,

        # 1. Prepare data for the boxplot: flatten the list of lists into a list of arrays
        # Each element in the list will be the averages for one subgroup (20, 50, etc.)

        # Configuration for the two rows to avoid copy-pasting code
        rows_config = [
            {"row_idx": 1, "cost_key": "exponential", "inset_loc": "upper right"},
            {"row_idx": 2, "cost_key": "lognormal", "inset_loc": "upper right"},
        ]

        # Common boxplot styles
        boxplot_props = {
            "width": 0.7,
            "showcaps": False,
            "medianprops": {"color": "black", "linewidth": 1.5},
            "flierprops": {"marker": "o", "markerfacecolor": "gray", "markersize": 2},
        }

        for config in rows_config:
            r = config["row_idx"]
            key = config["cost_key"]
            ax_hist, ax_cdf = axs[r, 0], axs[r, 1]

            # A. Main Plots (Hist + CDF)
            for subgroup_key, dists in k_dict_combo[key].items():
                all_dists = np.array(dists)
                if all_dists.size > 0:
                    ax_hist.hist(
                        all_dists,
                        bins=bins,
                        density=True,
                        histtype="step",
                        linewidth=2,
                        label=subgroup_key,
                    )
                    sns.ecdfplot(all_dists, ax=ax_cdf, label=subgroup_key)

            # B. Inset Plot (Wiring Cost Boxplot)
            ax_inset = inset_axes(
                ax_hist, width="45%", height="55%", loc=config["inset_loc"]
            )

            # Prepare data
            valid_subgroups = [
                k for k in wiring_cost_dict_combo[key] if wiring_cost_dict_combo[key][k]
            ]
            avg_costs_data = [
                np.array(wiring_cost_dict_combo[key][k]) for k in valid_subgroups
            ]

            # Plot Boxplot
            sns.boxplot(
                data=avg_costs_data,
                ax=ax_inset,
                palette=self.style.palette_colors[: len(valid_subgroups)],
                **boxplot_props,
            )

            # Format Inset
            ax_inset.set_ylabel("Cost", fontsize=self.style.font_size - 3)
            ax_inset.set_xticks(
                range(len(valid_subgroups)), valid_subgroups, rotation=0
            )
            ax_inset.tick_params(axis="both", labelsize=self.style.font_size - 3)
            ax_inset.set_facecolor("white")

        # Column 2 (CDFs)
        for ax in [axs[0, 1], axs[1, 1], axs[2, 1]]:
            ax.set_ylabel("CDF")
            ax.set_xlabel(r"Distance ($d$)")
            ax.set_xlim(right=1.0)  # Shared x-limit for CDFs

        for ax in [axs[0, 0], axs[1, 0], axs[2, 0]]:
            ax.set_ylabel(r"Probability Density, $P(d)$")
            ax.set_xlabel(r"Distance ($d$)")
            ax.set_xlim(right=0.20)  # Shared x-limit for CDFs

        # Legend (Only on first plot of each column to avoid clutter)
        axs[0, 0].legend(prop={"size": 8})
        axs[0, 1].legend(prop={"size": 8})

        # Panel Labels (A, B, C...)
        for r in range(3):
            for c in range(2):
                axs[r, c].set_title(
                    self.fig_id["fig2"][r][c],
                    loc="left",
                    fontsize=self.style.label_size,
                    fontweight="bold",
                )

        # Spanning Row Titles
        titles = [
            "Nearest Neighbor Distance",
            "Connection Probabilities: Exponential",
            "Connection Probabilities: Log-normal",
        ]

        for r, title in enumerate(titles):
            title_axs[r].set_title(
                title,
                fontsize=self.style.font_size + 2,
                loc="center",
                y=1.1,  # Lift title slightly above panel labels
            )

        return fig

    def plot_position_and_distance(
        self,
        reverse_homes,
        n_val,
        r_val,
        current_result_path,
        graph_path,
        width_type="max",
        bottom_section_height_mm=140,
        cbar_height_mm=15,
        main_title=None,
        cache_read=True,
        cache_dump=False,
    ):
        """
        Creates the specific multi-panel figure.
        """

        # When testing, set run_index to avoid runnning all
        scatter_index = "1"

        # --- 1. Calculate Dimensions ---
        n_top_rows = 2
        n_top_cols = len(reverse_homes)
        n_bottom_rows = 3
        n_bottom_cols = 2

        if width_type == "single_col":
            width_mm = self.style.single_col_width_mm
        elif width_type == "max":
            width_mm = self.style.max_width_mm
        elif isinstance(width_type, (int, float)):
            width_mm = width_type
        else:
            raise ValueError("width_type must be 'single_col', 'max', or a number")

        scatter_plot_width_mm = width_mm / n_top_cols
        scatter_plot_height_mm = scatter_plot_width_mm
        top_section_height_mm = (scatter_plot_height_mm * n_top_rows) + cbar_height_mm
        height_mm = top_section_height_mm + bottom_section_height_mm
        top_bottom_ratio = [top_section_height_mm, bottom_section_height_mm]

        print(f"Creating figure: {width_mm:.1f}mm (w) x {height_mm:.1f}mm (h)")

        # --- 2. Setup Layout ---
        # This call now returns 4 items
        fig, axs, bottom_axs, bottom_title_axs = self._setup_layout(
            n_top_rows,
            n_top_cols,
            n_bottom_rows,
            n_bottom_cols,
            width_mm=width_mm,
            height_mm=height_mm,
            top_to_bottom_ratio=top_bottom_ratio,
        )

        # --- 3. Plot Top Rows (Position and KDE) ---
        for i, h in enumerate(reverse_homes):
            print(f"Plotting top grid (col {i}) for type: {h}")
            ax_scatter = axs[0, i]
            ax_kde = axs[1, i]
            if "uni" in h:
                ext = f"{scatter_index}_{str(h)}_{str(n_val)}_{str(r_val)}_2.csv"
            else:
                ext = f"{scatter_index}*{str(n_val)}_{str(r_val)}_{str(h)}_2.csv"
            paths = utils.get_rfiles(current_result_path, ext)
            if not paths:
                print(f"Warning: No files found for {ext}. Skipping column {i}.")
                continue
            p = paths[0]
            info = re.split("_ |_| ", p.stem)
            if "pl" in p.stem:
                neighbors = int(info[5])
            elif "uni" in p.stem:
                neighbors = 0
            else:
                neighbors = 0
            df_pos = pd.read_csv(p, sep=",", index_col=0)
            pos_arr = df_pos[["x", "y"]].to_numpy()

            if "uni" in p.stem:
                ax_scatter.scatter(
                    pos_arr[:, 0],
                    pos_arr[:, 1],
                    s=10,
                    color="white",
                    alpha=1,
                    linewidth=0.5,
                    edgecolors="black",
                )

                title_text = rf"$N_{{\mathrm{{init}}}} = {1000}$"
            else:
                ax_scatter.scatter(
                    pos_arr[neighbors:, 0],
                    pos_arr[neighbors:, 1],
                    s=8,
                    color="grey",
                    alpha=1,
                    linewidth=0.5,
                    edgecolors="white",
                )
                ax_scatter.scatter(
                    pos_arr[:neighbors, 0],
                    pos_arr[:neighbors, 1],
                    s=8,
                    color="white",
                    alpha=1,
                    linewidth=0.5,
                    edgecolors="black",
                )
                title_text = f"$N_{{\mathrm{{init}}}} = {neighbors}$"

            ax_scatter.set_xlim((-0.01, 1.01))
            ax_scatter.set_ylim((-0.01, 1.01))
            ax_scatter.set_xticks((0.0, 1.0))
            ax_scatter.set_yticks((0.0, 1.0))
            ax_scatter.set_box_aspect(1)
            ax_scatter.grid(False)

            kde_plot = sns.kdeplot(
                df_pos,
                x="x",
                y="y",
                bw_adjust=0.2,
                fill=True,
                cbar=True,
                cbar_kws=dict(shrink=0.7, location="bottom"),
                levels=5,
                cmap="Greys",
                ax=ax_kde,
            )
            kde_plot.collections[0].colorbar.ax.tick_params(
                axis="x", rotation=90, labelsize=self.style.font_size - 2
            )
            ax_kde.set_xlim((-0.01, 1.01))
            ax_kde.set_ylim((-0.01, 1.01))
            ax_kde.set_xticks((0.0, 1.0))
            ax_kde.set_yticks((0.0, 1.0))
            ax_kde.set_box_aspect(1)
            ax_kde.grid(False)

            axs[0, i].set_title(
                title_text,
                loc="center",
                # fontsize=self.style.font_size,
                fontweight="normal",
            )

            axs[0, i].set_title(
                self.fig_id["scatter"][i],
                loc="left",
                # fontsize=self.style.font_size,
                fontweight="bold",
            )

            axs[1, i].set_title(
                self.fig_id["kde"][i],
                loc="left",
                # fontsize=self.style.font_size,
                fontweight="bold",
            )

            ax_scatter.set_xlabel("")
            ax_kde.set_xlabel(r"$x$")

            if i == 0:
                ax_scatter.set_ylabel(r"$y$", rotation=0)
                ax_kde.set_ylabel(r"$y$", rotation=0)
            else:
                ax_scatter.set_yticklabels([])
                ax_scatter.set_ylabel("")
                ax_kde.set_yticklabels([])
                ax_kde.set_ylabel("")
            ax_scatter.set_xticklabels([])

        # --- 4. Plot Bottom Rows (Distance and CDF) ---
        ax_dist = bottom_axs[0, 0]  # Plot K
        ax_cum_dist = bottom_axs[0, 1]  # Plot L
        bins = np.linspace(0, 0.2, 21, endpoint=True)

        # Unpack all three processed data structures
        nn_dict, k_dict_combo, wiring_cost_dict_combo = self._load_or_process_data(
            graph_path=graph_path,
            current_result_path=current_result_path,
            n_val=n_val,
            r_val=r_val,
            cache_read=cache_read,
            cache_dump=cache_dump,
        )

        # Plotting for Row 1
        for key in nn_dict.keys():
            if not nn_dict[key]:
                continue
            neigh_dists = np.asarray(nn_dict[key]).flatten()
            ax_dist.hist(
                neigh_dists,
                bins=bins,
                density=True,
                align="mid",
                histtype="step",
                linewidth=2,
                label=str(key),
            )
            sns.ecdfplot(neigh_dists, ax=ax_cum_dist, label=str(key))  # linewidth=2,

        # ... (Add your plotting code for ax_m, ax_n, ax_o, ax_p here) ...
        # --- 5. Load Data for Rows 2 & 3 (Connection Probability) ---
        ax_m = bottom_axs[1, 0]
        ax_n = bottom_axs[1, 1]
        ax_o = bottom_axs[2, 0]
        ax_p = bottom_axs[2, 1]

        # --- 6. Plot Bottom Row 2 (Exponential) ---
        cost_key = "exponential"
        for subgroup_key in k_dict_combo[cost_key].keys():
            all_dists_exp = np.array(k_dict_combo[cost_key][subgroup_key])
            if all_dists_exp.size > 0:
                ax_m.hist(
                    all_dists_exp,
                    bins=bins,
                    density=True,
                    histtype="step",
                    linewidth=2,
                    label=subgroup_key,
                )
                sns.ecdfplot(all_dists_exp, ax=ax_n, label=subgroup_key)  # linewidth=2,

        # --- 7. Plot Bottom Row 3 (Log-normal) ---
        cost_key = "lognormal"
        for subgroup_key in k_dict_combo[cost_key].keys():
            all_dists_log = np.array(k_dict_combo[cost_key][subgroup_key])
            if all_dists_log.size > 0:
                ax_o.hist(
                    all_dists_log,
                    bins=bins,
                    density=True,
                    histtype="step",
                    linewidth=2,
                    label=subgroup_key,
                )
                sns.ecdfplot(
                    all_dists_log, ax=ax_p, label=subgroup_key
                )  # , linewidth=1.0)

        # --- ADD INSET FOR LOG-NORMAL COST ---
        # --- ADD INSET FOR EXPONENTIAL COST (Boxplot) ---

        ax_inset_exp = inset_axes(ax_m, width="40%", height="50%", loc="upper right")
        cost_key = "exponential"

        # 1. Prepare data for the boxplot: flatten the list of lists into a list of arrays
        # Each element in the list will be the averages for one subgroup (20, 50, etc.)
        exp_avg_costs_data = [
            np.array(wiring_cost_dict_combo[cost_key][subgroup_key])
            for subgroup_key in wiring_cost_dict_combo[cost_key].keys()
            if wiring_cost_dict_combo[cost_key][subgroup_key]
        ]

        # 2. Prepare labels (subgroup_key)
        exp_labels = [
            subgroup_key
            for subgroup_key in wiring_cost_dict_combo[cost_key].keys()
            if wiring_cost_dict_combo[cost_key][subgroup_key]
        ]

        # 3. Use Seaborn boxplot
        sns.boxplot(
            data=exp_avg_costs_data,
            ax=ax_inset_exp,
            palette=self.style.palette_colors[: len(exp_labels)],  # Use styler colors
            width=0.7,
            showcaps=False,
            medianprops={"color": "black", "linewidth": 1.5},
            flierprops={"marker": "o", "markerfacecolor": "gray", "markersize": 2},
        )

        ax_inset_exp.set_ylabel(r"Cost", fontsize=self.style.font_size - 3)
        ax_inset_exp.set_xticks(range(len(exp_labels)), exp_labels, rotation=0)
        ax_inset_exp.tick_params(axis="y", labelsize=self.style.font_size - 3)
        ax_inset_exp.tick_params(axis="x", labelsize=self.style.font_size - 3)
        # ax_inset_exp.set_xlabel(r"$N_{nodes}$", fontsize=self.style.font_size - 4)

        # --- 7. Plot Bottom Row 3 (Log-normal) ---
        ax_o = bottom_axs[2, 0]  # Target histogram axis
        ax_p = bottom_axs[2, 1]  # Target CDF axis
        cost_key = "lognormal"

        # ... (plotting hist and cdf lines is unchanged) ...

        # --- ADD INSET FOR LOG-NORMAL COST (Boxplot) ---
        ax_inset_log = inset_axes(ax_o, width="40%", height="50%", loc="upper right")

        # Prepare data (repeat aggregation logic)
        logn_avg_costs_data = [
            np.array(wiring_cost_dict_combo[cost_key][subgroup_key])
            for subgroup_key in wiring_cost_dict_combo[cost_key].keys()
            if wiring_cost_dict_combo[cost_key][subgroup_key]
        ]
        logn_labels = [
            subgroup_key
            for subgroup_key in wiring_cost_dict_combo[cost_key].keys()
            if wiring_cost_dict_combo[cost_key][subgroup_key]
        ]

        # Use Seaborn boxplot
        sns.boxplot(
            data=logn_avg_costs_data,
            ax=ax_inset_log,
            palette=self.style.palette_colors[: len(logn_labels)],
            width=0.7,
            showcaps=False,
            medianprops={"color": "black", "linewidth": 1.5},
            flierprops={"marker": "o", "markerfacecolor": "gray", "markersize": 2},
        )

        ax_inset_log.set_ylabel(r"Cost", fontsize=self.style.font_size - 3)
        ax_inset_log.set_xticks(range(len(logn_labels)), logn_labels, rotation=0)
        ax_inset_log.tick_params(axis="y", labelsize=self.style.font_size - 3)
        ax_inset_log.tick_params(axis="x", labelsize=self.style.font_size - 3)
        # ax_inset_log.set_xlabel(r"$N_{nodes}$", fontsize=self.style.font_size - 4)
        ax_inset_log.set_facecolor("white")  # Ensure clear background

        # --- Main Figure Title ---
        if main_title:
            fig.suptitle(main_title, fontsize=12, fontweight="bold", y=1.02)

        # Row 1 (K, L)
        ax_dist.legend(prop={"size": 8})
        ax_dist.set_ylabel(r"Probability Density, $P(d)$")

        # Row 2 (M, N)
        ax_m.set_ylabel(r"Probability Density, $P(d)$")

        # Row 3 (O, P)
        ax_o.set_ylabel(r"Probability Density, $P(d)$")

        for ax in [ax_dist, ax_m, ax_o]:
            ax.set_xlabel(r"Distance ($d$)")

        shared_xlim = (-0.01, 1.0)
        for ax in [ax_cum_dist, ax_n, ax_p]:
            ax.set_xlabel(r"Distance ($d$)")
            ax.set_xlim(right=1.0)
            ax.set_ylabel(r"CDF")

        ax_cum_dist.legend(prop={"size": 8})

        # --- Panel Labels (K, L, M, N, O, P) ---
        for r in range(n_bottom_rows):
            for c in range(n_bottom_cols):
                bottom_axs[r, c].set_title(
                    self.fig_id["bottom"][r][c],
                    loc="left",
                    fontsize=self.style.font_size,
                    fontweight="bold",
                )

        # --- Spanning Row Titles (on invisible axes) ---
        titles = [
            "Nearest Neighbor Distance",
            r"Connection Probabilities: Exponential",
            r"Connection Probabilities: Log-normal",
        ]
        for r, title in enumerate(titles):
            bottom_title_axs[r].set_title(
                title,
                fontsize=self.style.font_size + 2,
                # fontweight="bold",
                loc="center",
                y=1.2,
            )

        return fig

    def plot_aggregated_dataframe(
        self,
        df,
        x_col,
        x_label,
        y_cols,
        y_labels,
        hue_col,
        width_type="max",
        height_per_row_mm=60,
    ):
        """
        Plots aggregated results from a DataFrame in a 2-column layout.
        If an odd number of plots is requested, the last one is centered.

        Args:
            df (pd.DataFrame): The source data.
            x_col (str): Column name for the X-axis.
            y_cols (list[str]): List of column names to plot on Y-axes.
            y_labels (list[str]): Corresponding labels for the Y-axes.
            width_type (str): 'max', 'single_col', or mm value.
            height_per_row_mm (float): Height of each row in mm.
        """

        n_plots = len(y_cols)
        n_rows = (n_plots + 1) // 2

        if width_type == "single_col":
            width_mm = self.style.single_col_width_mm
        elif width_type == "max":
            width_mm = self.style.max_width_mm
        elif isinstance(width_type, (int, float)):
            width_mm = width_type
        else:
            raise ValueError("Invalid width_type")

        total_height_mm = n_rows * height_per_row_mm
        figsize = self.style.get_figsize_mm(width_mm, total_height_mm)

        print(
            f"Creating Aggregated Boxplot Figure: {width_mm:.1f}mm x {total_height_mm:.1f}mm"
        )

        fig = plt.figure(figsize=figsize, constrained_layout=True)

        # --- 3. GridSpec Magic ---
        gs = fig.add_gridspec(nrows=n_rows, ncols=4)
        axs = []

        if hue_col:
            # Get unique values to ensure consistent mapping
            hue_order = sorted(df[hue_col].unique())
            # Slice the palette to match the number of hue categories (usually 2)
            comparison_palette = self.style.palette_colors[:len(hue_order)]
        else:
            comparison_palette = None

        for i, (col_name, label) in enumerate(zip(y_cols, y_labels)):
            row = i // 2
            col_pos = i % 2
            is_last = i == n_plots - 1
            is_odd_total = n_plots % 2 != 0

            if is_last and is_odd_total:
                ax = fig.add_subplot(gs[row, 1:3])  # Center
            else:
                if col_pos == 0:
                    ax = fig.add_subplot(gs[row, 0:2])
                else:
                    ax = fig.add_subplot(gs[row, 2:4])

            axs.append(ax)

            # sns.boxplot(
            #     data=df,
            #     x=x_col,
            #     y=col_name,
            #     ax=ax,
            #     hue=hue_col,
            #     palette=comparison_palette,
            #     width=0.7,
            #     medianprops={"color": "black", "linewidth": 1.5},
            #     flierprops={
            #         "marker": "o",
            #         "markerfacecolor": "gray",
            #         "markersize": 3,
            #         "markeredgecolor": "none",
            #     }
            # )

            # sns.stripplot(
            #     data=df,
            #     x=x_col,
            #     y=col_name,
            #     hue=hue_col,
            #     # order=x_order,            # <--- CRITICAL: Must match pointplot
            #     palette=comparison_palette,
            #     dodge=True,               # Separates the hue groups side-by-side
            #     alpha=0.15,               # Transparency so they don't dominate
            #     size=2,                   # Small dots
            #     jitter=0.2,              # Adds random noise to spread them out
            #     ax=ax,
            #     legend=False,             # No legend for the background dots
            #     # zorder=0                  # Forces it to the background
            # )

            sns.pointplot(
                data=df,
                x=x_col,
                y=col_name,
                ax=ax,
                hue=hue_col,
                palette=comparison_palette,
                # order=x_order,   # Ensure your custom sort order is kept
                dodge=0.4,       # Important: separates the two lines so they don't overlap
                join=False,      # Set to True if you want lines connecting the dots
                capsize=0.1,     # Adds the "T" bars to error lines
                scale=0.8,       # Adjusts the size of the points
                errorbar='sd',   # Shows Standard Deviation (use 'ci' for Conf. Interval)
            )

            ax.set_ylabel(label)

            ax.set_xlabel(x_label)

            # Panel Label (A, B, C...)
            panel_label = chr(65 + i)
            self._set_panel_label(ax, panel_label)

            if hue_col:
                handles, labels = ax.get_legend_handles_labels()
                if ax.get_legend():
                    ax.get_legend().remove()

                if i == 0:
                    ax.legend(
                        handles=handles, 
                        labels=labels, 
                        loc='best',        # Or 'upper right', 'lower left
                        fontsize=8, 
                        title_fontsize=8
                    )

        return fig
    
    def plot_aggregated_dataframe_variable_column(
        self,
        df,
        x_col,
        x_label,
        y_cols,
        y_labels,
        hue_col=None,
        width_type="max",
        height_per_row_mm=50,  # Slightly shorter rows since we have 4 columns
        n_cols=4,              # Added parameter to control columns (default 4 for matrix)
    ):
        """
        Plots aggregated results (Line with Markers) in a grid layout.
        Optimized for sparse data (e.g., 5 pruning steps).
        """
        n_plots = len(y_cols)
        # Calculate rows needed based on columns
        n_rows = (n_plots + n_cols - 1) // n_cols 

        if width_type == "single_col":
            width_mm = self.style.single_col_width_mm
        elif width_type == "max":
            width_mm = self.style.max_width_mm
        elif isinstance(width_type, (int, float)):
            width_mm = width_type
        else:
            raise ValueError("Invalid width_type")

        total_height_mm = n_rows * height_per_row_mm
        figsize = self.style.get_figsize_mm(width_mm, total_height_mm)

        print(f"Creating {n_cols}x{n_rows} Matrix Figure: {width_mm:.1f}mm x {total_height_mm:.1f}mm")

        fig = plt.figure(figsize=figsize, constrained_layout=True)
        gs = fig.add_gridspec(nrows=n_rows, ncols=n_cols)

        if hue_col:
            hue_order = sorted(df[hue_col].unique())
            comparison_palette = self.style.palette_colors[:len(hue_order)]
        else:
            comparison_palette = None

        for i, (col_name, label) in enumerate(zip(y_cols, y_labels)):
            # Calculate grid position
            row = i // n_cols
            col_pos = i % n_cols
            
            # --- CENTERING LOGIC ---
            # If this is the last plot and it's alone on the row, try to center it?
            # For a 4x4 matrix with 16 plots, no centering is needed. 
            # But if you have 15 plots, this logic handles simple placement.
            ax = fig.add_subplot(gs[row, col_pos])

            # --- THE PLOT ---
            # Lineplot with markers is best for 5 steps
            sns.lineplot(
                data=df,
                x=x_col,
                y=col_name,
                hue=hue_col,
                palette=comparison_palette,
                ax=ax,
                errorbar='sd',     # Standard Deviation band
                estimator='mean',
                marker='o',        # <--- CRITICAL: Shows the 5 actual steps
                markersize=6,      # Make dots visible
                linewidth=1.5,
                alpha=0.8
            )

            # Style adjustments
            ax.set_ylabel(label)
            ax.set_xlabel(x_label)
            # ax.grid(True, linestyle='--', alpha=0.3) # Helpful grid for reading values
            # ax.set_xscale('log')  # Log scale if x_col is pruning steps
            # Panel Label (A, B, C...)
            panel_label = chr(65 + i)
            self._set_panel_label(ax, panel_label)

            # Remove Legend from all but the first plot to prevent clutter
            if ax.get_legend():
                ax.get_legend().remove()
            
            if i == 0 and hue_col:
                handles, labels = ax.get_legend_handles_labels()
                ax.legend(
                    handles=handles, 
                    labels=labels, 
                    loc='best', 
                    fontsize=8,
                    frameon=True
                )

        return fig

    def savefigs(self, fig, fig_dir, fname_base, image_types):
        """Helper to save the figure in multiple formats."""
        if not isinstance(image_types, (list, tuple)):
            image_types = [image_types]

        for img_type in image_types:
            save_path = Path(fig_dir) / f"{fname_base}.{img_type}"
            print(f"Saving figure to: {save_path}")
            fig.savefig(save_path, bbox_inches="tight")


# ===================================================================
#  MAIN EXECUTION BLOCK
# ===================================================================
if __name__ == "__main__":

    # --- 1. Setup Style ---
    try:
        # You can now swap this to NatureStyler and the width will auto-adjust
        # journal_style = NatureStyler(font_size=7, palette='tol-bright')
        journal_style = PlosStyler(font_size=8, palette="wong")

        journal_style.apply_style(font_family="Arial")

    except Exception as e:
        print(f"Error applying style: {e}")
        print("Falling back to default Matplotlib style.")
        journal_style = BasePlotStyler()

    # --- 2. Instantiate Plotter ---
    plotter = ScientificPlotter(journal_styler=journal_style)

    # --- 3. Define Parameters ---
    nodes = 1000
    r_min = 1e-4
    homes = ["20", "50", "100", "200", "uni"]
    reverse_homes = homes[::-1]
    top_path = pathlib.Path.cwd()
    version_string = "2025-07-23"

    result_path = top_path / f"results_revision_{version_string}"
    graph_path = top_path / f"graphs_revision_{version_string}"

    positions_path = result_path / "positions"
    is_periodic = False
    current_result_path = (
        positions_path / f"results_periodic_{str(is_periodic)}_{version_string}"
    )

    image_types = [
        "png",
        "svg",
        "pdf",
        "tif",
    ]

    # --- 4. Generate and Save Plot ---
    print("Generating plot...")
    try:
        # --- CHANGED: Use new signature ---
        # We select 'max' width because we have 5 plots across.
        # We keep the 280mm height from your original code.
        # fig = plotter.plot_position_and_distance(
        #     reverse_homes=reverse_homes,
        #     n_val=nodes,
        #     r_val=r_min,
        #     current_result_path=current_result_path,
        #     graph_path=graph_path,
        #     width_type="max",  # Use 'max', 'single_col', or e.g. 190.5
        #     cache_dump=False,
        #     cache_read=True
        # )
        # fig2 = plotter.plot_spatial_setup(
        #     reverse_homes=reverse_homes,
        #     n_val=nodes,
        #     r_val=r_min,
        #     current_result_path=current_result_path,
        #     width_type="max",
        # )

        fig3 = plotter.plot_connectivity_metrics(
            nodes,
            r_min,
            current_result_path,
            graph_path,
            cache_read=True,
            cache_dump=False,
        )
        print("Saving plot...")
        # plotter.savefigs(fig, fig_dir,
        #                  f'placements_periodic_{plotter.is_periodic}',
        #                  image_types)

        # print(f"Done. Figure saved to {fig_dir.resolve()}")
        plt.show()

    except Exception as e:
        print(f"An error occurred during plotting: {e}")
        import traceback

        traceback.print_exc()

    journal_style.reset_style()
