import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.gridspec import GridSpec
import json
import sys
from analysis_normalize import add_normalized_metrics
import matplotlibheader as mplhead
import pdist
import plot_helper
import plotter as plotter_module
import pdist
import traceback
import plot_styler
import utils

print("running stuff")


fields = [
    "Path",
    "File",
    "Periodic",
    "Conformation",
    "Id",
    "Initially placed nodes",
    "Tuning parameter",
    "Target <k>",
    "Connection probability function",
    "Nodes",
    "Edges",
    "Density",
    "Connected",
    "Nodes (Giant)",
    "Edges (Giant)",
    "Density (Giant)",
    "Mean degree",
    "Median degree",
    "Diameter",
    "Average shortest path",
    "Mean clustering",
    "Transitivity",
    "Mean Rich club coefficient",
    "Median Rich club coefficient",
    "Degree assortativity",
    "Mean Betweenness centrality",
    "Median Betweenness centrality",
    "Mean Closeness centrality",
    "Median Closeness centrality",
    "Small-World Propensity",
    "Approximate_Small_World_Propensity",
    "Small-world coefficient (omega)",
    "Bound Small-world coefficient (omega)",
    "Small-world index (sigma)",
    "Q",
    "Inflation (Q)",
    "Communities",
    "Mean community size",
    "Median community size",
    "Global Efficiency",
    "Local Efficiency",
    "Mean Communicability",
    "Shortest communicability path",
    "Upper Mean Communicability (netneurotools)",
    "Lower Mean Communicability (netneurotools)",
    "Shortest communicability path (netneurotools)",
    "Upper Mean Communicability (netneurotools, norm)",
    "Lower Mean Communicability (netneurotools, norm)",
    "Shortest communicability path (netneurotools, norm)",
    "Median Communicability",
    "Max Communicability",
    "Mean Communicability (node normalized)",
    "Mean Communicability (edge normalized)",
    "Upper Mean Mean First Passage Time (netneurotools)",
    "Lower Mean Mean First Passage Time (netneurotools)",
    "Shortest Mean First Passage Time (netneurotools)",
    "Diffusion efficiency",
]

### Variables


RUN_FIGURE = {
    "fig1": False,
    "fig2": False,
    "fig3": False,
    "fig4": True,
    "fig5": False,
    "fig6": True,
    "fig7": False,
    "sup1": False,
    "sup2": False,
    "sup3": True,
}

periodic = [True, False]
r_list = [1e-3, 1e-4]
neurons = [1000, 500]


### Paths
top_path = pathlib.Path.cwd()
version_string = "2025-07-23"

result_path = top_path / f"results_revision_{version_string}"
graph_path = top_path / f"graphs_revision_{version_string}"

positions_path = result_path / "positions"

top_fdir = result_path / f"figures_revision_{version_string}"
top_fdir.mkdir(exist_ok=True)


testing = False

periodic = [False]  # , True]

image_types = [
    "png",
    "svg",
    "pdf",
    "tif",
]


###############################################################################
#                     Set Up plot style                                       #
###############################################################################

try:
    # journal_style = NatureStyler(font_size=7, palette='tol-bright')
    journal_style = plot_styler.PlosStyler(font_size=9, palette="wong")
    journal_style.apply_style(font_family="Arial")
except Exception as e:
    print(f"Error applying style: {e}")
    print("Falling back to default Matplotlib style.")
    journal_style = plot_styler.BasePlotStyler()

plotter = plot_helper.ScientificPlotter(journal_styler=journal_style)

###############################################################################
#                     Position and densities                                  #
###############################################################################
"""
5x2+1 fig
Positions from net 0
position, kde,  nearest neighbor distance, P(d) exp, P(d) log
"""

fig_id = [["a", "c", "e", "g", "i"], ["b", "d", "f", "h", "j"]]

plos_fdir = top_fdir / "plos"
plos_fdir.mkdir(exist_ok=True)

write_figures = True

homes = ["20", "50", "100", "200", "uni"]
reverse_homes = homes[::-1]
conformations = ["conformational", "longitudinal"]

n = neurons[0]
r = r_list[1]
nodes = n
r_min = r

for is_periodic in periodic:
    current_result_path = (
        positions_path / f"results_periodic_{str(is_periodic)}_{version_string}"
    )

    if RUN_FIGURE["fig1"]:

        print("Generating plot 1...")
        try:
            fig_1 = plotter.plot_spatial_setup(
                reverse_homes=reverse_homes,
                n_val=nodes,
                r_val=r_min,
                current_result_path=current_result_path,
                width_type="max",
            )

            if write_figures:
                print("Saving plot...")
                plotter.savefigs(fig_1, plos_fdir,
                                'figure_1',
                                image_types)
                print(f"Done. Figure saved to {plos_fdir.resolve()}")
            else:
                plt.show()

        except Exception as e:
            print(f"An error occurred during plotting of figure 1: {e}")
            traceback.print_exc()

    if RUN_FIGURE["fig2"]:

        print("Generating plot 2...")
        try:
            fig_2 = plotter.plot_connectivity_metrics(
                nodes,
                r_min,
                current_result_path,
                graph_path,
                height_mm=200,
                cache_read=True,
                cache_dump=False,
            )

            if write_figures:
                print("Saving plot...")
                plotter.savefigs(fig_2, plos_fdir,
                                'figure_2',
                                image_types)
                print(f"Done. Figure saved to {plos_fdir.resolve()}")
            else:
                plt.show()

        except Exception as e:
            print(f"An error occurred during plotting of figure 1: {e}")
            traceback.print_exc()

    if any([RUN_FIGURE["fig3"], RUN_FIGURE["fig4"], RUN_FIGURE["fig5"]]):
        sort_column = "Initially placed nodes"
        result_file = f"results_periodic_{is_periodic}_conformational_{r}_norm_latt.csv"
        result_df = pd.read_csv(result_path / result_file, index_col=False)
        print(result_df.head())
        result_df.replace('exponential', 'Exponential', inplace=True)
        result_df.replace('lognormal', 'Log-normal', inplace=True)

        grouped_df = (
            result_df.groupby(["Connection probability function",sort_column, "Tuning parameter"])[
                "Mean degree"
            ]
            .mean()
            .reset_index()
        )

        latex_code = grouped_df.to_latex(index=False, float_format="%.4f")

        print(latex_code)

        if RUN_FIGURE["fig3"]:
            data_columns_fig3 = [
                # "Density",
                "Mean degree",
                "Diameter",
                "Average shortest path",
                "Mean clustering",
                "Mean Betweenness centrality",
                "Mean Closeness centrality"
            ]
            x_label_columns_fig3 = [
                # r"Density",
                r"Average degree ($\langle k \rangle$)",
                r"Diameter",
                r"Average shortest path ($\langle d \rangle$)",
                r"Mean clustering ($\langle C \rangle$)",
                r"Betweenness centrality",
                r"Closeness centrality"
            ]

            fig_3 = plotter.plot_aggregated_dataframe(
                df=result_df,
                x_col=sort_column,
                x_label="$N_{{\mathrm{{init}}}}$",
                hue_col="Connection probability function",
                y_cols=data_columns_fig3,
                y_labels=x_label_columns_fig3,
                width_type="max",
            )

            if write_figures:
                print("Saving plot...")
                plotter.savefigs(fig_3, plos_fdir,
                                'figure_3',
                                image_types)
                print(f"Done. Figure saved to {plos_fdir.resolve()}")
            else:
                plt.show()

        if RUN_FIGURE["fig4"]:
            data_columns_fig4 = [
                "Small-World Propensity",
                "Degree assortativity",
                "Q",
                "Mean community size",
                "Mean Rich club coefficient",
            ]
            x_label_columns_fig4 = [
                r"Small-World Propensity ($\phi$)",
                r"Degree assortativity",
                r"Modularity ($Q$)",
                r"Mean community size",
                r"Mean Rich-club coefficient",
            ]

            fig_4 = plotter.plot_aggregated_dataframe(
                df=result_df,
                x_col=sort_column,
                x_label="$N_{{\mathrm{{init}}}}$",
                hue_col="Connection probability function",
                y_cols=data_columns_fig4,
                y_labels=x_label_columns_fig4,
                width_type="max",
            )

            if write_figures:
                print("Saving plot...")
                plotter.savefigs(fig_4, plos_fdir,
                                'figure_4',
                                image_types)
                print(f"Done. Figure saved to {plos_fdir.resolve()}")
            else:
                plt.show()

        if RUN_FIGURE["fig5"]:
            result_df["Norm. Diffusion efficiency"] = result_df["Diffusion efficiency"] / result_df["diffusion_efficiency_rand"]
            result_df["Norm. Local Efficiency"] = (
                result_df["Local Efficiency"] / result_df["local_efficiency_rand"]
            )
            result_df["Norm. Global Efficiency"] = (
                result_df["Global Efficiency"]
                / result_df["global_efficiency_rand"]
            )
            result_df["Norm. Communicability"] = (
                result_df["Communicability"]
                / result_df["communicability_rand"]
            )

            data_columns_fig5 = [
                "Norm. Global Efficiency",
                "Norm. Local Efficiency",
                "Norm. Diffusion efficiency",
                "Norm. Communicability",
            ]
            x_label_columns_fig5 = [
                r"Norm. Global Efficiency",
                r"Norm. Local Efficiency",
                r"Norm. Diffusion efficiency",
                r"Norm. Communicability",
            ]

            fig_5 = plotter.plot_aggregated_dataframe(
                df=result_df,
                x_col=sort_column,
                x_label="$N_{{\mathrm{{init}}}}$",
                hue_col="Connection probability function",
                y_cols=data_columns_fig5,
                y_labels=x_label_columns_fig5,
                width_type="max",
            )

            if write_figures:
                print("Saving plot...")
                plotter.savefigs(fig_5, plos_fdir,
                                'figure_5',
                                image_types)
                print(f"Done. Figure saved to {plos_fdir.resolve()}")
            else:
                plt.show()

            if RUN_FIGURE["sup2"]:
                sup2_columns = [
                    "Global Efficiency",
                    "Rel. Global Efficiency",
                    "Local Efficiency",
                    "Rel. Local Efficiency",
                    "Diffusion efficiency",
                    "Rel. Diffusion efficiency",
                    "Communicability",
                    "Rel. Communicability",
                ]
                sup2_labels = [
                    r"Global Efficiency",
                    r"Rel. Global Efficiency",
                    r"Local Efficiency",
                    r"Rel. Local Efficiency",
                    r"Diffusion efficiency",
                    r"Rel. Diffusion efficiency",
                    r"Communicability",
                    r"Rel. Communicability",
                ]

                fig_sup2 = plotter.plot_aggregated_dataframe(
                    df=result_df,
                    x_col=sort_column,
                    x_label="$N_{{\mathrm{{init}}}}$",
                    hue_col="Connection probability function",
                    y_cols=sup2_columns,
                    y_labels=sup2_labels,
                    width_type="max",
                )

                if write_figures:
                    print("Saving plot...")
                    plotter.savefigs(
                        fig_sup2,
                        plos_fdir,
                        "supplementary_figure_2",
                        image_types,
                    )
                    print(f"Done. Figure saved to {plos_fdir.resolve()}")
                else:
                    plt.show()

    if any([RUN_FIGURE["fig6"], RUN_FIGURE["fig7"], RUN_FIGURE["sup3"]]):
        sort_column = "Pruning degree"
        result_file = f"results_periodic_{is_periodic}_conformational_{r}_pruned_norm_latt.csv"
        result_df = pd.read_csv(result_path / result_file, index_col=False)

        print(result_df.head())
        result_df.replace('exponential', 'Exponential', inplace=True)
        result_df.replace('lognormal', 'Log-normal', inplace=True)

        counts = result_df.groupby(['Pruning degree', 'Connection probability function'])['Connected'] \
           .value_counts() \
           .unstack(fill_value=0)
        print(counts)

        # 1. Output the first 'counts' table (All functions, True/False)
        print("Latex for General Counts:")
        print(counts.to_latex(multirow=True)) # multirow=True makes nested indices look nicer

        # ... (Your logic filtering for Log-normal) ...
        # counts = (connected_log_df.groupby ... .unstack(fill_value=0))

        # 2. Output the second 'counts' table (Log-normal, True only)
        print("Latex for Log-normal Counts:")
        print(counts.to_latex(index=True))

        log_df = result_df[result_df["Connection probability function"] == "Log-normal"]
        connected_log_df = log_df[log_df["Connected"] == True]
        counts = (
            connected_log_df.groupby(["Pruning degree", "Connection probability function"])[
                "Connected"
            ]
            .value_counts()
            .unstack(fill_value=0)
        )
        print(counts)

        if RUN_FIGURE["fig6"]:
            data_columns_fig6 = [
                "Density",
                "Mean degree",
                "Diameter",
                "Average shortest path",
                "Mean clustering",
                "Small-World Propensity",
                "Mean Betweenness centrality",
                "Mean Closeness centrality",
                "Degree assortativity",
                "Q",
                "Mean community size",
                "Mean Rich club coefficient",
            ]
            x_label_columns_fig6 = [
                r"Density",
                r"Average degree ($\langle k \rangle$)",
                r"Diameter",
                r"Average shortest path ($\langle d \rangle$)",
                r"Mean clustering ($\langle C \rangle$)",
                r"Small-World Propensity ($\phi$)",
                r"Betweenness centrality",
                r"Closeness centrality",
                r"Degree assortativity",
                r"Modularity ($Q$)",
                "Mean community size",
                "Mean Rich-club coefficient",
            ]

            fig_6 = plotter.plot_aggregated_dataframe_variable_column(
                df=connected_log_df,
                x_col=sort_column,
                x_label="Pruning fraction",
                hue_col=None,#"Pruning degree",
                y_cols=data_columns_fig6,
                y_labels=x_label_columns_fig6,
                width_type="max",
                n_cols=3
            )

            if write_figures:
                print("Saving plot...")
                plotter.savefigs(fig_6, plos_fdir,
                                'figure_6',
                                image_types)
                print(f"Done. Figure saved to {plos_fdir.resolve()}")
            else:
                plt.show()

        if any([RUN_FIGURE["fig7"], RUN_FIGURE["sup3"]]):

            connected_log_df["Norm. Diffusion efficiency"] = (
                connected_log_df["Diffusion efficiency"]
                / connected_log_df["diffusion_efficiency_rand"]
            )
            connected_log_df["Norm. Local Efficiency"] = (
                connected_log_df["Local Efficiency"] / connected_log_df["local_efficiency_rand"]
            )
            connected_log_df["Norm. Global Efficiency"] = (
                connected_log_df["Global Efficiency"] / connected_log_df["global_efficiency_rand"]
            )
            connected_log_df["Norm. Communicability"] = (
                connected_log_df["Communicability"] / connected_log_df["communicability_rand"]
            )

            data_columns_fig7 = [
                "Norm. Global Efficiency",
                "Norm. Local Efficiency",
                "Norm. Diffusion efficiency",
                "Norm. Communicability",
            ]
            x_label_columns_fig7 = [
                r"Norm. Global Efficiency",
                r"Norm. Local Efficiency",
                r"Norm. Diffusion efficiency",
                r"Norm. Communicability",
            ]

            if RUN_FIGURE["fig7"]:
                fig_7 = plotter.plot_aggregated_dataframe_variable_column(
                    df=connected_log_df,
                    x_col=sort_column,
                    x_label="Pruning fraction",
                    hue_col=None,  # "Pruning degree",
                    y_cols=data_columns_fig7,
                    y_labels=x_label_columns_fig7,
                    width_type="max",
                    height_per_row_mm=60,
                    n_cols=2,
                )

                if write_figures:
                    print("Saving plot...")
                    plotter.savefigs(fig_7, plos_fdir,
                                    'figure_7',
                                    image_types)
                    print(f"Done. Figure saved to {plos_fdir.resolve()}")
                else:
                    plt.show()

            print(f"Generating supplementary figure 3...{RUN_FIGURE['sup3']}")
            if RUN_FIGURE["sup3"]:
                sup3_columns = [
                    "Global Efficiency",
                    "Rel. Global Efficiency",
                    "Local Efficiency",
                    "Rel. Local Efficiency",
                    "Diffusion efficiency",
                    "Rel. Diffusion efficiency",
                    "Communicability",
                    "Rel. Communicability",
                ]
                sup3_labels = [
                    r"Global Efficiency",
                    r"Rel. Global Efficiency",
                    r"Local Efficiency",
                    r"Rel. Local Efficiency",
                    r"Diffusion efficiency",
                    r"Rel. Diffusion efficiency",
                    r"Communicability",
                    r"Rel. Communicability",
                ]

                fig_sup3 = plotter.plot_aggregated_dataframe_variable_column(
                    df=connected_log_df,
                    x_col=sort_column,
                    x_label="Pruning fraction",
                    hue_col=None,  # "Pruning degree",
                    y_cols=sup3_columns,
                    y_labels=sup3_labels,
                    width_type="max",
                    height_per_row_mm=60,
                    n_cols=2,
                )

                if write_figures:
                    print("Saving plot...")
                    plotter.savefigs(
                        fig_sup3,
                        plos_fdir,
                        "supplementary_figure_3",
                        image_types,
                    )
                    print(f"Done. Figure saved to {plos_fdir.resolve()}")
                else:
                    plt.show()

L = 1
min_d = 1e-3
x = np.linspace(min_d, L, 2001)

if RUN_FIGURE["sup1"]:

    json_path = top_path / "results_json_2025-07-23-tuning"
    json_files = utils.get_rfiles(json_path, f"*k50_{r}.json")
    print(json_files)

    for is_periodic in periodic:
        fig, axs = plt.subplots(
            2,
            1,
            sharex=True,
            figsize=(132 * mplhead.mm, 120 * mplhead.mm),
            constrained_layout=True,
        )
        for ax in axs:
            ax.set_prop_cycle(mplhead.cb_cycler_lines)
            ax.set_ylabel(r"$P(d)$")
            ax.set_xlabel(r"Distance ($d$)")
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.vlines(
                2 * min_d, 0, 20, linestyles="dotted", color="k"
            )  # ,label=r'Smallest node distance')

        for jfile in json_files:
            print(jfile)
            with open(jfile) as f:
                d = json.load(f)
                print(d)
                for neighbors_or_k, tuning_parameter in d[str(is_periodic)].items():
                    print(f"Key: {is_periodic}")
                    print(f"Sub-key: {neighbors_or_k}, Sub-value: {tuning_parameter}")
                    position_res_path = (
                        top_path / f"results_periodic_{str(is_periodic)}"
                    )
                    if "alpha" in jfile.stem:
                        print("alpha")

                        axs[0].plot(
                            x / L,
                            pdist.exponential_decay(x, alpha=tuning_parameter, beta=1),
                            label=neighbors_or_k,
                            linewidth=2,
                        )
                        axs[0].set_title(
                            "Exponential", loc="center", fontsize=11
                        )  # , fontweight='bold')
                    elif "sigma" in jfile.stem:
                        print("sigma")
                        axs[1].plot(
                            x / L,
                            pdist.lognorm_decay(x, sigma=tuning_parameter, mu=0),
                            label=neighbors_or_k,
                            linewidth=2,
                        )
                        axs[1].set_title(
                            "Log-normal", loc="center", fontsize=11
                        )  # , fontweight='bold')

                axs[0].legend()
                # axs[0].set_ylim( 1)
                axs[0] = plotter_module.crisp_axis(axs[0])
                axs[1] = plotter_module.crisp_axis(axs[1])
                # axs[1].set_ylim(min_d, 2)

                fig_id = ["A", "B", "C"]
                axs[0].set_title(fig_id[0], loc="left", fontsize=11, fontweight="bold")
                axs[1].set_title(fig_id[1], loc="left", fontsize=11, fontweight="bold")

        if write_figures:
            plotter.savefigs(
                fig,
                plos_fdir,
                f"supplementary_figure_1_connection_probability_periodic_{is_periodic}",
                image_types,
            )
            print(f"Done. Figure saved to {plos_fdir.resolve()}")
        else:
            plt.show()
        # fig.tight_layout(pad=0.15)
        # plotter.savefigs(
        #     fig,
        #     plos_fdir,
        #     f"connection_probability_periodic_{is_periodic}",
        #     image_types,
        # )


journal_style.reset_style()
sys.exit()
