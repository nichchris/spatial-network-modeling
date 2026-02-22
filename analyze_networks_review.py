import pathlib
import re
import time
import itertools
import numpy as np
import networkx as nx
import small_world_propensity as swp
from netneurotools import metrics
import multiprocessing as mp
import pandas as pd
import json

import small_world_propensity as swp

import analysis, smallworld, coordinate, analyze_functions
import utils
import functools

unbuffered_print = functools.partial(print, flush=True)


def parallell_analyzer(p, version_string):
    """
    Just running analysis in parallell.
    """
    

    fields = [
        'Path',
        'File',
        'Periodic',
        'Conformation',
        'Id', 
        'Initially placed nodes', 
        'Tuning parameter',
        'Target <k>',
        'r min',
        'Connection probability function',
        'Nodes', 
        'Edges', 
        'Density', 
        'Connected', 
        'Nodes (Giant)', 
        'Edges (Giant)', 
        'Density (Giant)',
        'Mean degree', 
        'Median degree', 
        'Diameter',
        'Average shortest path', 
        'Mean clustering', 
        'Transitivity',
        'Mean Rich club coefficient',
        'Median Rich club coefficient',
        'Degree assortativity',
        'Mean Betweenness centrality',
        'Median Betweenness centrality',
        'Mean Closeness centrality',
        'Median Closeness centrality',
        'Small-World Propensity',
        'Small-world coefficient (omega)',
        'Bound Small-world coefficient (omega)',
        'Small-world index (sigma)',
        'Approximate_Small_World_Propensity',
        'Approximate Small-world coefficient (omega)',
        'Approximate Bound Small-world coefficient (omega)',
        'Approximate Small-world index (sigma)',
        'Q',
        'Inflation (Q)',
        'Communities',
        'Mean community size',
        'Median community size',
        'Global Efficiency',
        'Local Efficiency',
        'Mean Communicability', 
        'Shortest communicability path',
        'Upper Mean Communicability (netneurotools)',
        'Lower Mean Communicability (netneurotools)',
        'Shortest communicability path (netneurotools)',
        'Upper Mean Communicability (netneurotools, norm)', 
        'Lower Mean Communicability (netneurotools, norm)', 
        'Shortest communicability path (netneurotools, norm)',
        'Median Communicability',
        'Max Communicability', 
        'Mean Communicability (node normalized)',
        'Mean Communicability (edge normalized)',
        'Upper Mean Mean First Passage Time (netneurotools)', 
        'Lower Mean Mean First Passage Time (netneurotools)', 
        'Shortest Mean First Passage Time (netneurotools)',
        'Diffusion efficiency'
        ]
    
    data_dict = {field: None for field in fields}
    tic = utils.timer()           # Simple timer
    
    f = p.stem

    unbuffered_print(f'Running analysis on file: {p}.')

    info = re.split('_', p.stem)

    id = info[0]
    k_target = info[1]
    rmin = info[2]
    neighbors = info[3]
    tuning_parameter = info[4]
    connectivity_method = info[5]

    results_path = pathlib.Path.cwd() / f'results_revision_{version_string}'
    json_path = results_path / "results_json"
    json_path.mkdir(parents=True, exist_ok=True)

    current_json_name = f"{p.stem}_{p.parts[8]}_{re.split('_', p.parts[7])[1]}.json"
    current_json_file = json_path / current_json_name

    if current_json_file.is_file():
        unbuffered_print(f'File: {p} already exists')
        with open(current_json_file, 'r') as file:
            data_dict = json.load(file)
        return [data_dict]
    
    data_dict['Path'] = str(p)

    data_dict['File'] = p.stem

    data_dict['Periodic'] = re.split('_', p.parts[7])[1]

    data_dict['Conformation'] = p.parts[8]

    data_dict['Id'] = id

    data_dict['Initially placed nodes'] = neighbors

    data_dict['Target <k>'] = k_target
    data_dict['r min'] = rmin

    data_dict['Tuning parameter'] = tuning_parameter

    data_dict['Connection probability function'] = connectivity_method

    connectivity_method = p.parts[-3]
    data_dict['Connectivity method'] = connectivity_method        

    G = nx.read_gml(p, destringizer=float)
    adj = nx.to_numpy_array(G)

    utils.describe(G)

    nodes_G = len(G.nodes)
    data_dict['Nodes'] = nodes_G

    edges_G = len(G.edges)
    data_dict['Edges'] = edges_G

    density_G = nx.density(G)
    data_dict['Density'] = density_G

    connected = nx.is_connected(G)
    data_dict['Connected'] = str(connected)

    if not connected: 
        G = G.subgraph(max(nx.connected_components(G), key=len))
        adj = nx.to_numpy_array(G)
        
    unbuffered_print(f'First: {utils.describe(G)}')

    density_G_giant = nx.density(G)
    data_dict['Density (Giant)'] = density_G_giant

    nodes_G_giant = len(G.nodes)
    data_dict['Nodes (Giant)'] = nodes_G_giant

    edges_G_giant = len(G.edges)
    data_dict['Edges (Giant)'] = edges_G_giant

    k_mean = analysis.mean_degree(G)
    data_dict['Mean degree'] = k_mean

    k_median = analysis.median_degree(G)
    data_dict['Median degree'] = k_median
    
    diameter = nx.diameter(G)
    data_dict['Diameter'] = diameter
    
    shortest_path = nx.average_shortest_path_length(G)
    data_dict['Average shortest path'] = shortest_path
    
    c = nx.average_clustering(G)
    data_dict['Mean clustering'] = c

    c_global = nx.transitivity(G) # No weighted alternative
    data_dict['Transitivity'] = c_global

    # unbuffered_print(f'Before RC: {utils.timer(tic)}')
    rc_mean, rc_median,*_ = analysis.mean_rich_club_coefficient(G)
    data_dict['Mean Rich club coefficient'] = rc_mean
    data_dict['Median Rich club coefficient'] = rc_median

    degree_assortativity  = nx.degree_assortativity_coefficient(G)
    data_dict['Degree assortativity'] = degree_assortativity

    bc_mean, bc_median, *_  = analysis.betweenness_metrics(G)
    data_dict['Mean Betweenness centrality'] = bc_mean
    data_dict['Median Betweenness centrality '] = bc_median
    
    cc_mean, cc_median, *_ = analysis.closeness_metrics(G)
    data_dict['Mean Closeness centrality'] = cc_mean
    data_dict['Median Closeness centrality'] = cc_median
    
    phi_approx, omega_approx, omega_bound_approx, sigma_approx = smallworld.small_world_propensity_approximate(G)
    data_dict['Approximate_Small_World_Propensity'] = phi_approx
    data_dict['Approximate Small-world coefficient (omega)'] = omega_approx
    data_dict['Approximate Bound Small-world coefficient (omega)'] = omega_bound_approx
    data_dict['Approximate Small-world index (sigma)'] = sigma_approx

    phi = swp.small_world_propensity(adj)
    data_dict['Small-World Propensity Python port'] = phi['SWP'].values[0]

    if "conformational" in str(p):
        unbuffered_print(f'Before SWP: {utils.timer(tic)}')
        phi, omega, omega_bound, sigma = smallworld.small_world_propensity_generated_equivalents(G, nrand=100)
        data_dict['Small-World Propensity'] = phi
        data_dict['Small-world coefficient (omega)'] = omega
        data_dict['Bound Small-world coefficient (omega)'] = omega_bound
        data_dict['Small-world index (sigma)'] = sigma

        unbuffered_print(f'Before Q: {utils.timer(tic)}')
        [q_louvain, q_inflation, q_louvain_communities, q_louvain_mean_size, 
        q_louvain_median_size] = analysis.optimal_qlouvain_communities_double_pass(
        G, start=0.5, stop=10., first_increment=0.5, second_range=1, second_increment=.01, base_repeat=5, repeats=50)

        data_dict['Q'] = q_louvain
        data_dict['Inflation (Q)'] = q_inflation
        data_dict['Communities'] = q_louvain_communities
        data_dict['Mean community size'] = q_louvain_mean_size
        data_dict['Median community size'] = q_louvain_median_size
        
    # unbuffered_print(f'Before efficiency measures: {utils.timer(tic)}')
    local_eff = nx.local_efficiency(G)
    data_dict['Local Efficiency'] = local_eff
    global_eff = nx.global_efficiency(G)
    data_dict['Global Efficiency'] = global_eff

    # unbuffered_print(f'Before comm: {utils.timer(tic)}')
    # comm_mean, comm_med, comm_max, comm_min  = analysis.communicability_metrics(G)
    # data_dict['Mean Communicability'] = comm_mean
    # data_dict['Median Communicability'] = comm_med
    # data_dict['Max Communicability'] = comm_max
    # data_dict['Shortest communicability path'] = comm_min
    # data_dict['Mean Communicability (edge normalized)'] = comm_mean/edges_G_giant
    # data_dict['Mean Communicability (node normalized)'] = comm_med/nodes_G_giant

    # From netneurotools:
    comm_min, comm_umean, comm_lmean = analysis.shortest_communicability_path(G)
    data_dict['Upper Mean Communicability (netneurotools)'] = comm_umean
    data_dict['Lower Mean Communicability (netneurotools)'] = comm_lmean
    data_dict['Shortest communicability path (netneurotools)'] = comm_min

    comm_min, comm_umean, comm_lmean = analysis.shortest_communicability_path(G,normalize=True)
    data_dict['Upper Mean Communicability (netneurotools, norm)'] = comm_umean
    data_dict['Lower Mean Communicability (netneurotools, norm)'] = comm_lmean
    data_dict['Shortest communicability path (netneurotools, norm)'] = comm_min
    
    mfpt_min, mfpt_umean, mfpt_lmean = analysis.mean_first_passage_time_metrics(G)
    data_dict['Upper Mean Mean First Passage Time (netneurotools)'] = mfpt_umean
    data_dict['Lower Mean Mean First Passage Time (netneurotools)'] = mfpt_lmean
    data_dict['Shortest Mean First Passage Time (netneurotools)'] = mfpt_min

    diffusion_efficiency,*_ = metrics.diffusion_efficiency(adj)
    data_dict['Diffusion efficiency'] = diffusion_efficiency

    unbuffered_print(f'End for file {f}')
    utils.timer(tic)
    
    with open(current_json_file, 'w') as f:
        json.dump(data_dict, f)

    return [data_dict]
        # current_flow_betweenness_centrality(https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.centrality.current_flow_betweenness_centrality.html#networkx.algorithms.centrality.current_flow_betweenness_centralityt)
        

if __name__ == "__main__":
    # time.sleep(15000)
    tic = utils.timer()
    start_time = time.localtime()
    current_time = time.strftime("%H:%M:%S", start_time)
    print("Start time is: ", current_time)

    version_string = "2025-07-23"
    pool_size = 6

    current_path = pathlib.Path.cwd()
    graph_path = pathlib.Path.cwd() / f'graphs_revision_{version_string}'
    results_path = pathlib.Path.cwd() / f'results_revision_{version_string}'
    results_path.mkdir(parents=True, exist_ok=True)
    
    pruning_degrees = [0, 0.005, 0.01, .05, 0.1]

    periodic = [True, False]
    conformations = ['conformational']#, 'longitudinal']
    
    testing = True

    r_list = [1e-3, 1e-4]

    for r in r_list:
        ext = f'*_{str(r)}_*.gml'

        for conformation in conformations:
            for is_periodic in periodic:
                outer_data_list = []
                network_path = graph_path / f"periodic_{is_periodic}" / conformation /str(r)
                paths = utils.get_rfiles(network_path, ext)
                results_file = f"results_periodic_{is_periodic}_{conformation}_{str(r)}.csv"
                if testing:
                    print(network_path, len(paths))        
                    for p in paths:
                        outer_data_list.append(analyze_functions.default_analyzer(p, version_string))
                else:
                    with mp.Pool(processes=pool_size) as pool:
                        args_list = [(path, version_string) for path in paths]

                        # outer_data_list = list(pool.starmap(parallell_analyzer, args_list))
                        outer_data_list = list(pool.starmap(analyze_functions.default_analyzer, args_list))
                        # async_result = p.map_async(parallell_analyzer, paths)
                        # data_list = async_result.get()

                print(len(outer_data_list))
                flat_list = list(itertools.chain.from_iterable(outer_data_list))
                print(len(flat_list))

                current_results_path = results_path / results_file
                df_results = pd.DataFrame.from_dict(flat_list)
                df_results.to_csv(current_results_path)

                toc = utils.timer(tic)

    toc = utils.timer(tic)

    # for r in r_list:
    #     prune_ext = f'*_{str(r)}_20_*.gml'
    #     for conformation in conformations:
    #         for is_periodic in periodic:
    #             # Try with pruning of longest connections:
    #             pruning_outer_data_list = []
                
    #             pruning_network_path = graph_path / f"periodic_{is_periodic}" / conformation / str(r)
                
    #             prune_paths = utils.get_rfiles(pruning_network_path, prune_ext)
    #             prune_results_file = f"results_periodic_{is_periodic}_{conformation}_{str(r)}_pruned.csv"
                
    #             if testing:
    #                 print(pruning_network_path, len(prune_paths))        
    #                 for p in prune_paths:
    #                     pruning_outer_data_list.append(analyze_functions.pruning_analyzer(p, pruning_degrees, version_string))
    #             else:
    #                 with mp.Pool(processes=pool_size) as pool:
                        
    #                     args_list = [(prune_path, pruning_degrees, version_string) for prune_path in prune_paths]

    #                     pruning_outer_data_list = list(pool.starmap(analyze_functions.pruning_analyzer, args_list))
    #                     # async_result = p.map_async(parallell_analyzer, paths)
    #                     # data_list = async_result.get()

    #             print(len(pruning_outer_data_list))
    #             prune_flat_list = list(itertools.chain.from_iterable(pruning_outer_data_list))
    #             print(len(prune_flat_list))

    #             current_results_path = results_path / prune_results_file
    #             df_results = pd.DataFrame.from_dict(prune_flat_list)
    #             df_results.to_csv(current_results_path)
        
    # toc = utils.timer(tic)