import utils
import re
import pathlib
import math
import networkx as nx
import numpy as np
import pandas as pd
import json
import sys

import analysis
import geometric
import pdist

###############################################################################
#       Set up utility functions.                                             #
###############################################################################
def to_bool(value: str) -> bool:
    """
    Converts a string to a boolean, raising an error for invalid values.
    Handles 'true'/'false' case-insensitively and strips whitespace.
    """
    clean_value = value.strip().lower()
    if clean_value == 'true':
        return True
    elif clean_value == 'false':
        return False
    else:
        raise ValueError(f"Invalid boolean string: '{value}'")
    
def p_dist(dist, L=1, beta=1, alpha=3):
    return beta * math.exp(-dist / (alpha * L))

def new_lognorm_decay(dist, L=1, beta=10.0, alpha=0.00):
    return pdist.lognorm_decay(dist, L=L, sigma=beta, mu=alpha)

if __name__ == '__main__':
    print('running stuff')
    tic = utils.timer()
    print(f"Start time is {tic}.")
    ###############################################################################
    #       Set up results folders.                                               #
    ###############################################################################
    top_path = pathlib.Path.cwd()

    version_string = "2025-07-23"


    # res_path = top_path / f'results_revision_{version_string}'
    # res_path.mkdir(exist_ok=True)

    graph_path = top_path / f'graphs_revision_{version_string}'
    graph_path.mkdir(parents=True, exist_ok=True)

    ###############################################################################
    #       Set up parameters to test.                                            #
    ###############################################################################

    mu = 0.0
    beta = 1.0
    l = 1

    # Set up random seed handling:
    seed = 123456
    seeds = np.random.SeedSequence(seed)
    child_seed = seeds.spawn(500*2*2*2+300*2*2)
    
    # Seed index
    s = 0

    json_version_string = "2025-07-23-tuning"
    pos_version_string = "2025-07-23"

    json_path = top_path / f'results_json_{json_version_string}'

    r_list = [1e-3, 1e-4]
    for r in r_list:
        # Read JSON-files
        json_files = utils.get_rfiles(json_path, f"*{r}.json")
        print(json_files)
        
        for jfile in json_files:
            print(jfile)
            with open(jfile) as f:
                d = json.load(f)
                print(d)
                for is_periodic, data in d.items():
                    clean_is_periodic = to_bool(is_periodic)

                    for neighbors_or_k, tuning_parameter in data.items():
                        print(f"Key: {is_periodic}")
                        print(f"Sub-key: {neighbors_or_k}, Sub-value: {tuning_parameter}")

                        position_res_path = top_path / f'results_periodic_{str(is_periodic)}_{pos_version_string}'
                        this_graph_path = graph_path / f'periodic_{str(is_periodic)}'
                        this_graph_path.mkdir(parents=True, exist_ok=True)
                        
                        if "k50" in jfile.stem:
                            if neighbors_or_k == "1000":
                                pos_ext = f'*_uni_*{r}_2.csv'
                            else:
                                pos_ext = f'*_{r}_{neighbors_or_k}_*2.csv'
                            experiment_type = "conformational"
                            max_id = 100
                        
                        elif "long" in jfile.stem:
                            experiment_type = "longitudinal"
                            max_id = 20
                            pos_ext = '*uni*_2.csv'
                        
                        conf_path = this_graph_path / experiment_type / str(r)
                        conf_path.mkdir(parents=True, exist_ok=True)
                        paths = utils.get_rfiles(position_res_path, pos_ext)

                        for p in paths:
                            info = re.split('_ |_| ', p.stem)
                            id = info[0]
                            method = info[1]
                            if method == 'pl':
                                alpha = info[2]
                                nodes = info[3]
                                neighbors = info[5]
                            else:
                                nodes = info[2]
                                alpha = 0
                                neighbors = nodes
                            
                            # if "alpha" in jfile.stem:

                            if int(id) < max_id:
                                print(f"Is_Periodic: {is_periodic} and tuning parameter {tuning_parameter} for group {neighbors_or_k}")
                                print(jfile)
                                print(p)

                                df_pos = pd.read_csv(p, sep=',', index_col=0)
                                dpos = dict(enumerate(list(df_pos[['x', 'y']].to_numpy())))
                                dpos = {k: v.tolist() for k, v in dpos.items()}
                                rng = np.random.default_rng(child_seed[s])

                                is_connected = False
                                net_tries = 0
                                net_tries = 0

                                if "sigmas" in jfile.stem:
                                    while not is_connected:

                                        G = geometric.waxman_graph_mod(int(nodes), pos=dpos,
                                                                        p_dist=new_lognorm_decay,
                                                                        L=l, periodic=clean_is_periodic, seed=rng,
                                                                        beta=tuning_parameter, alpha=mu)
                                        ext = '_lognormal.gml'
                                        is_connected = nx.is_connected(G)
                                        net_tries+=1

                                elif "alpha" in jfile.stem:
                                    while not is_connected:
                                        G = geometric.waxman_graph_mod(int(nodes), pos=dpos,
                                                                    p_dist=None,
                                                                    L=l, periodic=clean_is_periodic, seed=rng,
                                                                    beta=beta, alpha=tuning_parameter)
                                        ext = '_exponential.gml'
                                        is_connected = nx.is_connected(G)
                                        net_tries+=1
                                
                                print(f"Tries to create connected network of type {ext} is {net_tries}")
                                s += 1
                            
                                print(info, len(G.edges())/int(nodes), f'Mean degree is {analysis.mean_degree(G)}')

                                graph_f = conf_path / \
                                    pathlib.Path(id + "_"  + str(50) + "_" + str(r) + "_" + str(neighbors_or_k) + "_" + str(tuning_parameter) + ext)
                                nx.write_gml(G, graph_f)
                                toc = utils.timer(tic)
                                print(f"Passed time is {toc}.")

        toc = utils.timer(tic)
        print(f"Total passed time at end is {toc}.")
# for periodic in is_periodic:
#     print(f'RUNNING TESTS FOR PERIODIC = {str(periodic)}')
#     res_path = top_path / f'results_periodic_{str(periodic)}'
#     res_path.mkdir(exist_ok=True)

#     ext = '*uni*2.csv'
#     paths = utils.get_rfiles(res_path, ext)

#     df_sigmas = pd.DataFrame(0, index=range(5), columns=sigmas)
#     df_alphas = pd.DataFrame(0, index=range(5), columns=alphas)

#     mus = [0.0]
#     betas = [1]

#     print(f'Alphas: {alphas}')
#     print(f'Sigmas: {sigmas}')

#     seed = 12345
#     seeds = np.random.SeedSequence(seed)
#     child_seed = seeds.spawn(len(paths)*len(is_periodic)*len(sigmas))
#     print(len(paths)*len(is_periodic)*len(sigmas))
#     s = 0

#     for p in paths:

#         info = re.split('_ |_| ', p.stem)
#         id = info[0]
#         method = info[1]
#         if method == 'pl':
#             alpha = info[2]
#             nodes = info[3]
#             neighbors = info[4]
#         else:
#             nodes = info[2]
#             alpha = 0
#             neighbors = 0
#         # print(info, alpha)
#         if int(id) < 5:
#             print(p.stem)
#             df_pos = pd.read_csv(p, sep=',', index_col=0)
#             dpos = dict(enumerate(list(df_pos[['x', 'y']].to_numpy())))
#             dpos = {k: v.tolist() for k, v in dpos.items()}

#             alphas_k = []
#             sigmas_k = []

#             for i, param in enumerate(product(sigmas, mus)):
#                 rng = np.random.default_rng(child_seed[s])

#                 G = geometric.waxman_graph_mod(int(nodes), pos=dpos,
#                                             p_dist=new_lognorm_decay,
#                                             L=1, periodic=periodic, seed=rng,
#                                             beta=param[0], alpha=param[1])
#                 s += 1
#                 sigmas_k.append(analysis.mean_degree(G))

#                 print(param, len(G.edges())/int(nodes), f'Mean degree is {analysis.mean_degree(G)}')
#                 print(G)

#             for i, param in enumerate(product(betas, alphas)):
#                 rng = np.random.default_rng(child_seed[s])

#                 G = geometric.waxman_graph_mod(int(nodes), pos=dpos,
#                                             p_dist=None,
#                                             L=1, periodic=periodic, seed=rng,
#                                             beta=param[0], alpha=param[1])
#                 s += 1
#                 alphas_k.append(analysis.mean_degree(G))
                
#                 print(param, len(G.edges())/int(nodes), f'Mean degree is {analysis.mean_degree(G)}')
#                 print(G)

#             df_sigmas.loc[int(id)] = sigmas_k
#             df_alphas.loc[int(id)] = alphas_k

#     df_alphas.to_csv(f'output_uni_alphas_{str(len(alphas))}_periodic_{str(periodic)}.csv')
#     df_sigmas.to_csv(f'output_uni_sigmas_{str(len(sigmas))}_periodic_{str(periodic)}.csv')
#     print(s)
# """
# ###############################################################################
# #           Continuing with checking clustered distributions                  #
# ###############################################################################
# # sys.exit()
# """
# top_path = pathlib.Path.cwd()

# homes = [20, 50, 100, 200]

# for h in homes:
#     for periodic in is_periodic:
#         print(f'RUNNING TESTS FOR PERIODIC = {str(periodic)}')
#         res_path = top_path / f'results_periodic_{str(periodic)}'
#         res_path.mkdir(exist_ok=True)


#         ext = f'*pl*_{str(h)}_2.csv'
#         paths = utils.get_rfiles(res_path, ext)
        
#         mus = [0]
#         betas = [1]

#         df_sigmas = pd.DataFrame(0, index=range(5), columns=sigmas)
#         df_alphas = pd.DataFrame(0, index=range(5), columns=alphas)

#         seed = 12345
#         seeds = np.random.SeedSequence(seed)
#         child_seed = seeds.spawn(len(paths)*2*49)
#         print(len(paths)*2*49)
#         s = 0

#         for p in paths:

#             info = re.split('_ |_| ', p.stem)
#             id = info[0]
#             method = info[1]
#             if method == 'pl':
#                 alpha = info[2]
#                 nodes = info[3]
#                 neighbors = info[4]
#             else:
#                 nodes = info[2]
#                 alpha = 0
#                 neighbors = 0

#             if int(id) < 5:
#                 print(info, alpha)
#                 df_pos = pd.read_csv(p, sep=',', index_col=0)
#                 dpos = dict(enumerate(list(df_pos[['x', 'y']].to_numpy())))
#                 dpos = {k: v.tolist() for k, v in dpos.items()}
    
#                 alphas_k = []
#                 sigmas_k = []

#                 for i, param in enumerate(product(sigmas, mus)):
#                     rng = np.random.default_rng(child_seed[s])
#                     G = geometric.waxman_graph_mod(int(nodes), pos=dpos,
#                                                 p_dist=new_lognorm_decay,
#                                                 L=1, periodic=periodic, seed=rng,
#                                                 beta=param[0], alpha=param[1])
#                     while not nx.is_connected(G):
#                         G = geometric.waxman_graph_mod(int(nodes), pos=dpos,
#                                                     p_dist=new_lognorm_decay,
#                                                     L=1, periodic=periodic, seed=rng,
#                                                     beta=param[0], alpha=param[1])
#                         print(f'not connected for the {tries}\'th time for lognormal')

#                     s+=1
#                     sigmas_k.append(analysis.mean_degree(G))
#                     print(param, len(G.edges())/int(nodes), f'Mean degree is {analysis.mean_degree(G)}')


#                 for i, param in enumerate(product(betas, alphas)):
#                     rng = np.random.default_rng(child_seed[s])
#                     tries = 0
#                     G = geometric.waxman_graph_mod(int(nodes), pos=dpos,
#                                                 p_dist=None,
#                                                 L=1, periodic=periodic, seed=rng,
#                                                 beta=param[0], alpha=param[1])
#                     while not nx.is_connected(G):
#                         G = geometric.waxman_graph_mod(int(nodes), pos=dpos,
#                                                     p_dist=None,
#                                                     L=1, periodic=periodic, seed=rng,
#                                                     beta=param[0], alpha=param[1])
#                         tries+=1
#                         print(f'not connected for the {tries}\'th time for exponential')

#                     s += 1
#                     alphas_k.append(analysis.mean_degree(G))
#                     print(param, len(G.edges())/int(nodes), f'Mean degree is {analysis.mean_degree(G)}')

#                 df_sigmas.loc[int(id)] = sigmas_k
#                 df_alphas.loc[int(id)] = alphas_k

#         df_alphas.to_csv(f'output_pl_{str(h)}_alphas_{str(len(alphas))}_periodic_{str(periodic)}.csv')
#         df_sigmas.to_csv(f'output_pl_{str(h)}_sigmas_{str(len(sigmas))}_periodic_{str(periodic)}.csv')
#         print(s)
