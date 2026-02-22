import utils
import stats
import re
import pathlib
from itertools import product
import math
import time
from itertools import accumulate, combinations, product
from os.path import exists
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import scipy as sp
import sys
import json
import multiprocessing as mp
import functools

import analysis
import geometric
import pdist

unbuffered_print = functools.partial(print, flush=True)


print('running stuff')
###############################################################################
#           Starting with checking uniform distributions                      #
###############################################################################

rng = np.random.default_rng()

# n = 1000
# runs = 50

###############################################################################
#           Set up parameters to test.                                        #
###############################################################################

sigmas = [65.4]  # 12.8],6.1]

alphas = [0.059]  # 0.1394 ,0.22]


numbers = 1000
# Ranges 20 < <k> < 300
sigmas = np.geomspace(3.15, 75.3, numbers)
alphas = np.linspace(0.055, .39, numbers)

# sigmas = [
#         75.30, 65.3, 43.7, 32.85, 26.17, 21.8, 18.54, 16.12, 14.3, 12.88, 
#         11.63, 10.59, 9.7, 9.01, 8.3, 7.75, 7.3, 6.85, 6.47, 6.11, 5.75, 
#         5.502, 5.22, 4.96, 4.74, 4.53, 4.32, 4.146, 3.98, 3.82, 3.2669, 
#         3.522, 3.298, 3.268, 3.15
#         ]

# alphas = [
#         0.049, 0.059, 0.073, 0.084, 0.0946, 0.1038, 0.114, 0.122, 0.13, 
#         0.1394, 0.1464, 0.155, 0.1628, 0.171, 0.1791, 0.1867, 0.195, 
#         0.203, 0.2118, 0.219, 0.2275, 0.2357, 0.245, 0.256, 0.2617, 
#         0.271, 0.2801, 0.28899, 0.2983, 0.3093, 0.318, 0.3285, 0.339, 
#         0.3495, 0.3597, 0.37
#         ]


def p_dist(dist, L=1, beta=1, alpha=3):
    return beta * math.exp(-dist / (alpha * L))

def new_lognorm_decay(dist, L=1, beta=10.0, alpha=0.00):
    return pdist.lognorm_decay(dist, L=L, sigma=beta, mu=alpha)

is_periodic = [True, False]

top_path = pathlib.Path.cwd()

r_list = [1e-3, 1e-4]
neurons = [1000]#, 500]
"""
n_threshold = 10
for n in neurons:
    for r in r_list:
        for periodic in is_periodic:
            unbuffered_print(f'RUNNING TESTS FOR PERIODIC = {str(periodic)}')
            res_path = top_path / f'results_periodic_{str(periodic)}'
            res_path.mkdir(exist_ok=True)

            ext = f'*_uni_{str(n)}_{str(r)}*2.csv'
            paths = utils.get_rfiles(res_path, ext)

            df_sigmas = pd.DataFrame(0, index=range(n_threshold), columns=sigmas)
            df_alphas = pd.DataFrame(0, index=range(n_threshold), columns=alphas)

            mus = [0.0]
            betas = [1]

            unbuffered_print(f'Alphas: {alphas}')
            unbuffered_print(f'Sigmas: {sigmas}')

            seed = 12345
            seeds = np.random.SeedSequence(seed)
            child_seed = seeds.spawn(len(paths)*len(is_periodic)*len(sigmas))
            unbuffered_print(len(paths)*len(is_periodic)*len(sigmas))
            s = 0

            for p in paths:

                info = re.split('_ |_| ', p.stem)
                id = info[0]
                method = info[1]
                if method == 'pl':
                    alpha = info[2]
                    nodes = info[3]
                    neighbors = info[4]
                else:
                    nodes = info[2]
                    alpha = 0
                    neighbors = 0
                # unbuffered_print(info, alpha)
                if int(id) < n_threshold:
                    unbuffered_print(p.stem)
                    df_pos = pd.read_csv(p, sep=',', index_col=0)
                    dpos = dict(enumerate(list(df_pos[['x', 'y']].to_numpy())))
                    dpos = {k: v.tolist() for k, v in dpos.items()}

                    alphas_k = []
                    sigmas_k = []

                    for i, param in enumerate(product(sigmas, mus)):
                        rng = np.random.default_rng(child_seed[s])

                        G = geometric.waxman_graph_mod(int(nodes), pos=dpos,
                                                    p_dist=new_lognorm_decay,
                                                    L=1, periodic=periodic, seed=rng,
                                                    beta=param[0], alpha=param[1])
                        s += 1
                        sigmas_k.append(analysis.mean_degree(G))

                        unbuffered_print(param, len(G.edges())/int(nodes), f'Mean degree is {analysis.mean_degree(G)}')
                        unbuffered_print(G)

                    for i, param in enumerate(product(betas, alphas)):
                        rng = np.random.default_rng(child_seed[s])

                        G = geometric.waxman_graph_mod(int(nodes), pos=dpos,
                                                    p_dist=None,
                                                    L=1, periodic=periodic, seed=rng,
                                                    beta=param[0], alpha=param[1])
                        s += 1
                        alphas_k.append(analysis.mean_degree(G))
                        
                        unbuffered_print(param, len(G.edges())/int(nodes), f'Mean degree is {analysis.mean_degree(G)}')
                        unbuffered_print(G)

                    df_sigmas.loc[int(id)] = sigmas_k
                    df_alphas.loc[int(id)] = alphas_k

            df_alphas.to_csv(f'output_uni_alphas_{str(len(alphas))}_periodic_{str(periodic)}_{str(n)}_{str(r)}.csv')
            df_sigmas.to_csv(f'output_uni_sigmas_{str(len(sigmas))}_periodic_{str(periodic)}_{str(n)}_{str(r)}.csv')
            unbuffered_print(s)
# """
###############################################################################
#           Continuing with checking clustered distributions                  #
###############################################################################
# sys.exit()
# """
# Ranges <k> = 50
# """
# homes = [20, 50, 100, 200]

def run_optimization(home):
    h = home
    numbers = 12
    # homes_list = [["20", "50", "100", "200", "uni"],
    #      ["10", "25", "50", "100", "uni"]]
    version_string = '2025-07-23-tuning'
    top_path = pathlib.Path.cwd()# / f'results_json_250601'
    sub_json_path = top_path / f'results_json_{version_string}'
    sub_json_path.mkdir(parents=True, exist_ok=True)

    if home == '20':
        sigma_start = 28
        sigma_stop = 85
        alpha_start = .042
        alpha_stop = .080
        print(home,sigma_start,sigma_stop, alpha_start, alpha_stop)
    elif home == '50':
        sigma_start = 22
        sigma_stop = 53
        alpha_start = .065
        alpha_stop = .093
        print(home,sigma_start,sigma_stop, alpha_start,alpha_stop)
    elif home == '100':
        sigma_start = 20.5
        sigma_stop = 37
        alpha_start = .079
        alpha_stop = .102
        print(home,sigma_start,sigma_stop, alpha_start,alpha_stop)
    elif home == '200':
        sigma_start = 19.3
        sigma_stop = 35
        alpha_start = .0817
        alpha_stop = .1055
        print(home,sigma_start,sigma_stop, alpha_start,alpha_stop)
    elif home == 'uni':
        sigma_start = 18.5 #22
        sigma_stop = 27.3
        alpha_start = .0859
        alpha_stop = .1045
        print(home,sigma_start,sigma_stop, alpha_start,alpha_stop)

    # sigmas = np.geomspace(sigma_start, sigma_stop, numbers)
    # alphas = np.linspace(alpha_start, alpha_stop, numbers)
    # sigmas = np.geomspace(18, 85, numbers)
    # alphas = np.linspace(0.042, .105, numbers)
    
    r_list = [1e-4, 1e-3]
    neurons = [1000]#, 500]
    n = neurons[0]
    is_periodic = [False]#, True]

    n_threshold = 10
            
    for r in r_list:
        for periodic in is_periodic:
            unbuffered_print(f'RUNNING TESTS FOR PERIODIC = {str(periodic)}, r = {r}, and home = {h}')
            res_path = top_path / f'results_periodic_{str(periodic)}_{version_string}'
            res_path.mkdir(exist_ok=True)

            if "uni" in h:
                ext = f'*{str(h)}_{str(n)}_{str(r)}_2.csv'
            else:
                ext = f'*{str(n)}_{str(r)}_{str(h)}_2.csv'                    
            
            paths = utils.get_rfiles(res_path, ext)
            
            mus = [0]
            betas = [1]

            df_sigmas = pd.DataFrame(0, index=range(n_threshold), columns=sigmas)
            df_alphas = pd.DataFrame(0, index=range(n_threshold), columns=alphas)

            seed = 12345
            seeds = np.random.SeedSequence(seed)
            child_seed = seeds.spawn(n_threshold*2*numbers*n_threshold)
            s = 0

            for p in paths:

                info = re.split('_ |_| ', p.stem)
                id = info[0]
                method = info[1]
                if method == 'pl':
                    alpha = info[2]
                    nodes = info[3]
                    neighbors = info[4]
                else:
                    nodes = info[2]
                    alpha = 0
                    neighbors = 0

                if int(id) < n_threshold:
                    unbuffered_print(f'RUNNING TESTS FOR PERIODIC = {str(periodic)} file {p}')
                    unbuffered_print(info, alpha, p.stem, periodic)
                    df_pos = pd.read_csv(p, sep=',', index_col=0)
                    dpos = dict(enumerate(list(df_pos[['x', 'y']].to_numpy())))
                    dpos = {k: v.tolist() for k, v in dpos.items()}
        
                    alphas_k = []
                    sigmas_k = []

                    for i, param in enumerate(product(sigmas, mus)):
                        rng = np.random.default_rng(child_seed[s])
                        tries = 0
                        G = geometric.waxman_graph_mod(int(nodes), pos=dpos,
                                                    p_dist=new_lognorm_decay,
                                                    L=1, periodic=periodic, seed=rng,
                                                    beta=param[0], alpha=param[1])
                        
                        # while not nx.is_connected(G) and tries < 1000:
                        #     G = geometric.waxman_graph_mod(int(nodes), pos=dpos,
                        #                                 p_dist=new_lognorm_decay,
                        #                                 L=1, periodic=periodic, seed=rng,
                        #                                 beta=param[0], alpha=param[1])
                        #     tries+=1
                        #     unbuffered_print(f'not connected for the {tries}\'th time for lognormal')
                        #     unbuffered_print(param, len(G.edges())/int(nodes), f'Mean degree is {analysis.mean_degree(G)}')

                        s+=1
                        sigmas_k.append(analysis.mean_degree(G))

                    for i, param in enumerate(product(betas, alphas)):
                        rng = np.random.default_rng(child_seed[s])
                        tries = 0
                        G = geometric.waxman_graph_mod(int(nodes), pos=dpos,
                                                    p_dist=None,
                                                    L=1, periodic=periodic, seed=rng,
                                                    beta=param[0], alpha=param[1])
                        
                        # while not nx.is_connected(G) and tries < 1000:
                        #     G = geometric.waxman_graph_mod(int(nodes), pos=dpos,
                        #                                 p_dist=None,
                        #                                 L=1, periodic=periodic, seed=rng,
                        #                                 beta=param[0], alpha=param[1])
                        #     tries+=1
                        #     unbuffered_print(f'not connected for the {tries}\'th time for exponential')
                        #     unbuffered_print(param, len(G.edges())/int(nodes), f'Mean degree is {analysis.mean_degree(G)}')

                        s += 1
                        alphas_k.append(analysis.mean_degree(G))

                    df_sigmas.loc[int(id)] = sigmas_k
                    df_alphas.loc[int(id)] = alphas_k

            df_sigmas.to_csv(sub_json_path / f'output_pl_{str(h)}_sigmas_{str(len(sigmas))}_periodic_{str(periodic)}_{str(n)}_{str(r)}.csv')
            df_alphas.to_csv(sub_json_path / f'output_pl_{str(h)}_alphas_{str(len(alphas))}_periodic_{str(periodic)}_{str(n)}_{str(r)}.csv')
    
    return h
# """        
"""
    elif number == 1000:
        for n in neurons:
            k_target = np.array([ 20.,  40.,  60.,  80., 100., 120., 140., 160., 180., 200., 220., 240., 260., 280., 300.])*(n/1000)
            neighbors = str(n)

            for r in r_list:
                alphas_dict_long = {"True": {},
                                    "False": {}}
                sigmas_dict_long = {"True": {},
                                    "False": {}}
                for periodic in is_periodic:
                    alphas_file = f'output_uni_alphas_{str(number)}_periodic_{str(periodic)}_{str(n)}_{str(r)}.csv'
                    sigmas_file = f'output_uni_sigmas_{str(number)}_periodic_{str(periodic)}_{str(n)}_{str(r)}.csv'

                    df_alphas = pd.read_csv(top_path / alphas_file, index_col=0)
                    df_sigmas = pd.read_csv(top_path / sigmas_file, index_col=0)
                    
                    for k in k_target:
                        s_alpha_diff = abs(df_alphas.mean()-k)
                        alphas_dict_long[str(periodic)][str(k)] = np.round(float(s_alpha_diff.idxmin()), 5)

                        s_sigma_diff = abs(df_sigmas.mean()-k)
                        sigmas_dict_long[str(periodic)][str(k)] = np.round(float(s_sigma_diff.idxmin()), 2)

                alpha_uni_save_json =  sub_json_path / f'dict_alphas_uni_{r}.json'
                sigma_uni_save_json =  sub_json_path / f'dict_sigmas_uni_{r}.json'

                with open(sigma_uni_save_json, 'w') as f:
                    json.dump(sigmas_dict_long, f)

                with open(alpha_uni_save_json, 'w') as f:
                    json.dump(alphas_dict_long, f)

    # # """ 


if __name__ == "__main__":
    tic = utils.timer()
    start_time = time.localtime()
    current_time = time.strftime("%H:%M:%S", start_time)
    print("Start time is: ", current_time)

    homes = ["20", "50", "100", "200", "uni"]
    r_list = [1e-4, 1e-3]

    testing = False

    # if testing:
    #     for h in homes:
    #         run_optimization(h)
    # else:
    #     with mp.Pool(processes=5) as pool:
    #         foo = list(pool.imap_unordered(run_optimization, homes))
    
    toc = utils.timer(tic)

    
    numbers_list = [250, 500, 1000]
    neurons = [1000]#, 500]
    # homes_list = [["20", "50", "100", "200", "uni"],
    #         ["10", "25", "50", "100", "uni"]]

    # homes = homes_list[0]
    json_version_string = "2025-07-23-tuning"

    top_path = pathlib.Path.cwd()# / f'results_json_250601'
    sub_json_path = top_path / f'results_json_{json_version_string}'
    sub_json_path.mkdir(parents=True, exist_ok=True)


    for number in numbers_list:
        if number == 1000:
            for n in neurons:
                for r in r_list:
                    alphas_dict_fifty = {"True": {},
                                "False": {}}
                    sigmas_dict_fifty = {"True": {},
                                        "False": {}} 
                    for h in homes:
                        for periodic in is_periodic:  
                            if h == 'uni':
                                neighbors = '1000'
                            else:
                                neighbors = h
                            
                            alphas_file = f'output_pl_{h}_alphas_{str(number)}_periodic_{str(periodic)}_{str(n)}_{str(r)}.csv'
                            sigmas_file = f'output_pl_{h}_sigmas_{str(number)}_periodic_{str(periodic)}_{str(n)}_{str(r)}.csv'

                            df_alphas = pd.read_csv(sub_json_path / alphas_file, index_col=0)
                            df_sigmas = pd.read_csv(sub_json_path / sigmas_file, index_col=0)

                            k_target = 50*(n/1000)

                            s_alpha_diff = abs(df_alphas.mean()-k_target)
                            alphas_dict_fifty[str(periodic)][neighbors] = np.round(float(s_alpha_diff.idxmin()), 6)

                            s_sigma_diff = abs(df_sigmas.mean()-k_target)
                            sigmas_dict_fifty[str(periodic)][neighbors] = np.round(float(s_sigma_diff.idxmin()), 3)

                            print(f'Lowest diff r {r} periodic {periodic} for k {neighbors} alpha is {s_alpha_diff.min()} and for sigma is {s_sigma_diff.min()}.')
                    
                    alpha_save_json =  sub_json_path / f'dict_alphas_k{str(int(k_target))}_{r}.json'
                    sigma_save_json =  sub_json_path / f'dict_sigmas_k{str(int(k_target))}_{r}.json'

                    with open(alpha_save_json, 'w') as f:
                        json.dump(alphas_dict_fifty, f)

                    with open(sigma_save_json, 'w') as f:
                        json.dump(sigmas_dict_fifty, f)        