from typing import List, Optional, Tuple
import networkx as nx
import numpy as np
import cdlib.algorithms as cdalgs
import cdlib.evaluation as cdeval
import networkx.algorithms.community as nx_comm
from netneurotools import metrics

import geometric, coordinate


def process_graph(df, graph_file_path):
    # Convert the dataframe to a networkx graph
    G = nx.from_pandas_adjacency(df)
    
    # Check if the graph is connected
    connected_full = nx.is_connected(G)
    print(f"G_full is connected: {connected_full}")
    
    # Get the largest connected component (giant component)
    G_giant = G.subgraph(max(nx.connected_components(G), key=len)).copy()
    
    # Print the number of nodes and edges in the full network and the giant component
    print(f"G_full: Full network has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges, giant component has {G_giant.number_of_nodes()} nodes and {G_giant.number_of_edges()} edges.")
    
    # Save the graph to a GML file
    nx.write_gml(G, graph_file_path)
    
    return G

def remove_unconnected_nodes(adj):
    """
    Remove rows and columns that don't have any connections.
    """
    connected_rows = np.any(adj > 0, axis=1)
    connected_cols = np.any(adj > 0, axis=0)
    return adj[np.ix_(connected_rows, connected_cols)]

def get_density(adj):
    """
    Calculate density of graph. This is given by the equation:
    d = l/n(n-1), 
    for directed networks and 
    d = 2*l/n(n-1)
    for undirected networks.
    From adjacency matrix this simplifies to any connection not equal to zero
    counting as a connection.
    """
    
    if np.any(np.isnan(adj)):
        raise ValueError("The adjacency matrix contains NaN values, stopping execution.")
    elif np.sum(np.trace(adj)) != 0:
        raise ValueError("The adjacency matrix contains self loops, stopping execution.")
    
    edges = np.sum(adj > 0)
    nodes = adj.shape[0]
    
    return edges/(nodes*(nodes-1))

def prune_adj(adj, density_threshold):
    # Calculate node degrees (connectivity)
    """
    Prune the given adjacency matrix to meet the specified density threshold.
    
    Parameters
    ----------
    adj : numpy.ndarray
        The input adjacency matrix.
    density_threshold : float
        The target density threshold (between 0 and 1) to prune the graph to.
    
    Returns
    -------
    numpy.ndarray
        The pruned adjacency matrix.
    """

    nodes = adj.shape[0]
    max_connections = nodes * (nodes - 1)
    min_threshold = ((nodes - 1) * 2) / max_connections
    # if density_threshold < min_threshold:
    #     raise ValueError("Density threshold is too low.")
    
    # Create a copy of the adjacency matrix for pruning
    pruned_adj = adj.copy()

    current_density = get_density(pruned_adj)
    if current_density <= density_threshold:
        return pruned_adj

    iteration = 0
    while True:
        degrees = np.sum(pruned_adj > 0, axis=1)
        # Sort nodes by degree in descending order
        sorted_node_indices = np.argsort(-degrees)

        for node_index in sorted_node_indices:
            # Skip nodes with 2 or fewer connections
            if degrees[node_index] <= 2:
                continue

            # Get indices of non-zero connections for the current node
            connected_indices = np.where(pruned_adj[node_index, :] > 0)[0]

            # Sort the connection weights
            connection_weights = np.sort(pruned_adj[node_index, connected_indices])

            # Prune the weakest connection
            if len(connection_weights) > 0:
                prune_index = np.where(pruned_adj[node_index, :] == connection_weights[0])[0][0]
                if degrees[prune_index] > 2:  # Check if receiving node will have more than 2 connections
                    pruned_adj[node_index, prune_index] = 0
                    pruned_adj[prune_index, node_index] = 0  # Ensure symmetry
                    degrees[node_index] -= 1
                    degrees[prune_index] -= 1
                    # Sort degrees???

            # Check if the current density meets the threshold
            current_density = get_density(pruned_adj)
            if current_density <= density_threshold:
                break
        
        iteration+=1
        if iteration > max_connections:
            return pruned_adj        

    return pruned_adj

def invert_weights(G, weight=None, name="inverse"):
    """
    Invert the weights of all edges in the graph G.
    
    Parameters:
    G (networkx.Graph): The input graph.
    weight (str or None): The edge attribute to invert. If None, no inversion is performed.
    
    Returns:
    networkx.Graph: The graph with inverted edge weights.
    """    
    if weight is not None:
        for u, v, data in G.edges(data=True):
            if weight in data:
                data[name] = 1 / data[weight]
    return G

def convert_weights_to_int(G, factor=100, weight=None, name="intweight"):
    """
    Invert the weights of all edges in the graph G.
    
    Parameters:
    G (networkx.Graph): The input graph.
    weight (str or None): The edge attribute to invert. If None, no inversion is performed.
    
    Returns:
    networkx.Graph: The graph with inverted edge weights.
    """    
    if weight is not None:
        for u, v, data in G.edges(data=True):
            if weight in data:
                data[name] = int(data[weight] * factor)
    return G

def diff_binary_nets(G1, G2):
    """ Finds the dustance and bins these for alle connections in the network.
    
    Parameters
    ----------
    G : Graph
        Networkx graph object containing spatial coordinates in a pos value.
    Bins : 1-d array
        Bins.
    Returns
    -------
    """

    a1 = nx.to_numpy_array(G1)
    a2 = nx.to_numpy_array(G2)
    diff = a1 - a2
    return diff

def connection_ranges(G, domain=(0, 0, 1, 1), periodic=False):
    """ Finds the distance and bins these for alle connections in the network.
    G must contain position as "pos" in node data.

    Parameters
    ----------
    G : Graph
        Networkx graph object containing spatial coordinates in a pos value.
    Bins : 1-d array
        Bins.
    Returns
    -------
    """
    
    pos = dict(G.nodes(data="pos"))

    dists = coordinate.periodic_dist(np.asarray(
        [*pos.values()]), domain=domain, periodic=periodic)
    a = nx.to_numpy_array(G)

    d = dists[a.astype(bool)]

    return list(d.flatten())


def log_bin(x, n=10, min=0, max=5, ):
    """
    Log-binning
    changelog:
    geomspace from logspace
    removed n per decade.
    """
    # n = max*n+1

    bins = np.geomspace(min, max, n, endpoint=True)
    width = bins[1:] - bins[:-1]

    hist = np.histogram(x, bins=bins)
    hist_norm = hist[0]/width
    hist_norm = hist_norm/sum(hist_norm)
    print(hist_norm, bins, width, sum(width))
    return hist_norm, bins, width


def log_hist(ax, x, n=10, min=0, max=5, log=True):
    """
    Log-binning

    n is number of bins per decade (10e (min to max))
    """

    hist_norm, bins, width = log_bin(x, n=n, min=min, max=max)
    ax.bar(bins[:-1], hist_norm, width*0.8)

    if log == True:
        ax.set_xscale('log')
        ax.set_yscale('log')

    return hist_norm, bins, width

# def swi(G, niter=5, nrand=10, seed=None):
#     """ Calculates the small world index (SWI) of input graph.

#     Parameters
#     ----------

#     Returns
#     -------

#     """


def mean_degree(G):
    """ Simple function to describe input graph G. By definition, 
    <k> = 2*E/N
    E - Edges (Links)
    N - Nodes (Vertices)
    """

    k_mean =  2*len(G.edges)/len(G.nodes)
    
    return k_mean


def median_degree(G):
    """ Simple function to describe input graph G.
    """

    degrees = np.array([d for n, d in G.degree()])
    k_med = np.median(degrees)    
    return k_med

def mean_edge_weight(G, weight='weight'):
    """ Calculate the mean weight of the edges in the graph G. """
    total_weight = sum(data[weight] for u, v, data in G.edges(data=True))
    mean_weight = total_weight / len(G.edges)
    return mean_weight

def median_edge_weight(G, weight='weight'):
    """ Calculate the median weight of the edges in the graph G. """
    weights = [data[weight] for u, v, data in G.edges(data=True)]
    median_weight = np.median(weights)
    return median_weight

# def mean_degree(G, weighted=False):
#     """ Calculate the mean degree of the graph G.
#     If weighted is True, calculate the mean weighted degree.
#     """
#     if weighted:
#         k_mean = np.mean([d for n, d in G.degree(weight='weight')])
#     else:
#         k_mean = 2 * len(G.edges) / len(G.nodes)
#     return k_mean

# def median_degree(G, weighted=False):
#     """ Calculate the median degree of the graph G.
#     If weighted is True, calculate the median weighted degree.
#     """
#     if weighted:
#         degrees = np.array([d for n, d in G.degree(weight='weight')])
#     else:
#         degrees = np.array([d for n, d in G.degree()])
#     k_med = np.median(degrees)
#     return k_med

# def mean_in_out_degree(G, weighted=False):
#     """ Calculate the mean in-degree and out-degree for directed graphs.
#     If weighted is True, calculate the mean weighted in-degree and out-degree.
#     """
#     if weighted:
#         k_mean_in = np.mean([d for n, d in G.in_degree(weight='weight')])
#         k_mean_out = np.mean([d for n, d in G.out_degree(weight='weight')])
#     else:
#         k_mean_in = np.mean([d for n, d in G.in_degree()])
#         k_mean_out = np.mean([d for n, d in G.out_degree()])
#     return k_mean_in, k_mean_out

# def median_in_out_degree(G, weighted=False):
#     """ Calculate the median in-degree and out-degree for directed graphs.
#     If weighted is True, calculate the median weighted in-degree and out-degree.
#     """
#     if weighted:
#         in_degrees = np.array([d for n, d in G.in_degree(weight='weight')])
#         out_degrees = np.array([d for n, d in G.out_degree(weight='weight')])
#     else:
#         in_degrees = np.array([d for n, d in G.in_degree()])
#         out_degrees = np.array([d for n, d in G.out_degree()])
#     k_med_in = np.median(in_degrees)
#     k_med_out = np.median(out_degrees)
#     return k_med_in, k_med_out

def betweenness_metrics(G, weight=None):
    """
    Calculate the mean, median, maximum, and minimum betweenness centrality values for the graph G.
    
    Parameters:
    G (networkx.Graph): The input graph.
    
    Returns:
    tuple: A tuple containing the mean, median, maximum, and minimum betweenness centrality values.
    """
    bc = nx.betweenness_centrality(G, weight=weight)

    bc_mean = np.mean(list(bc.values()))
    bc_med = np.median(list(bc.values()))
    bc_max = np.max(list(bc.values()))
    bc_min = np.min(list(bc.values()))

    return bc_mean, bc_med, bc_max, bc_min


def mean_rich_club_coefficient(G, normalized=False, seed=None):
    """
    Note: may need updating in future if networkx is updated.
    """
    rc = nx.rich_club_coefficient(G, normalized=normalized, seed=seed)

    rc_mean = np.mean(list(rc.values()))
    rc_med = np.median(list(rc.values()))
    rc_max = np.max(list(rc.values()))
    rc_min = np.min(list(rc.values()))

    return rc_mean, rc_med, rc_max, rc_min

def closeness_metrics(G, distance=None):
    """
    Calculate the mean, median, maximum, and minimum closeness centrality values for the graph G.
    
    Parameters:
    G (networkx.Graph): The input graph.
    distance (str or None): The edge attribute to use as distance. If None, use unweighted closeness centrality.
    
    Returns:
    tuple: A tuple containing the mean, median, maximum, and minimum closeness centrality values.
    """
    cc = nx.closeness_centrality(G, distance=distance)

    cc_mean = np.mean(list(cc.values()))
    cc_med = np.median(list(cc.values()))
    cc_max = np.max(list(cc.values()))
    cc_min = np.min(list(cc.values()))

    return cc_mean, cc_med, cc_max, cc_min

def communicability_metrics(G):
    """
    Calculate the mean, median, maximum, and minimum communicability values for the graph G.
    
    Parameters:
    G (networkx.Graph): The input graph.
    
    Returns:
    tuple: A tuple containing the mean, median, maximum, and minimum communicability values.
    """
    communicability_dict = nx.communicability(G)

    communicability_values = []
    for node, comm_dict in communicability_dict.items():
        for target_node, value in comm_dict.items():
            communicability_values.append(value)

    comm_mean = np.mean(communicability_values)
    comm_med = np.median(communicability_values)
    comm_max = np.max(communicability_values)
    comm_min = np.min(communicability_values)

    return comm_mean, comm_med, comm_max, comm_min


def shortest_communicability_path(G, normalize=False):
    """
    Computes the shortest path communicability of a graph.

    This function determines whether the adjacency matrix of the given graph is binary or weighted.
    It then computes the shortest path communicability using the appropriate method from the 
    netneurotools library. The function returns the lowest value in the resulting communicability 
    matrix, with the diagonal set to zero.

    Parameters:
    -----------
    G : networkx.Graph
                    The input graph for which to compute the shortest path communicability.
    normalize : bool, optional
                    If True, the communicability matrix is normalized. Default is False.

    Returns:
    --------
    float
                    The lowest value in the communicability matrix, with the diagonal set to zero.

    Example:
    --------
    >>> G = nx.erdos_renyi_graph(10, 0.5)
    >>> result = shortest_path_communicability(G)
    >>> print(result)
    """

    adj = nx.to_numpy_array(G)

    if np.all((adj == 0) | (adj == 1)):
        communicability = metrics.communicability_bin(adj, normalize=normalize)
    else:
        communicability = metrics.communicability_wei(adj, normalize=normalize)

    if np.iscomplexobj(communicability):
        communicability = np.real(communicability)

    # np.fill_diagonal(communicability, 0)
    mask_off_diag = ~np.eye(communicability.shape[0], dtype=bool)
    off_diag_values = communicability[mask_off_diag]

    # 2. Calculate metrics on the combined data
    mean_communicability = np.mean(off_diag_values)
    min_communicability = np.min(off_diag_values)
    # umean_communicability = np.mean(communicability[np.triu_indices_from(communicability, k=1)])
    # lmean_communicability = np.mean(communicability[np.tril_indices_from(communicability, k=-1)])
    # min_communicability = min(min(communicability[np.triu_indices_from(communicability, k=1)]), min(communicability[np.tril_indices_from(communicability, k=-1)]))

    return mean_communicability, min_communicability

def mean_first_passage_time_metrics(G, tol=0.001):
    """
    Computes the mean first passage time

    Parameters:
    -----------
    G : networkx.Graph
                    The input graph for which to compute the shortest path communicability.
    tol : bool, optional
                    value, default 0.001

    Returns:
    --------
    float
                    The lowest value in the communicability matrix, with the diagonal set to zero.

    Example:
    --------
    >>> G = nx.erdos_renyi_graph(10, 0.5)
    >>> result = shortest_path_communicability(G)
    >>> print(result)
    """

    adj = nx.to_numpy_array(G)
    
    mfpt = metrics.mean_first_passage_time(adj, tol=0.001)
    
    min_mfpt = mfpt[mfpt.nonzero()].min()

    umean_mfpt = np.mean(mfpt[np.triu_indices_from(mfpt, k=1)])
    lmean_mfpt = np.mean(mfpt[np.tril_indices_from(mfpt, k=-1)])

    return min_mfpt, umean_mfpt, lmean_mfpt

def hierarchical_modularity(G):
    """ Measures the hierarchical modularity (finds C(k))
    
    """
    # get degrees and 
    dict = {}
    G.degree
    return dict


def neuron_n_nearest_neighbors(pos, n, domain=(0, 0, 1, 1), periodic=False):
    """
    Compute Euclidean distances to the n nearest neighbors for each neuron.

    Parameters
    ----------
    pos : ndarray
        An array of shape (N, D) representing the positions of N neurons in D-dimensional space.
    n : int
        The number of nearest neighbors to consider for each neuron.
    domain : tuple, optional
        The spatial domain boundaries used for periodic distance calculation.
        Format depends on dimensionality, e.g., (xmin, ymin, xmax, ymax) for 2D.
    periodic : bool, optional
        If True, apply periodic boundary conditions when computing distances.

    Returns
    -------
    list of float
        A flattened list of Euclidean distances to the n nearest neighbors for all neurons.
    """

    dists = coordinate.periodic_dist(pos, domain, periodic=periodic)

    sort_dists = np.sort(dists, axis=1)

    n_neighbors = sort_dists[:, 1:(n+1)]

    return list(n_neighbors.flatten())


# def neuron_n_nearest_neighbors_reviewed(
#     pos: np.ndarray,
#     n: int,
#     domain: Optional[Tuple[float, ...]] = None,
#     periodic: bool = False,
# ) -> List[float]:  # Added return type hint
#     """
#     Compute Euclidean distances to the n nearest neighbors for each neuron.

#     Parameters
#     ----------
#     pos : ndarray
#         An array of shape (N, D) representing the positions of N neurons.
#     n : int
#         The number of nearest neighbors to consider.
#     domain : tuple, optional
#         The spatial domain boundaries. Required if periodic=True.
#     periodic : bool, optional
#         If True, apply periodic boundary conditions.

#     Returns
#     -------
#     list of float
#         A flattened list of Euclidean distances to the n nearest neighbors.
#     """

#     N = pos.shape[0]

#     # Handle case with 0 or 1 neuron
#     if N <= 1:
#         return []

#     # --- Address Edge Case ---
#     # We can find at most N-1 neighbors.
#     # k is the number of neighbors we will actually find.
#     k = min(n, N - 1)

#     # If k is 0 (e.g., N=1 or n=0), return empty
#     if k == 0:
#         return []

#     # Calculate the full (N, N) distance matrix
#     dists = coordinate.periodic_dist(pos, domain, periodic=periodic)

#     # --- Efficiency Improvement ---
#     # Use np.partition to find the k+1 smallest distances (k neighbors + 1 self-distance)
#     # This is much faster than sorting the entire row.
#     # We partition by the k-th index (which is the (k+1)-th element)
#     partitioned_dists = np.partition(dists, kth=k, axis=1)

#     # We only care about the k+1 smallest elements (indices 0 to k)
#     smallest_k_plus_1 = partitioned_dists[:, 0 : (k + 1)]

#     # We must sort these k+1 elements to ensure index 0 is the self-distance (0.0)
#     sorted_smallest = np.sort(smallest_k_plus_1, axis=1)

#     # Get the distances, skipping the first element (self-distance)
#     # This gives exactly k neighbors
#     n_neighbors = sorted_smallest[:, 1 : (k + 1)]

#     return list(n_neighbors.flatten())


def hist_list(a, bins=10, range=None, **kwargs):
    """ Function to go through list of values. gives hist and error for all 
    arrays in array
    
    """
    hists = []
    bin_edges = []

    for sub_a in a:
        hist, bin_edge = np.histogram(sub_a, bins=bins, range=None, **kwargs)
        hists.append(hist)
        bin_edges.append(bin_edge)

    return hists, bin_edges

def mean_error(a, axis=0):
    """ Calculates the mean and standard error of the mean along the given axis.
    
    Parameters
    ----------
    a : array_like
        array to calculate mean and eom for
    axis : None or int or tuple of ints, optional
        Axis or axes along which the standard deviation is computed. The 
        default is to compute the standard deviation of the flattened array.
    Returns
    -------
    a_mean : 
    a_eom :
    """
    
    a = np.asarray(a)
    a_mean = np.mean(a, axis=axis)
    a_eom = np.std(a, axis=axis, ddof=1) /np.sqrt(a.shape[axis])

    return a_mean, a_eom


def optimal_markov_clustering(G, start=1.1, stop=2.5, num=15, q_res=1.0, **kwargs):
    """ Optimized markov clustering. See [1] for reference (available under MIT
    License).

    Parameters
    ----------

    Returns
    -------
    ..[1] https://github.com/GuyAllard/markov_clustering
    """

    inflations = np.around(np.linspace(start=start, stop=stop, num=num), 3)
    
    qs = []
    zs = []

    for inflation in inflations:
        coms = cdalgs.markov_clustering(G, inflation=inflation)

        qs.append(nx_comm.modularity(G, coms.communities, resolution=q_res))
        zs.append(cdeval.z_modularity(G, coms).score)
    
    q_i = inflations[qs.index(max(qs))]
    coms = cdalgs.markov_clustering(G, inflation=q_i)
    q_coms = len(coms.communities)
    q_mean_size = np.mean([len(c) for c in coms.communities])
    q_median_size = np.median([len(c) for c in coms.communities])

    z_i = inflations[zs.index(max(zs))]
    coms = cdalgs.markov_clustering(G, inflation=z_i)
    z_coms = len(coms.communities)

    z_mean_size = np.mean([len(c) for c in coms.communities])
    z_median_size = np.median([len(c) for c in coms.communities])

    return max(qs), q_i, q_coms, q_mean_size, q_median_size, max(zs), z_i, z_coms, z_mean_size, z_median_size

def optimal_qlouvain_communities_single_pass(G, start=1.0, stop=4.0, num=31, q_res=1.0, base_repeat=5, repeats=50, **kwargs):
    """ Optimized louvain communities based on modularity.

    Parameters
    ----------

    Returns
    -------

    """

    resolutions = np.round(np.linspace(start=start, stop=stop, num=num), 4)
    
    qs = []

    for resolution in resolutions:
        eval_q = []
        for i in range(base_repeat):
            coms = cdalgs.louvain(G, resolution=resolution)
            eval_q.append(nx_comm.modularity(G, coms.communities, resolution=q_res))
        qs.append(np.mean(eval_q))

    q_r = resolutions[qs.index(max(qs))]
    q = []
    q_coms = []
    q_mean_size = []
    q_median_size = []

    for i in range(repeats):
        coms = cdalgs.louvain(G, resolution=q_r)
        q.append(nx_comm.modularity(G, coms.communities, resolution=q_res))
        q_coms.append(len(coms.communities))
        q_mean_size.append(np.mean([len(c) for c in coms.communities]))
        q_median_size.append(np.median([len(c) for c in coms.communities]))
    print(np.mean(q), np.std(q))

    return np.mean(q), q_r, np.mean(q_coms), np.mean(q_mean_size), np.mean(q_median_size)


def optimal_qlouvain_communities_double_pass(G, start=1.0, stop=4.0, first_increment=0.5, second_range=5, second_increment=0.05, q_res=1.0, base_repeat=5, repeats=50, **kwargs):
    """ Optimized louvain communities based on modularity.

    Parameters
    ----------

    Returns
    -------

    """
    num = int((stop-start)/first_increment) + 1
    resolutions = np.round(np.linspace(start=start, stop=stop, num=num), 4)
    qs = []
    for resolution in resolutions:
        eval_q = []
        for i in range(base_repeat):
            coms = cdalgs.louvain(G, resolution=resolution)
            eval_q.append(nx_comm.modularity(G, coms.communities, resolution=q_res))
        qs.append(np.mean(eval_q))

    q_r = resolutions[qs.index(max(qs))]
    second_start = max(q_r - second_range, 0.5)
    second_stop = q_r + second_range 
    second_num = int((second_stop-second_start)/second_increment) + 1
    resolutions = np.round(np.linspace(start=second_start, stop=second_stop, num=second_num), 4)

    qs = []
    for resolution in resolutions:
        eval_q = []
        for i in range(base_repeat):
            coms = cdalgs.louvain(G, resolution=resolution)
            eval_q.append(nx_comm.modularity(G, coms.communities, resolution=q_res))
        qs.append(np.mean(eval_q))

    q_r = resolutions[qs.index(max(qs))]

    q = []
    q_coms = []
    q_mean_size = []
    q_median_size = []

    for i in range(repeats):
        coms = cdalgs.louvain(G, resolution=q_r)
        q.append(nx_comm.modularity(G, coms.communities, resolution=q_res))
        q_coms.append(len(coms.communities))
        q_mean_size.append(np.mean([len(c) for c in coms.communities]))
        q_median_size.append(np.median([len(c) for c in coms.communities]))

    return np.mean(q), q_r, np.mean(q_coms), np.mean(q_mean_size), np.mean(q_median_size)

def optimal_zlouvain_communities(G, start=1.0, stop=4.0, num=31, base_repeat=5, repeats=50, **kwargs):
    """ Optimized louvain communities based on modularity. 

    Parameters
    ----------

    Returns
    -------

    """
    resolutions = np.round(np.linspace(start=start, stop=stop, num=num), 4)

    zs = []

    for resolution in resolutions:
        eval_z = []
        for i in range(base_repeat):
            coms = cdalgs.louvain(G, resolution=resolution)
            eval_z.append(cdeval.z_modularity(G, coms).score)
        zs.append(np.mean(eval_z))

    z_r = resolutions[zs.index(max(zs))]
    z = []
    z_coms = []
    z_mean_size = []
    z_median_size = []

    for i in range(repeats):
        coms = cdalgs.louvain(G, resolution=z_r)
        z.append(cdeval.z_modularity(G, coms).score)
        z_coms.append(len(coms.communities))
        z_mean_size.append(np.mean([len(c) for c in coms.communities]))
        z_median_size.append(np.median([len(c) for c in coms.communities]))
    print(np.mean(z), np.std(z))
    return np.mean(z), z_r, np.mean(z_coms), np.mean(z_mean_size), np.mean(z_median_size)

def optimal_infomap(G, q_res=1.0, **kwargs):
    """ Optimized Infomap communities based on modularity. 

    Parameters
    ----------

    Returns
    -------

    """

    coms = cdalgs.infomap(G)
    q = nx_comm.modularity(G, coms.communities, resolution=q_res)
    z = cdeval.z_modularity(G, coms).score
    coms_len = len(coms.communities)
    mean_size = np.mean([len(c) for c in coms.communities])
    median_size = np.median([len(c) for c in coms.communities])

    return q, z, coms_len, mean_size, median_size