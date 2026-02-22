import numpy as np
import analysis
import networkx as nx
from networkx.utils import not_implemented_for, py_random_state

__all__ = ["random_reference", "lattice_reference", "sigma", "omega"]

def compare_graph_weights(G1, G2, weight=None):
    """ Compare the weights of two graphs G1 and G2 to see if they are the same. """
    if set(G1.edges) != set(G2.edges):
        return False
    
    for u, v in G1.edges:
        if G1[u][v].get(weight, 1) != G2[u][v].get(weight, 1):
            return False
    return True

def omega(G, niter=5, nrand=10, seed=None):
    """Returns the small-world coefficient (omega) of a graph

    The small-world coefficient of a graph G is:

    omega = Lr/L - C/Cl

    where C and L are respectively the average clustering coefficient and
    average shortest path length of G. Lr is the average shortest path length
    of an equivalent random graph and Cl is the average clustering coefficient
    of an equivalent lattice graph.

    The small-world coefficient (omega) measures how much G is like a lattice
    or a random graph. Negative values mean G is similar to a lattice whereas
    positive values mean G is a random graph.
    Values close to 0 mean that G has small-world characteristics.

    Parameters
    ----------
    G : NetworkX graph
        An undirected graph.

    niter: integer (optional, default=5)
        Approximate number of rewiring per edge to compute the equivalent
        random graph.

    nrand: integer (optional, default=10)
        Number of random graphs generated to compute the maximal clustering
        coefficient (Cr) and average shortest path length (Lr).

    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.


    Returns
    -------
    omega : float
        The small-world coefficient (omega)

    Notes
    -----
    The implementation is adapted from the algorithm by Telesford et al. [1]_.

    References
    ----------
    .. [1] Telesford, Joyce, Hayasaka, Burdette, and Laurienti (2011).
           "The Ubiquity of Small-World Networks".
           Brain Connectivity. 1 (0038): 367-75.  PMC 3604768. PMID 22432451.
           doi:10.1089/brain.2011.0038.
    """
    randMetrics = {"C": [], "L": []}

    Cl = nx.average_clustering(G)
 
    niter_lattice_reference = niter
    niter_random_reference = niter * 2

    for _ in range(nrand):
        # Generate random graph
        Gr = nx.random_reference(G, niter=niter_random_reference, seed=seed)
        randMetrics["L"].append(nx.average_shortest_path_length(Gr))

        # Generate lattice graph
        Gl = nx.lattice_reference(G, niter=niter_lattice_reference, seed=seed)

        # Replace old clustering coefficient, if clustering is higher in
        # generated lattice reference
        Cl_temp = nx.average_clustering(Gl)
        if Cl_temp > Cl:
            Cl = Cl_temp

    C = nx.average_clustering(G)
    L = nx.average_shortest_path_length(G)
    Lr = np.mean(randMetrics["L"])

    omega = (Lr / L) - (C / Cl)

    return omega

def small_world_propensity_generated_equivalents(G, nrand=100, seed=None):
    """ Calculates the small world propensity (SWP) of input graph. uses equivalent
    rendom graph. Based on NetworkX omega.
    
    random_reference(G, niter=1, connectivity=True, seed=None)
    lattice_reference(G, niter=5, D=None, connectivity=True, seed=None)

    Parameters
    ----------

    Returns

    -------
    References
    ----------
    .. [1] Muldoon, S. F. et al. Small-World Propensity and Weighted Brain 
           Networks. Sci. Rep.6, 22057; doi: 10.1038/srep22057 (2016).
    """
    n = len(G.nodes)
    m = len(G.edges)

    rand_ref = {"cr": [], "lr": []}
    k = 2 * m / n

    for _ in range(nrand):
        rand_G = nx.gnm_random_graph(n, m, seed=seed)
        while not nx.is_connected(rand_G):
            rand_G = nx.gnm_random_graph(n, m, seed=seed)

        rand_ref['cr'].append(nx.average_clustering(rand_G))
        rand_ref['lr'].append(nx.average_shortest_path_length(rand_G))
    
    latt_G = nx.connected_watts_strogatz_graph(n, int(np.ceil(k)), 0.0, seed=seed)

    c = nx.average_clustering(G)
    l = nx.average_shortest_path_length(G)

    cl = nx.average_clustering(latt_G)
    ll = nx.average_shortest_path_length(latt_G)

    cr = np.mean(rand_ref['cr'])
    lr = np.mean(rand_ref['lr'])

    del_c = (cl - c) / (cl - cr)
    del_l = (l - lr) / (ll - lr)
    
    if del_c > 1.0:
        del_c = 1.0
    if del_l > 1.0:
        del_l = 1.0

    if del_c < 0.0:
        del_c = 0.0
    if del_l < 0.0:
        del_l = 0.0

    phi = 1 - np.sqrt((del_c**2 + del_l**2) / 2)

    omega = (lr / l) - (c / cl)
    omega_bound = 1 - abs(omega)

    sigma = (c / cr) / (l / lr)

    return phi, omega, omega_bound, sigma

def small_world_propensity(G, weight=None, niter=5, nrand=10, seed=None):
    """ Calculates the small world propensity (SWP) of input graph. 
    Uses equivalent random graph generated by rewiring the original graph. 
    Based on NetworkX omega.
    
    Given by ...

    Where ...
    random_reference(G, niter=1, connectivity=True, seed=None)
    lattice_reference(G, niter=5, D=None, connectivity=True, seed=None)

    Parameters
    ----------
    G : NetworkX graph
        An undirected graph.

    niter: integer (optional, default=5)
        Approximate number of rewiring per edge to compute the equivalent
        random graph.

    nrand: integer (optional, default=10)
        Number of random graphs generated to compute the maximal clustering
        coefficient (Cr) and average shortest path length (Lr).

    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.


    Returns
    -------
    pni : float
        The small-world propensity (phi)

    Notes
    -----
    The implementation is adapted from the algorithm by Muldoon et al. [1]_.

    References
    ----------
    .. [1] Muldoon, S. F. et al. Small-World Propensity and Weighted Brain 
           Networks. Sci. Rep.6, 22057; doi: 10.1038/srep22057 (2016).
    """
    Cl = nx.average_clustering(G, weight=weight)
    Ll = nx.average_shortest_path_length(G, weight=weight)

    randomMetrics = {"Cr": [], "Lr": []}

    niter_lattice_reference = niter
    niter_random_reference = niter * 2

    for _ in range(nrand):
        # Generate random graph
        Gr = nx.random_reference(G, niter=niter_random_reference, seed=seed)
        # randMetrics["L"].append(nx.average_shortest_path_length(Gr))
        randomMetrics['Cr'].append(nx.average_clustering(Gr, weight=weight))
        randomMetrics['Lr'].append(nx.average_shortest_path_length(Gr, weight=weight))

        # Generate lattice graph
        Gl = nx.lattice_reference(G, niter=niter_lattice_reference, seed=seed)

        # Replace old clustering coefficient and path length, if 
        # clustering is higher in generated lattice reference
        Cl_temp = nx.average_clustering(Gl, weight=weight)
        if Cl_temp > Cl:
            Cl = Cl_temp
            Ll = nx.average_shortest_path_length(Gl, weight=weight)

    Cr = np.mean(randomMetrics['Cr'])
    Lr = np.mean(randomMetrics['Lr'])

    C = nx.average_clustering(G, weight=weight)
    L = nx.average_shortest_path_length(G, weight=weight)

    del_c = (Cl - C) / (Cl - Cr)
    del_l = (L - Lr) / (Ll - Lr)
    
    if del_c > 1.0:
        del_c = 1.0
    if del_l > 1.0:
        del_l = 1.0

    if del_c < 0.0:
        del_c = 0.0
    if del_l < 0.0:
        del_l = 0.0

    phi = 1 - np.sqrt((del_c**2 + del_l**2) / 2)

    omega = (Lr / L) - (C / Cl)
    omage_bound = 1 - abs(omega)

    sigma = (C / Cr) / (L / Lr)

    return phi, omega, omage_bound, sigma


def small_world_propensity_approximate(G):
    """ Calculates the small world propensity (SWP) of input graph. uses equivalent
    rendom graph. Based on NetworkX omega.

    Parameters
    ----------

    Returns
    -------
    
    References
    ----------
    .. [1] Muldoon, S. F. et al. Small-World Propensity and Weighted Brain 
           Networks. Sci. Rep.6, 22057; doi: 10.1038/srep22057 (2016).
    """
    c = nx.average_clustering(G)
    l = nx.average_shortest_path_length(G)

    n = len(G)
    m = len(G.edges())

    k = 2 * m / n

    cr = k / n
    lr = np.log(n) / np.log(k)
    
    ll = n / (2*k)
    cl = 3/4 * (k-2) / (k-1)

    del_c = (cl - c) / (cl - cr)
    del_l = (l - lr) / (ll - lr)

    if del_c > 1.0:
        del_c = 1.0
    if del_l > 1.0:
        del_l = 1.0

    if del_c < 0.0:
        del_c = 0.0
    if del_l < 0.0:
        del_l = 0.0

    phi = 1 - np.sqrt((del_c**2 + del_l**2) / 2)
    
    omega = (lr / l) - (c / cl)
    omage_bound = 1 - abs(omega)

    sigma = (c / cr) / (l / lr)

    return phi, omega, omage_bound, sigma


def bound_omega(G, niter=5, nrand=10, seed=None):
    """ Calculates the normalized small world index between 0 and 1. Higher 
    values indicate more small-world. 
    
    References
    ----------
    .. [1] Neal, Zachary P. *How small is it? Comparing indices of small 
    worldliness*. Network Science 5.1 (2017): 30-44.
    """

    omega_dash = 1 - abs(nx.omega(G, niter=niter, nrand=nrand, seed=seed))

    return omega_dash