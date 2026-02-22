import numpy as np
import networkx as nx
from typing import List, Tuple, Dict, Any, Optional

import coordinate

def get_all_edge_distances(
    G: nx.Graph, domain: Tuple[float, ...] = (0, 0, 1, 1), periodic: bool = True
) -> np.ndarray:
    """
    Finds the distances for all existing connections (edges) in the network.

    This function correctly handles both directed and undirected graphs.
    - For Undirected: Counts each edge (i, j) only once.
    - For Directed: Counts each edge (i, j) and (j, i) as separate edges.

    Parameters
    ----------
    G : networkx.Graph or networkx.DiGraph
        Networkx graph object containing spatial coordinates in a "pos" attribute.
    domain : tuple, optional
        The spatial domain boundaries for periodic distance calculation.
    periodic : bool, optional
        If True, apply periodic boundary conditions.

    Returns
    -------
    np.ndarray
        A 1D NumPy array of Euclidean distances for all relevant edges.
    """

    # Explicitly define the node order to guarantee alignment
    nodelist = list(G.nodes())

    if len(nodelist) < 2:
        return np.array([])  # No edges possible if < 2 nodes

    # Get node positions as a NumPy array in the correct order
    pos_dict = nx.get_node_attributes(G, "pos")

    # Ensure all nodes in nodelist have a position
    if len(pos_dict) != len(nodelist):
        missing = [n for n in nodelist if n not in pos_dict]
        raise ValueError(f"Missing 'pos' attribute for nodes: {missing}")

    pos_array = np.array([pos_dict[n] for n in nodelist])

    # Calculate the full (N, N) pairwise distance matrix
    dists = coordinate.periodic_dist(pos_array, domain=domain, periodic=periodic)

    # Get the (N, N) adjacency matrix, using the same nodelist
    a = nx.to_numpy_array(G, nodelist=nodelist)

    # --- This logic handles both graph types ---
    if G.is_directed():
        # For a DIRECTED graph, we use the full matrix.
        # An edge (i, j) is different from (j, i).
        mask = a.astype(bool)
    else:
        # For an UNDIRECTED graph, we use the upper triangle.
        # This counts each edge only ONCE.
        mask = np.triu(a.astype(bool))

    # Select only the distances where an edge exists
    edge_distances = dists[mask]

    return edge_distances


def calculate_network_cost_stats(
    G: nx.Graph, domain: Tuple[float, ...] = (0, 0, 1, 1), periodic: bool = True
) -> Dict[str, Any]:
    """
    Calculates the total wiring cost and normalized cost statistics.

    This function is the recommended way to analyze and compare graphs.

    Parameters
    ----------
    G : networkx.Graph or networkx.DiGraph
        Networkx graph object.
    domain : tuple, optional
        The spatial domain boundaries.
    periodic : bool, optional
        If True, apply periodic boundary conditions.

    Returns
    -------
    Dict[str, Any]
        A dictionary containing:
        - 'total_cost': The sum of all unique edge distances.
        - 'num_edges': The total number of unique edges.
        - 'average_cost_per_edge': The 'total_cost' / 'num_edges'.
    """

    # 1. Get the list of all unique edge distances
    edge_distances = get_all_edge_distances(G, domain, periodic)

    # 2. Calculate "Cost of Wiring"
    total_cost = np.sum(edge_distances)

    # 3. Get the number of edges
    num_edges = len(edge_distances)

    # 4. Calculate Normalization (Average Cost per Edge)
    # This is the most direct way to compare networks of different densities.
    average_cost_per_edge = 0.0
    if num_edges > 0:
        average_cost_per_edge = total_cost / num_edges

    return {
        "total_cost": total_cost,
        "num_edges": num_edges,
        "average_cost_per_edge": average_cost_per_edge,
    }


# --- --- --- --- --- --- --- --- --- ---
# ---           EXAMPLE USAGE         ---
# --- --- --- --- --- --- --- --- --- ---
if __name__ == "__main__":
    # --- 1. Create an UNDIRECTED graph ---
    G_undirected = nx.Graph()
    # Add nodes with positions
    G_undirected.add_node(0, pos=(0.1, 0.1))
    G_undirected.add_node(1, pos=(0.9, 0.1))
    G_undirected.add_node(2, pos=(0.5, 0.8))
    # Add edges
    G_undirected.add_edges_from([(0, 1), (0, 2)])  # Two edges

    stats_undirected = calculate_network_cost_stats(G_undirected)

    print("--- UNDIRECTED GRAPH ---")
    print(f"Stats: {stats_undirected}")
    # Expected:
    # num_edges = 2
    # dist(0,1) = 0.8
    # dist(0,2) = sqrt(0.4^2 + 0.7^2) = 0.806
    # total_cost = 0.8 + 0.806 = 1.606
    # average_cost_per_edge = 1.606 / 2 = 0.803

    # --- 2. Create a DIRECTED graph ---
    G_directed = nx.DiGraph()
    # Add same nodes
    G_directed.add_node(0, pos=(0.1, 0.1))
    G_directed.add_node(1, pos=(0.9, 0.1))
    G_directed.add_node(2, pos=(0.5, 0.8))
    # Add edges: one uni-directional, one bi-directional
    G_directed.add_edge(0, 1)  # Edge 0 -> 1
    G_directed.add_edge(0, 2)  # Edge 0 -> 2
    # G_directed.add_edge(2, 0)  # Edge 2 -> 0 (different edge!)

    stats_directed = calculate_network_cost_stats(G_directed)

    print("\n--- DIRECTED GRAPH ---")
    print(f"Stats: {stats_directed}")
    # Expected:
    # num_edges = 3
    # dist(0,1) = 0.8
    # dist(0,2) = 0.806
    # dist(2,0) = 0.806
    # total_cost = 0.8 + 0.806 + 0.806 = 2.412
    # average_cost_per_edge = 2.412 / 3 = 0.804
