"""
Module containing overlap checks as well as routines for creating specific
placement patterns of neurons.

List of functions
-----------------



See also:
https://www.researchgate.net/post/can_anyone_help_me_for_handling_random_non-overlapping_circles_using_c
"""

import sys

import numpy as np
import pandas as pd
import scipy as sp

import stats


def box_to_domain(box):
    """ Create domain from box for overlap function.
    """
    lower_domain = list(np.zeros(len(box), dtype=int))
    upper_domain = list(box)
    
    domain = tuple(lower_domain + upper_domain)
    
    return domain

def periodic_dist(pos, domain=(0, 0, 1, 1), periodic=False):
    """Function to get minimal distance between all positions assuming a 
    periodic boundary. Takes a domain of n dimension. Positions are adjusted to
    start from zero, of the domain is shifted.

    Parameters
    ----------
    pos : ndarray
        Array of arrays containing positions to check. Minimum 2D positions.
    box : array-like
        Box size for modulo. Container restricting positions
    Returns:
    --------
    dists : float
        Distances between all position pairs in the periodic box.
    """
    if pos.shape[0] < 2:
        return np.array([]).reshape(0, 0)

    if periodic:
        pos = pos - domain[:int(len(domain)/2)]

        box = [y-x for x, y in zip(domain[:len(domain)//2],
                            domain[len(domain)//2:])]
        half_box = np.array(box)/2
        dists = sp.spatial.distance.cdist(pos, pos)
        shift_pos = pos + half_box
        shift_pos = shift_pos % box
        dists_shift = sp.spatial.distance.cdist(shift_pos, shift_pos)
        dists = np.minimum(dists, dists_shift)
        for dim, half_box_dir in enumerate(half_box):
            half_box_shift = np.zeros(len(half_box))
            half_box_shift[dim] = half_box_dir
            shift_pos = pos + half_box_shift
            shift_pos = shift_pos % box
            dists_shift = sp.spatial.distance.cdist(shift_pos, shift_pos)
            dists = np.minimum(dists, dists_shift)
    else:
        dists = sp.spatial.distance.cdist(pos, pos)

    return dists

def overlap(pos, r, box, periodic=False):
    """ Checks for overlapping cell bodies in initial placement and move any
    overlapping cells to new positions. Returns index pair for overlapping 
    positions.

    Parameters
    ----------
    pos : float or array-like, shape (n,m)
        Position of neuron(s)
    r : float
        Minimum radius for cell body of neuron.
    box : array-like, shape 1xN
        Dimensions of space for neurons to occupy.
    Returns
    -------
    overlaps : array-like, shape (n,2)
        Id pairs of overlapping position pairs based on 2 * r.
    """

    domain = box_to_domain(box)

    d = periodic_dist(pos, domain=domain, periodic=periodic)
    overlaps = np.transpose(np.nonzero((np.triu(d) > 0)
                                       & (np.triu(d) < 2 * r)))

    return overlaps


def uni_pos(n, box, seed=12345):
    """ Simple function to generate n uniform distributions in box-dimension.

    Parameters
    ----------
    n : int
        Number of positions to generate.
    box : array-like
        Length in each dimension of box to place n.
    seed: int or Random Number Generator
        Initial state of random number generator (default is 12345)
    Returns
    -------
    pos : ndarray
        Position of n in the box.
    """

    rng = np.random.default_rng(seed)

    l = len(box)
    pos = rng.uniform(np.zeros(l), box, (n, l))

    return pos


def uni_overlap(pos, r, box, periodic=False, trs=1000, seed=12345):
    """ Check for overlapping cell bodies in initial placement and move any
    overlapping cells to new positions. Does changes in-place.

    Parameters
    ----------
    pos : float or array-like, shape (n,m)
        Position of neuron(s)
    r : float
        Minimum radius for cell body of neuron.
    box : array-like, shape 1xN
        Dimensions of space for neurons to occupy.
    trs : int
        Number of iterations for while loop to check overlap (default value is
        1000).
    seed: int or Random Number Generator
        Initial state of random number generator (default is 12345)
    Returns
    -------
    pos : float or array-like, shape (n,m)
        Updated position with no overlapping cell bodies.
    """

    rng = np.random.default_rng(seed)

    ovrlp_logic = True
    tcount = 0

    while ovrlp_logic == True and tcount <= trs:
        ovrlp_logic = False
        overlaps = overlap(pos, r, box, periodic=periodic)

        if overlaps.any():
            n_unq = np.unique(overlaps[:, 1])
            pos[n_unq] = uni_pos(len(n_unq), box, seed=rng)
            ovrlp_logic = True

        tcount += 1
        if tcount > trs:
            sys.exit("Too many tries to avoid overlap.")

    return pos


def uni_place(n, r, box, periodic=False, trs=10000, seed=12345, **kwargs):
    """ Simple function to generate n uniform placements in box-dimension.
    Places all n neurons initially and checks for, and corrects, overlap.

    Parameters
    ----------
    n : int
        Number of positions to generate.
    r : float
        Minimum radius for cell body of neuron.
    box : array-like, shape 1xN
        Dimensions of space for neurons to occupy.
    seed: int or SeedSequence
        Initial state of random number generator (default is 12345)
    Returns
    -------
    pos : ndarray
        Position of n in the box.
    """

    rng = np.random.default_rng(seed)

    pos = uni_pos(n, box, seed=rng)
    pos = uni_overlap(pos, r, box, periodic=periodic, trs=trs, seed=rng)

    return pos


def sf_pos(n, r, a, box, home, periodic=False, seed=12345):
    """Function to get new position if there is overlap in all positions. If 
    not all are overlapping, we assume they are in a different position.

    Parameters
    ----------
    n : int
        Number of positions to generate.
    r : float
        Smallest radius allowed for neurons. Passes 2 * r to scale free 
        generator avoid overlap close to the lowest value.
    a : float
        Scaling parameter of power law.
    box : array-like, shape 1xN
        Dimensions of space for neurons to occupy.
        stats.power_law().
    home : array-like, shape Nx1
        Senter of neuron. Neurons with no home act as anchors, and additional
        neurons are placed around these.
    box : array-like, 1xN shaped (default is [1.0, 1.0, 1.0])
        Size of box to place neurons in.
    seed: int or SeedSequence
        Initial state of random number generator (default is 12345)
    Returns:
    --------
    pos : ndarray
        Position of n in the box.
    """

    rng = np.random.default_rng(seed)
    l = len(box)

    pos = rng.uniform(np.zeros(l), box, (n, l))

    pos[np.where(home < 0)] = uni_overlap(pos[np.where(home < 0)],
                                2*r, #double check if this should be 2 or just r
                                box,
                                periodic=periodic,
                                trs=1000,
                                seed=rng)

    pos[np.where(home >= 0)] = stats.power_law(len(pos[np.where(home >= 0)]),
                                                2*r,
                                                box,
                                                a=a,
                                                seed=rng,
                                                periodic=False)
    pos = pos % box

    shift = np.array([pos[int(i)] for i in home[np.where(home >= 0)]])

    pos[np.where(home >= 0)] = (pos[np.where(home >= 0)] + shift)

    pos = pos % box

    return pos


def sf_overlap(pos, r, a, box, home, periodic=False, trs=1000, seed=12345):
    """ Simple function to generate n scale free positions in box-dimension.
    First generates n uniform values un the dimensions of the box, then
    transforms these to a power law distribution according to [1]. The value
    r_min gives the lowest value for the power law, and a is the scaling
    parameter. The positions are restricted to the box through modulo.

    1. Newman, Mark EJ. "Power laws, Pareto distributions and Zipf's law."
       Contemporary physics 46.5 (2005): 323-351.

    Parameters
    ----------
    pos : float or array-like, shape (n,m)
        Position of neuron(s)
    r : float
        Smallest radius allowed for neurons. Passes 2 * r to scale free 
        generator.
    a : float
        Scaling parameter of power law.
    box : array-like
        Length in each dimension of box to place n.
    home : array-like, shape Nx1
        Senter of neuron. Neurons with no home act as anchors, and additional
    trs : int
        Number of iterations for while loop to check overlap (default value is
        1000)
    seed: int or SeedSequence
        Initial state of random number generator (default is 12345)
    Returns
    -------
    pos : float or array-like, shape (n,m)
        Updated position with no overlapping cell bodies.
    """

    l = len(box)

    rng = np.random.default_rng(seed)

    ovrlp_logic = True
    tcount = 0

    while ovrlp_logic == True and tcount <= trs:
        ovrlp_logic = False
        overlaps = overlap(pos, r, box, periodic=periodic)

        if overlaps.any():
            n_unq = np.unique(overlaps[:, 1])

            pos[n_unq] = stats.power_law(len(n_unq),
                                          2 * r,
                                          box,
                                          a=a,
                                          seed=rng,
                                          periodic=False)
            pos = pos % box

            shift = np.array([pos[int(i)] for i in home[n_unq]])

            pos[n_unq] = (pos[n_unq] + shift)
            pos = pos % box

            ovrlp_logic = True

        tcount += 1
        if tcount > trs:
            sys.exit("Too many tries to avoid overlap.")

    return pos


def sf_place(n, r, a, box, home, periodic=False, trs=10000, seed=12345):
    """
    Initialize placement of neurons and avoid overlap using preferential
    placement mimicking the Barabasi-Albert model.

    Parameters
    ----------
    n : int
        Number of positions to generate.
    r : float
        Smallest radius allowed for neurons. Passes 2 * r to scale free 
        generator.
    a : float
        Scaling parameter of power law.
    box : array-like
        Length in each dimension of box to place n.
    home : array-like, shape Nx1 or fl.oat
        Senter of neuron. Neurons with no home act as anchors, and additional
    trs : int
        Number of iterations for while loop to check overlap (default value is
        1000)
    seed: int or SeedSequence
        Initial state of random number generator (default is 12345)
    Returns
    -------
    pos : float or array-like, shape (n,m)
        Position with no overlapping cell bodies.
    """

    rng = np.random.default_rng(seed)

    if isinstance(home, float):
        home_num = int(np.floor(home * n))
        home = rng.integers(home_num, size=n)
        home[:home_num] = -1

    pos = sf_pos(n, r, a, box, home, periodic=periodic, seed=rng)
    pos = sf_overlap(pos, r, a, box, home, periodic=periodic, trs=trs, seed=rng)

    return pos, home


def place_neurons(n, r, box, periodic=False, seed=12345, method='uniform', **kwargs):
    """
    Initialize placement of neurons and avoid overlap.

    Parameters
    ----------
    n : int
        Number of neurons to place.
    r : float or array-like, shape (n,)
        Minimum radius for cell body of neuron.
    box : array-like, shape 1xN
        Size of space for neurons to occupy.
    method : string
        Metric for calculating pairwise distances (default value 'metric').

    Returns
    -------
    pos : float or array-like, shape (n,m)
        Updated position with no overlapping cell bodies.

    """
    if isinstance(seed, int):
        seed = np.random.SeedSequence(seed)

    if method.lower()=='uniform':
        pos = uni_place(n, r, box, periodic=periodic, seed=seed, **kwargs)
    else:
        sys.exit("Method not recognized.")
    return pos


def pos_init2d(n, pos, home=np.nan):
    """ Small function to initiate df for positions of neurons.

    """

    id = np.arange(0, n, 1)

    pos_df = pd.DataFrame({
        'id': id,
        'x': pos[:, 0],
        'y': pos[:, 1],
        'home': home
    })
    return pos_df


def pos_init3d(n, pos, home=np.nan):
    """ Small function to initiate df for positions of neurons.

    """

    id = np.arange(0, n, 1)

    pos_df = pd.DataFrame({
        'id': id,
        'x': pos[:, 0],
        'y': pos[:, 1],
        'z': pos[:, 2],
        'home': home
    })

    return pos_df


def RandSphere(x0):
    rand_ons = np.random.randn(3)
    rand_ons /= np.linalg.norm(rand_ons, axis=0)
    rand_ons = rand_ons*1e-6
    rand_ons[0] = rand_ons[0] + x0[0]
    rand_ons[1] = rand_ons[1] + x0[1]
    rand_ons[2] = rand_ons[2] + x0[2]
    return rand_ons


def RandTwo(x0):
    '''
    take all make 3d random with power law distibution (set up variabl?)
    uiform distribution around unitial neurons? or power law? or exp...
    check distribution

    '''
    dx = np.random.uniform(-1, 1, size = 3)
    dx = 1e-6 * dx / np.sqrt(np.sum(dx**2))
    x1 = x0+dx

    return x1

def ln_pos(mu, sig, scale, m, box, home, seed=12345):
    """ Simple function to generate n log normal positions in box-dimension.
    First generates n uniform values un the dimensions of the box, then
    transforms these to a power law distribution according to [1]. The value
    r_min gives the lowest value for the power law, and a is the scaling
    parameter. The positions are restricted to the box through modulo.

    1. Newman, Mark EJ. "Power laws, Pareto distributions and Zipf's law."
       Contemporary physics 46.5 (2005): 323-351.

    Parameters
    ----------
    n : int
        Number of positions to generate.
    r_min: float
        Minimum value for generated power law.
    a: float
        Power law exponent.
    box : array-like
        Length in each dimension of box to place n.
    seed: int or rng Generator
        Seed for defaul_rng() or chosen rng passed (default value 1234)
    Returns
    -------
    pos : ndarray
        Position of n in the box.
    """

    l = len(box)
    rng = np.random.default_rng(seed)
    pos = rng.lognormal(mu, sig, (m, l))*scale

    shift = np.array([pos[int(i)] for i in home[np.where(home >= 0)]])
    pos[np.where(home >= 0)] = (pos[np.where(home >= 0)] + shift)

    pos = pos % box

    return pos


def ln_overlap(mu, sig, scale, m, box, home, seed=12345):
    """Function to get new position if there is overlap in all positions. If not
    all are overlapping, we assume they are in a different position.

    Parameters
    ----------
    n : int
        Position of neuron 1.
    posb : float
        Position of neuron 2.
    r : float
        Smallest radius allowed for neurons.
    box : array-like, 1xN shaped (default is [1.0, 1.0, 1.0])
        Size of box to place neurons in.

    Returns:
    --------
    posb : float
    overlaps : bool

    """

    rng = np.random.default_rng(seed)

    pos = None
    return pos


def ln_place(mu, sig, scale, m, box, home, seed=12345):
    """ Simple function to generate n uniform placements in box-dimension.
    Places all n neurons initially and checks for, and corrects, overlap.

    Parameters
    ----------
    n : int
        Number of positions to generate.
    box : array-like
        Length in each dimension of box to place n.
    seed: int or SeedSequence
        Initial state of random number generator (default is 12345)
    Returns
    -------
    pos : ndarray
        Position of n in the box.
    """

    rng = np.random.default_rng(seed)

    pos = ln_pos(mu, sig, scale, m, box, seed=rng)
    pos = ln_overlap(mu, sig, scale, m, box, seed=rng)

    return pos
