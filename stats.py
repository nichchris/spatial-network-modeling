"""
Collection of statistical stuff.
"""
import numpy as np
import sys

def rng_streams(seed, n):
    """ Checks input type and returns rng stream as necessary.

    Parameters
    ----------

    Returns
    -------
    
    """

    if isinstance(seed, int):
        seeds = np.random.SeedSequence(seed)
    else:
        seeds = seed
        
    child_seed = seeds.spawn(n)
    streams = [np.random.default_rng(s) for s in child_seed]

    return streams


def polar(r, seed=12345):
    """ Random direction of neurons in polar coordinate. Returns as 
    xy-coordinates.
    
    """
    rng = np.random.default_rng(seed)

    theta =  rng.uniform(0, 2*np.pi, len(r))

    p = np.zeros((len(r), 2))

    p[:, 0] = r * np.cos(theta)
    p[:, 1] = r * np.sin(theta)

    return p

def spherical(r, seed=12345):
    """ Random direction of neurons in polar coordinate. Returns as 
    xyz-coordinates.

    Parameters
    ----------
    r : float
        Smallest value (x_min in [1]) allowed for position. 
    seed: int or SeedSequence
        Initial state of random number generator (default is 12345)
    Returns
    -------
    p : float or array-like, shape (n,m)
        Position of n in the box.    
    """

    rng = np.random.default_rng(seed)

    theta =  rng.uniform(0, 2*np.pi, len(r))
    phi = rng.uniform(0, np.pi, len(r))

    p = np.zeros((len(r), 3))

    p[:, 0] = r * np.cos(theta) * np.sin(phi)
    p[:, 1] = r * np.sin(theta) * np.sin(phi)
    p[:, 2] = r * np.cos(phi)

    return p

def power_law1D(n, r, box, a=1.5, seed=12345, periodic=False):
    """ Statistical method to yield power law position in 1d. Uses function
    from [1] to generate power law distribution.
    
    1. Newman, Mark EJ. "Power laws, Pareto distributions and Zipf's law."
       Contemporary physics 46.5 (2005): 323-351.

    Parameters
    ----------
    n : int
        Number of positions to generate.
    r : float
        Smallest value (x_min in [1]) allowed for position. 
    a : float
        Scaling parameter of power law.
    box : array-like
        Length in each dimension of box to place n.
    seed: int or SeedSequence
        Initial state of random number generator (default is 12345)
    periodic : bool
        Periodic boundary box
    Returns
    -------
    p : float or array-like, shape (n,m)
        Position of n in the box.    
    """

    rng = np.random.default_rng(seed)

    sign = rng.integers(2, size=n)*2 -1

    uni = rng.uniform(0, 1, n)

    p = r * (1 - uni)**(-1/(a - 1)) * sign

    if periodic==True:
        p = p % box

    return p


def power_law2D(n, r, box, a=1.5, seed=12345, periodic=False):
    """ Statistical method to yield power law position in 2d. Uses function
    from [1] to generate power law distribution.
    
    1. Newman, Mark EJ. "Power laws, Pareto distributions and Zipf's law."
       Contemporary physics 46.5 (2005): 323-351.

    Parameters
    ----------
    n : int
        Number of positions to generate.
    r : float
        Smallest value (x_min in [1]) allowed for position. 
    a : float
        Scaling parameter of power law.
    box : array-like
        Length in each dimension of box to place n.
    seed: int or SeedSequence
        Initial state of random number generator (default is 12345)
    periodic : bool
        Periodic boundary box
    Returns
    -------
    p : float or array-like, shape (n,m)
        Position of n in the box.    
    """

    rng = np.random.default_rng(seed)

    uni = rng.uniform(0, 1, n)
    v = r * (1 - uni)**(-1 / (a - 1))

    p = polar(v, seed=rng)

    if periodic == True:
        p = p % box

    return p


def power_law3D(n, r, box, a=1.5, seed=12345, periodic=False):
    """ Statistical method to yield power law position in 3d. Uses function
    from [1] to generate power law distribution.
    
    1. Newman, Mark EJ. "Power laws, Pareto distributions and Zipf's law."
       Contemporary physics 46.5 (2005): 323-351.

    Parameters
    ----------
    n : int
        Number of positions to generate.
    r : float
        Smallest value (x_min in [1]) allowed for position. 
    a : float
        Scaling parameter of power law.
    box : array-like
        Length in each dimension of box to place n.
    seed: int or SeedSequence
        Initial state of random number generator (default is 12345)
    periodic : bool
        Periodic boundary box
    Returns
    -------
    p : float or array-like, shape (n,m)
        Position of n in the box.    
    """

    rng = np.random.default_rng(seed)

    uni = rng.uniform(0, 1, n)
    v = r * (1 - uni)**(-1 / (a - 1))

    p = spherical(v, seed=rng)

    if periodic == True:
        p = p % box

    return p


def power_law(n, r, box, a=1.5, seed=12345, periodic=False):
    """ Function to choose sf model to use to return correct placements.
    Uses function from [1] to generate power law distribution.
    
    1. Newman, Mark EJ. "Power laws, Pareto distributions and Zipf's law."
       Contemporary physics 46.5 (2005): 323-351.

    Parameters
    ----------
    n : int
        Number of positions to generate.
    r : float
        Smallest value (x_min in [1]) allowed for position. 
    a : float
        Scaling parameter of power law.
    box : array-like
        Length in each dimension of box to place n.
    seed: int or SeedSequence
        Initial state of random number generator (default is 12345)
    periodic : bool
        Periodic boundary box
    Returns
    -------
    p : float or array-like, shape (n,m)
        Position of n in the box.    
    """

    if len(box) == 1:
        sf = power_law1D(n, r, box, a=a, seed=seed, periodic=periodic)
    elif len(box) == 2:
        sf = power_law2D(n, r, box, a=a, seed=seed, periodic=periodic)
    elif len(box) == 3:
        sf = power_law3D(n, r, box, a=a, seed=seed, periodic=periodic)
    else:
        sys.exit("Dimensions beyond 3D are not implemented.")

    return sf

def log_norm1D(n, r, box,  mu, sig=1, scale=1e-1, seed=12345, periodic=False):
    """ Statistical method to yield power law position.
    
    """
    rng = np.random.default_rng(seed)

    sign = rng.integers(2, size=n)*2 -1

    p = rng.lognormal(mu, sig, size=n)* scale * sign

    if periodic==True:
        p = p % box

    return p


def log_norm2D(n, r, box,  mu, sig=1, scale=1e-1, seed=12345, periodic=False):
    """ Statistical method to yield power law position.
    
    """

    rng = np.random.default_rng(seed)

    v = rng.lognormal(mu, sig, size=n) * scale

    p = polar(v)

    if periodic == True:
        p = p % box

    return p

def log_norm3D(n, r, box,  mu, sig=1, scale=1e-1, seed=12345, periodic=False):
    """ Statistical method to yield power law position.
    
    """

    rng = np.random.default_rng(seed)

    v = rng.lognormal(mu, sig, size=n) * scale

    p = spherical(v)

    if periodic == True:
        p = p % box

    return p


def log_norm(n, r, box,  mu, sig=1, scale=1e-1, seed=12345, periodic=False):
    """ Function to choose sf model to use to return correct placements.

    Parameters
    ----------

    Returns
    -------
    """

    if len(box) == 1:
        lnorm = log_norm1D(n, r, box,  mu, sig=1, scale=1e-1, seed=12345, periodic=False)
    elif len(box) == 2:
        lnorm = log_norm2D(n, r, box,  mu, sig=1, scale=1e-1, seed=12345, periodic=False)
    elif len(box) == 3:
        lnorm = log_norm3D(n, r, box,  mu, sig=1, scale=1e-1, seed=12345, periodic=False)
    else:
        sys.exit("Dimensions beyond 3D are not implemented.")

    return lnorm


def expo1D(n, r, box, scale=1e-1, seed=12345, periodic=False):
    """ Statistical method to yield power law position.
    
    """
    rng = np.random.default_rng(seed)

    sign = rng.integers(2, size=n) * 2 - 1

    p = (rng.exponential(scale, n) + 2*r) * sign

    if periodic == True:
        p = p % box

    return p


def expo2D(n, r, box, scale=1e-1, seed=12345, periodic=False):
    """ Statistical method to yield power law position.
    
    """

    rng = np.random.default_rng(seed)

    v = rng.exponential(scale, n) + 2 * r

    p = polar(v)

    if periodic == True:
        p = p % box

    return p


def expo3D(n, r, box, scale=1e-1, seed=12345, periodic=False):
    """ Statistical method to yield power law position.
    
    """

    rng = np.random.default_rng(seed)

    v = rng.exponential(scale, n) + 2 * r

    p = spherical(v)

    if periodic == True:
        p = p % box

    return p


def expo(n, r, box, scale=1e-1, seed=12345, periodic=False):
    """ Function to choose sf model to use to return correct placements.

    Parameters
    ----------

    Returns
    -------
    """

    if len(box) == 1:
        expo = expo1D(n, r, box, scale=1e-1, seed=12345, periodic=False)
    elif len(box) == 2:
        expo = expo2D(n, r, box, scale=1e-1, seed=12345, periodic=False)
    elif len(box) == 3:
        expo = expo3D(n, r, box, scale=1e-1, seed=12345, periodic=False)
    else:
        sys.exit("Dimensions beyond 3D are not implemented.")

    return expo


def lognorm_decay(dists, L=None, sigma=10, mu=0.00):
    """ Somple visualisation of exponentially decaying dependancy for
    chance to connect as proposed by Waxman.

    Parameters
    ----------

    Returns
    -------

    """
    if L is None:
        L = max(dists)

    dists = dists / L
    p = (1 / (np.sqrt(2 * np.pi) * sigma * dists)) * np.exp(
        (-1 / 2) * ((np.log(dists) - mu) / sigma)**2)

    return p
