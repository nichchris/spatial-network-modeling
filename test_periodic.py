import numpy as np
import scipy as sp
import coordinate


if __name__ == '__main__':
    rands = np.random.uniform(0,1, (1000, 2))
    pos = np.array([
        # [0,0],
        # [.999, .999],        
        # [0,.999],
        # [.999, 0],
        [0.25, 0.25],
        [0.25, 0.75],
        [0.75, 0.25],
        [0.75, 0.75],
        [0.52012024, 0.9977939 ], [0.49485442, 0.01256627],
        [0.03131782, 0.49644336], [0.99977804, 0.51482676],
        [0.99977804, 0.51482676], [0.03131782, 0.49644336]
        ])
    dists = coordinate.periodic_dist(rands, periodic=False)
    print(f'max test pos, periodic: {np.max(coordinate.periodic_dist(pos))}')
    print(f'max test pos, not periodic: {np.max(coordinate.periodic_dist(pos, periodic=False))}')    
    print(f'max test random, periodic: {np.max(coordinate.periodic_dist(rands))}')
    print(f'max test random, not periodic: {np.max(coordinate.periodic_dist(rands, periodic=False))}')
    print(f'sqrt 1/2: {np.sqrt(1/2)}')
    print(f'sqrt 2: {np.sqrt(2)}')
    # print(np.argwhere(dists>0.704).shape)
    # print(np.argwhere(dists>0.704))
    # for pair in np.argwhere(dists>0.96):
    #     print(f'long pair is {rands[pair[0]]} and {rands[pair[1]]}')
