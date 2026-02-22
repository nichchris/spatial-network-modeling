import multiprocessing as mp
import time

import numpy as np

import packing
import stats
import utils

if __name__ == "__main__":

    # NEURON INFO:
    tic = time.time()           # Simple timer
    start_time = time.localtime()
    current_time = time.strftime("%H:%M:%S", start_time)
    print("Start time is: ", current_time)

    neurons = [1000]

    is_tuning = False

    if is_tuning:
        version_string = "2025-07-23-tuning"
        seed = 12345
    elif not is_tuning:
        version_string = "2025-07-23"
        seed = 123456

    rng = np.random.default_rng(seed)
    streams = stats.rng_streams(seed, len(neurons))

    n = 100

    id = np.arange(n)

    r = 1e-3
    r_list = [1e-3, 1e-4]
    print(f'r is set to: {r}')
    a = [3 / 2]
    periodic = [True, False]
    box_2d = [1., 1.]

    phi_2d = 0.83
    phi_3d = 0.64

    boxes = packing.box_transform(box_2d, r, phi_2d, phi_3d)

    methods = ['uni', 'pl_3-2']

    seeds = np.random.SeedSequence(seed)
    child_seed = seeds.spawn(len(neurons)*len(methods)*len(r_list))

    stream = 0

    # home_frac = [.05, .10, .20, .40]
    home_frac = [.02, 0.05, .10, .20]

    for neuron in neurons:
        for r in r_list:
            ticer = time.time()           # Simple timer
            starter_time = time.localtime()
            currenter_time = time.strftime("%H:%M:%S", starter_time)
            print("Start multiprocess for uni_mesh at ", neuron," time is: ", currenter_time)


            uni_mesh = utils.combinations_to_run(id,
                                                methods[0],
                                                neuron,
                                                r,
                                                boxes,
                                                periodic,
                                                version_string,
                                                seed=child_seed[stream])

            stream += 1

            print("Starting multiprocess")
            with mp.Pool(processes=10) as p:
                p.map(utils.uni_run, uni_mesh)

            ender_time = time.localtime()
            currenter_time = time.strftime("%H:%M:%S", ender_time)
            tocer = time.time() - ticer
            print("End time is: ", currenter_time)
            print("Elapsed time multiprocess is: ", tocer)
            print("Elapsed time is ", np.floor(tocer/60),
                "minutes and ", tocer % 60, " seconds.")
            print("\n")


            ticer = time.time()           # Simple timer
            starter_time = time.localtime()
            currenter_time = time.strftime("%H:%M:%S", starter_time)
            print("Start multiprocess for pl_3_2_mesh at ", neuron, " time is: ",
                currenter_time)

            pl_3_2_mesh = utils.combinations_to_run(id,
                                                    methods[1],
                                                    neuron,
                                                    r,
                                                    a[0],
                                                    boxes,
                                                    home_frac,
                                                    periodic,
                                                    version_string,
                                                    seed=child_seed[stream])
            stream += 1

            print("Starting multiprocess")
            with mp.Pool(processes=10) as p:
                p.map(utils.pl_run, pl_3_2_mesh)

            ender_time = time.localtime()
            currenter_time = time.strftime("%H:%M:%S", ender_time)
            tocer = time.time() - ticer
            print("End time is: ", currenter_time)
            print("Elapsed time multiprocess is: ", tocer)
            print("Elapsed time is ", np.floor(tocer/60),
                "minutes and ", tocer % 60, " seconds.")
            print("\n")

        end_time = time.localtime()
        current_time = time.strftime("%H:%M:%S", end_time)
        print("End time is: ", current_time)
        toc = time.time() - tic
        print("Total elapsed time is ", np.floor(toc / 60), "minutes and ", toc % 60,
                " seconds.")

print('Done!')
