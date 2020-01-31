"""
*** SCRIPT FILE ***
$ python3 estimation.py

this code estimates all connectivity on the dataset, and returns connectivity values in csv.
output file's name: J_[end time]_[firing threshold]Hz.csv
csv format: reference id, target id, J_ij, J_ji, J_ij threshold, J_ji threshold, max log posterior

"""

import csv
import numpy as np

from src import util
from src import glmcc
import params


def estimate_single_connection(t_sp, synaptic_delay):
    """
    optimize log posterior for synaptic delay
    """
    glm_list = [glmcc.GLMCC(delay=delay) for delay in synaptic_delay]  # optimize synaptic delay
    tmp_log_posterior = -1e05  # minimum log posterior
    max_posterior_idx = None

    for i, glm in enumerate(glm_list):
        if glm.fit(t_sp):
            print(glm.max_log_posterior)
            if glm.max_log_posterior is not None and glm.max_log_posterior > tmp_log_posterior:
                max_posterior_idx = i
                tmp_log_posterior = glm.max_log_posterior
        else:
            pass

    if max_posterior_idx is None:
        return None
    else:
        optimal_glm = glm_list[max_posterior_idx]
        return optimal_glm


def estimate_all_connection(data_dir, prefix, extension, save_dir, start, end, window, synaptic_delay, fr_threshold):
    spike_trains = util.load_trains(data_dir, prefix, extension, start, end, fr_threshold)
    print("{} neuron files loaded\n".format(len(spike_trains)))
    pairs = util.list_pairs(spike_trains)

    result = []
    # iterate for all the neuron pairs
    for pair in pairs:
        print("estimating the connection from {} to {}".format(pair[1], pair[0]))
        t_sp = util.relative_sptime(target=spike_trains[pair[0]], reference=spike_trains[pair[1]], window=window)
        optimal_glm = estimate_single_connection(t_sp=t_sp, synaptic_delay=synaptic_delay)

        if optimal_glm is not None:
            print("optimal synaptic delay: {}".format(optimal_glm.delay))
            print("J_ij, J_ji: {}".format(optimal_glm.theta[-2:]))
            print("J thresholds: {}".format(optimal_glm.j_thresholds))
            print("presence of connectivity: {}".format(np.abs(optimal_glm.theta[-2:]) >= optimal_glm.j_thresholds))

            # csv format
            result.append((pair[0], pair[1],  # reference id, target id
                           optimal_glm.theta[-2], optimal_glm.theta[-1],  # estimated J_ij, J_ji
                           optimal_glm.j_thresholds[0], optimal_glm.j_thresholds[1],  # statistical test
                           optimal_glm.max_log_posterior))  # log posterior optimized for synaptic delay
        print()

    with open(save_dir + 'J_{}_{}Hz.csv'.format(int(end/1000), fr_threshold), 'w') as f:
        csv.writer(f).writerows(result)
        print("result saved in {}".format(save_dir + 'J_{}_{}Hz.csv'.format(int(end/1000), fr_threshold)))
    return result


if __name__ == '__main__':
    estimate_all_connection(data_dir=params.DATA_DIR, prefix=params.PREFIX, extension=params.EXTENSION,
                            save_dir=params.SAVE_DIR, start=params.START, end=params.END, window=params.WINDOW,
                            synaptic_delay=params.SYNAPTIC_DELAY, fr_threshold=params.FIRING_RATE_THRESHOLD)
