"""
*** SCRIPT FILE ***
$ python3 glmplot.py

"""

import numpy as np

from src import util
from src import glmcc
import params


def estimate_single_connection(t_sp, synaptic_delay):
    glm_list = [glmcc.GLMCC(delay=delay) for delay in synaptic_delay]  # optimize synaptic delay
    tmp_log_posterior = -1e05  # minimum log posterior
    max_posterior_idx = None

    for i, glm in enumerate(glm_list):
        if glm.fit(t_sp):
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


def glm_plot(data_dir, prefix, extension, target_id, reference_id, save_dir, start, end, window, synaptic_delay):
    if target_id == reference_id:
        print("target id and reference id cannot take the same value\n")
        return None

    else:
        spike_trains = util.load_trains(data_dir, prefix, extension, start, end)
        t_sp = util.relative_sptime(target=spike_trains[target_id], reference=spike_trains[reference_id], window=window)

        print("estimating the connection from {} to {}".format(reference_id, target_id))
        optimal_glm = estimate_single_connection(t_sp=t_sp, synaptic_delay=synaptic_delay)
        if optimal_glm is not None:
            print("optimal synaptic delay: {}".format(optimal_glm.delay))
            print("log posterior: {}".format(optimal_glm.max_log_posterior))
            print("J_ij, J_ji: {}".format(optimal_glm.theta[-2:]))
            print("J thresholds: {}".format(optimal_glm.j_thresholds))
            print("presence of connectivity: {}".format(np.abs(optimal_glm.theta[-2:]) >= optimal_glm.j_thresholds))
            optimal_glm.plot(t_sp=t_sp, target_id=target_id, reference_id=reference_id,
                             save_path=save_dir + 'cc_from{}to{}.png'.format(reference_id, target_id))
        else:
            print("GLM could not be fitted correctly.")
            return None


if __name__ == '__main__':
    glm_plot(data_dir=params.DATA_DIR, prefix=params.PREFIX, extension=params.EXTENSION,
             target_id=params.PLOT_TARGET_ID, reference_id=params.PLOT_REFERENCE_ID, save_dir=params.SAVE_DIR,
             start=params.START, end=params.END, window=params.WINDOW, synaptic_delay=params.SYNAPTIC_DELAY)
