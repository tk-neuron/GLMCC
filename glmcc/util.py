"""
useful functions for preprocessing data for GLMCC, and output results
"""


def spiketime_relative(spiketime_tar, spiketime_ref, window_size=50.0):
    """
    calculate relative spike time (target - reference) within a time window
    :param spiketime_tar: list, target neuron's spiketime [msec]
    :param spiketime_ref: list, reference neuron's spiketime [msec]
    :param window_size: default: 50.0, float or int, window size when making CC histogram
    :return t_sp: relative spiketime
    """

    t_sp = []
    min_i, max_i = 0, 0

    for j, tsp_j in enumerate(spiketime_ref):
        # reuse min_i and max_i values for next iteration to decrease the amount of elements to scan
        min_i = _search_max_idx(lst=spiketime_tar, upper=tsp_j - window_size, start_idx=min_i)
        max_i = _search_max_idx(lst=spiketime_tar, upper=tsp_j + window_size, start_idx=max_i)

        # a list of relative spike time
        t_sp.extend([(spiketime_tar[i] - spiketime_ref[j]) for i in range(min_i, max_i)])

    return t_sp


def _search_max_idx(lst, upper, start_idx=0):
    """
    returns index of the largest element of all smaller than upper in lst; scans from start_idx
    note that lst has to be a list of elements monotonously increasing
    """
    idx = start_idx
    while len(lst) > idx and lst[idx] <= upper:
        idx += 1
    return idx


def glm_summary(glm):
    """
    fitted GLM's summary
    :param glm: GLMCC class instance
    """
    print("estimated J_ij, J_ji: {}, {}".format(round(glm.theta[-2], 2), round(glm.theta[-1], 2)))
    print("threshold: {}, {}".format(round(glm.j_thresholds[0], 2), round(glm.j_thresholds[1], 2)))
    print("max log posterior: " + str(int(glm.max_log_posterior)))
