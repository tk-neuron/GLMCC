"""
utility functions when handling files
"""

import glob


def load_trains(data_dir, prefix, extension, start, end, fr_threshold=0.0):
    """
    load spike trains from dataset directory
    :param data_dir: str, dataset directory
    :param prefix: str, filename prefix
    :param extension: str, file extension
    :param start: float, start time of measurement
    :param end: float, end time of measurement
    :param fr_threshold: float, firing rate threshold when loading spike trains
    :return: dict, keys: neuron id (correspond to filename), values: spike train (list)
    """
    paths = glob.glob(data_dir + prefix + '*.' + extension)
    trains = dict()

    for path in paths:
        try:
            neuron_id = int(path.split('/')[-1].split(prefix)[1].split('.' + extension)[0])
        except ValueError:
            continue

        with open(path) as f:
            train = f.readlines()

        firing_rate = len(train) * 1000 / (end - start)
        # print(str(round(firing_rate, 3)) + 'Hz')

        if firing_rate >= fr_threshold:
            train = [float(sp) for sp in train if start <= float(sp) <= end]
            trains[neuron_id] = train

    return trains


def list_pairs(trains):
    """
    list all possible combinations of neurons
    :param trains: dict, keys: neuron id, values: spike train
    :return: list of tuples
    """
    neuron_ids = sorted(list(trains.keys()))
    pairs = []
    for i, idx in enumerate(neuron_ids):
        for j in range(i+1, len(neuron_ids)):
            pairs.append((idx, neuron_ids[j]))
    return pairs


def relative_sptime(target, reference, window):
    """
    calculate relative spike time (target - reference) within a time window
    :param target: list of float, target neuron's spike time
    :param reference: list of float, referece neuron's spike time
    :param window: window size (default=50)
    :return: a list of relative spike time (float)
    """

    def search_max_idx(lst, upper, start_idx=0):
        """
        returns index of the largest element of all smaller than upper in lst; scans from start_idx
        note that lst has to be a list of elements monotonously increasing
        """
        idx = start_idx
        while len(lst) > idx and lst[idx] <= upper:
            idx += 1
        return idx

    t_sp = []
    min_i, max_i = 0, 0

    for j, tsp_j in enumerate(reference):
        # reuse min_i and max_i values for next iteration to decrease the amount of elements to scan
        min_i = search_max_idx(lst=target, upper=tsp_j - window, start_idx=min_i)
        max_i = search_max_idx(lst=target, upper=tsp_j + window, start_idx=max_i)

        # a list of relative spike time
        t_sp.extend([(target[i] - reference[j]) for i in range(min_i, max_i)])

    return t_sp
