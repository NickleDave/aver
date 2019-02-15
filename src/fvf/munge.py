import json
from distutils.util import strtobool
from typing import NamedTuple

import numpy as np


class RTResults(NamedTuple):
    """NamedTuple that represents reaction time results"""
    search_types: tuple
    display_sizes: tuple
    target_present: tuple
    conditions: list
    RTs: dict
    mean_RTs: dict
    std_RTs: dict
    mean_RTs_to_plot: dict


def munge_rt_results(rt_json):
    """munge results from a reaction_times.json file into format for plotting"""
    with open(rt_json) as fp:
        reaction_times = json.load(fp)

    search_types = []
    display_sizes = []
    target_present = []
    conditions = []
    RTs = {}
    mean_RTs = {}
    std_RTs = {}
    for key, val in reaction_times.items():
        # convert text key back into Python types
        split_key = key.split(',')  # split_key[0] will be search type, {'easy', 'medium' 'hard'}
        split_key[1] = int(split_key[1])  # display size, an int
        split_key[2] = bool(strtobool(split_key[2].strip()))  # target present, Boolean

        # add to conditions that will be returned
        search_types.append(split_key[0])
        display_sizes.append(split_key[1])
        target_present.append(split_key[2])

        tup_key = tuple(split_key)
        conditions.append(tup_key)
        rt_arr = np.asarray(val)
        RTs[tup_key] = rt_arr # convert to s
        mean_RTs[tup_key] = np.mean(rt_arr)
        std_RTs = np.std(rt_arr)

    search_types = tuple(set(search_types))
    display_sizes = tuple(set(display_sizes))
    target_present = tuple(set(target_present))

    mean_rts_to_plot = {}
    # even more munging
    for search_type in search_types:
        for is_target_present in target_present:
            mean_rt_vals = []
            for display_size in display_sizes:
                condition_tup = tuple([search_type, display_size, is_target_present])
                val = mean_RTs[condition_tup]
                mean_rt_vals.append(val)
            mean_rts_to_plot[tuple([search_type, is_target_present])] = tuple([np.asarray(display_sizes),
                                                                              np.asarray(mean_rt_vals)])

    return RTResults(search_types, display_sizes, target_present, conditions, RTs, mean_RTs, std_RTs,
                     mean_rts_to_plot)
