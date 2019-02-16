import json
from distutils.util import strtobool
from typing import NamedTuple

import numpy as np
from scipy import stats


class RTResults(NamedTuple):
    """NamedTuple that represents reaction time results
    from running simulation with FVF framework.

    Fields
    ------
    search_types: tuple
        Unique set of search types in results. Tuple of str elements.
        E.g., ('easy', 'medium', 'hard')
    display_sizes: tuple
        Unique set of display sizes in results. Tuple of int elements.
        E.g., (6, 12, 18)
    target_present: tuple
        Unique set of "target present or absent" conditions in results.
        Tuple of bool elements. E.g., (True, False).
    conditions: list
        Each permutation of (search type, display size, target present or absent)
        that appears in the results. List of tuples.
    RTs_by_condition: dict
        Reaction times by condition. Dict where key is one condition, and
        the corresponding value is a numpy array with all reaction times
        for that condition.
    mean_RTs_by_condition: dict
        Mean reaction times by condition. Dict where key is one condition, and
        the corresponding value is numpy.mean(reaction_times).
    std_RTs_by_condition: dict
        Standard deviation of reaction times by condition. Dict where key is
        one condition, and the corresponding value is numpy.std(reaction_times).
    mean_RTs_all_display_sizes: dict
        Mean reaction times for each search type, target present or absent,
        for all display sizes. Dict where key has form (search type, is target present),
        and the corresponding value is a numpy array of mean reaction times, with each
        element corresponding to one display size from display_sizes.
    mean_RTs_regress_results: dict
        Results of performing linear regression on reaction times v. display size for
        each item in mean-RTs_all_display_sizes
    std_RTs_all_display_sizes: dict
        Standard deviation of reaction times for each search type, target present or absent,
        for all display sizes.
    """
    search_types: tuple
    display_sizes: tuple
    target_present: tuple
    conditions: list
    RTs_by_condition: dict
    mean_RTs_by_condition: dict
    std_RTs_by_condition: dict
    mean_RTs_all_display_sizes: dict
    mean_RTs_regress_results: dict
    std_RTs_all_display_sizes: dict


class LinRegressResults(NamedTuple):
    """NamedTuple that represents results of linear regression.
    The returned values from scipy.stats.linregress, but in a NamedTuple.

    Fields
    ------
    slope : float
        slope of the regression line
    intercept : float
        intercept of the regression line
    r-value : float
        correlation coefficient
    p-value : float
        two-sided p-value for a hypothesis test whose null hypothesis is that the slope is zero.
    stderr : float
        Standard error of the estimate
    """
    slope: float
    intercept: float
    r_value: float
    p_value: float
    std_err: float


def reaction_times(rt_json, responses_json):
    """munge results from a reaction_times.json file into format for plotting

    Parameters
    ----------
    rt_json : str
        path to a reaction_times.json file created by running fvf.main
    responses_json : str
        path to a responses.json file saved created running fvf.main

    Returns
    -------
    reaction_time_results : RTResults
        instance of RTResults NamedTuple with the following fields:
            search_types: tuple
                Unique set of search types in results. Tuple of str elements.
                E.g., ('easy', 'medium', 'hard')
            display_sizes: tuple
                Unique set of display sizes in results. Tuple of int elements.
                E.g., (6, 12, 18)
            target_present: tuple
                Unique set of "target present or absent" conditions in results.
                Tuple of bool elements. E.g., (True, False).
            conditions: list
                Each permutation of (search type, display size, target present or absent)
                that appears in the results. List of tuples.
            RTs_by_condition: dict
                Reaction times by condition. Dict where key is one condition, and
                the corresponding value is a numpy array with all reaction times
                for that condition.
            mean_RTs_by_condition: dict
                Mean reaction times by condition. Dict where key is one condition, and
                the corresponding value is numpy.mean(reaction_times).
            std_RTs_by_condition: dict
                Standard deviation of reaction times by condition. Dict where key is
                one condition, and the corresponding value is numpy.std(reaction_times).
            mean_RTs_all_display_sizes: dict
                Mean reaction times for each search type, target present or absent,
                for all display sizes. Dict where key has form (search type, is target present),
                and the corresponding value is a numpy array of mean reaction times, with each
                element corresponding to one display size from display_sizes.
            mean_RTs_regress_results: dict
                Results of performing linear regression on reaction times v. display size for
                each item in mean-RTs_all_display_sizes
            std_RTs_all_display_sizes: dict
                Standard deviation of reaction times for each search type, target present or absent,
                for all display sizes.
    """
    with open(rt_json) as fp:
        RTs = json.load(fp)

    with open(responses_json) as fp:
        responses = json.load(fp)

    search_types = []
    display_sizes = []
    target_present = []
    conditions = []
    RTs_by_condition = {}
    mean_RTs_by_condition = {}
    std_RTs_by_condition = {}
    for key, val in RTs.items():
        # convert text key back into Python types
        split_key = key.split(',')
        search_type = split_key[0]
        display_size = int(split_key[1])
        is_target_present = bool(strtobool(split_key[2].strip()))

        # add to conditions that will be returned
        search_types.append(search_type)
        display_sizes.append(display_size)
        target_present.append(is_target_present)

        tup_key = tuple([search_type, display_size, is_target_present])
        conditions.append(tup_key)
        rt_arr = np.asarray(val)
        RTs_by_condition[tup_key] = rt_arr

        # keep only correct trials, as in Young Hulleman 2013
        response_arr = np.asarray(responses[key])
        RTs_to_use = np.equal(response_arr, is_target_present)
        mean_RTs_by_condition[tup_key] = np.mean(rt_arr[RTs_to_use])
        std_RTs_by_condition[tup_key] = np.std(rt_arr[RTs_to_use])

    search_types = tuple((set(search_types)))
    display_sizes = tuple(
        sorted(  # sorted, so display_sizes is in ascending numerical order
            set(display_sizes)
        ))
    target_present = tuple(set(target_present))

    mean_RTs_all_display_sizes = {}
    mean_RTs_regress_results = {}
    std_RTs_all_display_sizes = {}
    # even more munging
    for search_type in search_types:
        for is_target_present in target_present:
            mean_RT_vals = []
            std_RT_vals = []
            for display_size in display_sizes:
                condition_tup = tuple([search_type, display_size, is_target_present])
                mean_val = mean_RTs_by_condition[condition_tup]
                mean_RT_vals.append(mean_val)
                std_val = std_RTs_by_condition[condition_tup]
                std_RT_vals.append(std_val)
            key = tuple([search_type, is_target_present])
            mean_RT_vals = np.asarray(mean_RT_vals)
            mean_RTs_all_display_sizes[key] = mean_RT_vals
            slope, intercept, r_value, p_value, std_err = stats.linregress(display_sizes,
                                                                           mean_RT_vals)
            regress_result = LinRegressResults(slope, intercept, r_value, p_value, std_err)
            mean_RTs_regress_results[key] = regress_result
            std_RT_vals = np.asarray(std_RT_vals)
            std_RTs_all_display_sizes[key] = std_RT_vals

    return RTResults(search_types,
                     display_sizes,
                     target_present,
                     conditions,
                     RTs_by_condition,
                     mean_RTs_by_condition,
                     std_RTs_by_condition,
                     mean_RTs_all_display_sizes,
                     mean_RTs_regress_results,
                     std_RTs_all_display_sizes)


class NumFixationsResults(NamedTuple):
    """NamedTuple that represents number of fixations results
    from running simulation with FVF framework.

    Fields
    ------
    search_types: tuple
        Unique set of search types in results. Tuple of str elements.
        E.g., ('easy', 'medium', 'hard')
    display_sizes: tuple
        Unique set of display sizes in results. Tuple of int elements.
        E.g., (6, 12, 18)
    target_present: tuple
        Unique set of "target present or absent" conditions in results.
        Tuple of bool elements. E.g., (True, False).
    conditions: list
        Each permutation of (search type, display size, target present or absent)
        that appears in the results. List of tuples.
    num_fixations_by_condition: dict
        Number of fixations by condition. Dict where key is one condition, and
        the corresponding value is a numpy array with fixation counts for all trials
        for that condition.
    mean_num_fixations_by_condition: dict
        Mean number of fixations by condition. Dict where key is one condition, and
        the corresponding value is numpy.mean(number of fixations).
    std_num_fixations_by_condition: dict
        Standard deviation of number of fixations by condition. Dict where key is
        one condition, and the corresponding value is numpy.std(number of fixations).
    mean_num_fixations_all_display_sizes: dict
        Mean number of fixations for each search type, target present or absent,
        for all display sizes. Dict where key has form (search type, is target present),
        and the corresponding value is a numpy array of mean number of fixations, with each
        element corresponding to one display size from display_sizes.
    """
    search_types: tuple
    display_sizes: tuple
    target_present: tuple
    conditions: list
    num_fixations_by_condition: dict
    mean_num_fixations_by_condition: dict
    std_num_fixations_by_condition: dict
    mean_num_fixations_all_display_sizes: dict


def num_fixations(nf_json):
    """munge results from a num_fixations.json file into format for plotting

    Parameters
    ----------
    nf_json : str


    Returns
    -------
    num_fixations_results : NumFixationsResults
        instance of RTResults NamedTuple with the following fields:
            search_types: tuple
                Unique set of search types in results. Tuple of str elements.
                E.g., ('easy', 'medium', 'hard')
            display_sizes: tuple
                Unique set of display sizes in results. Tuple of int elements.
                E.g., (6, 12, 18)
            target_present: tuple
                Unique set of "target present or absent" conditions in results.
                Tuple of bool elements. E.g., (True, False).
            conditions: list
                Each permutation of (search type, display size, target present or absent)
                that appears in the results. List of tuples.
            num_fixations_by_condition: dict
                Number of fixations by condition. Dict where key is one condition, and
                the corresponding value is a numpy array with fixation counts for all trials
                for that condition.
            mean_num_fixations_by_condition: dict
                Mean number of fixations by condition. Dict where key is one condition, and
                the corresponding value is numpy.mean(number of fixations).
            std_num_fixations_by_condition: dict
                Standard deviation of number of fixations by condition. Dict where key is
                one condition, and the corresponding value is numpy.std(number of fixations).
            mean_num_fixations_all_display_sizes: dict
                Mean number of fixations for each search type, target present or absent,
                for all display sizes. Dict where key has form (search type, is target present),
                and the corresponding value is a numpy array of mean number of fixations, with each
                element corresponding to one display size from display_sizes.
    """
    with open(nf_json) as fp:
        num_fix = json.load(fp)

    search_types = []
    display_sizes = []
    target_present = []
    conditions = []
    num_fixations_by_condition = {}
    mean_num_fixations_by_condition = {}
    std_num_fixations_by_condition = {}
    for key, val in num_fix.items():
        # convert text key back into Python types
        split_key = key.split(',')
        search_type = split_key[0]
        display_size = int(split_key[1])
        is_target_present = bool(strtobool(split_key[2].strip()))

        # add to conditions that will be returned
        search_types.append(search_type)
        display_sizes.append(display_size)
        target_present.append(is_target_present)

        tup_key = tuple([search_type, display_size, is_target_present])
        conditions.append(tup_key)
        nf_arr = np.asarray(val)
        num_fixations_by_condition[tup_key] = nf_arr

        mean_num_fixations_by_condition[tup_key] = np.mean(nf_arr)
        std_num_fixations_by_condition[tup_key] = np.std(nf_arr)

    search_types = tuple((set(search_types)))
    display_sizes = tuple(
        sorted(  # sorted, so display_sizes is in ascending numerical order
            set(display_sizes)
        ))
    target_present = tuple(set(target_present))

    mean_num_fixations_all_display_sizes = {}
    # even more munging
    for search_type in search_types:
        for is_target_present in target_present:
            mean_num_fixations_vals = []
            for display_size in display_sizes:
                condition_tup = tuple([search_type, display_size, is_target_present])
                mean_val = mean_RTs_by_condition[condition_tup]
                mean_num_fixations_vals.append(mean_val)
            key = tuple([search_type, is_target_present])
            mean_num_fixations_vals = np.asarray(mean_num_fixations_vals)
            mean_num_fixations_all_display_sizes[key] = mean_num_fixations_vals

    return NumFixationsResults(search_types,
                               display_sizes,
                               target_present,
                               conditions,
                               num_fixations_by_condition,
                               mean_num_fixations_by_condition,
                               std_num_fixations_by_condition,
                               mean_num_fixations_all_display_sizes)
