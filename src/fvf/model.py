"""Simple fixation-based conceptual framework to explain
reaction times on item-based visual search tasks

Implementation of model proposed in:
Hulleman, J., & Olivers, C. (2017).
The impending demise of the item in visual search.
Behavioral and Brain Sciences, 40, E132.
doi:10.1017/S0140525X15002794
"""
from typing import NamedTuple

import numpy as np


class MaxItemsBySearchType(NamedTuple):
    easy: int
    medium: int
    hard: int


class Trial(NamedTuple):
    """class that represents a visual search task trial;
    returned after calling VisSearchModel.run_trial()

    Fields
    ------
    response: bool
        True if subject responds that target is present,
        False if subject responds that target is absent.
    reaction_time: int
        in units of milliseconds
    seen_arr: np.ndarray
        elements that are True were seeing during the series of fixations
    num_fixations: int
        number of fixations
    fix_locs: list
        locations of fixations
    fvf_sizes: list
        size of fvf for each fixation
    fvf_per_fix: list
        actual "contents" of fvf for each fixation
    """
    response: bool
    reaction_time: int
    seen_arr: np.ndarray
    num_fixations: int
    fix_locs: list
    fvf_sizes: list
    fvf_per_fix: list


class FVFModel:
    """Model of a subject performing visual search tasks
    within the fixation-based framework proposed by Hulleman & Olivers 2017.
    """
    def __init__(self,
                 min_items=1,
                 max_items_by_search_type=MaxItemsBySearchType(30, 7, 1),
                 prev_patch_memory=4,
                 fixation_duration=250,
                 quit_threshold=0.85):
        """__init__ function

        Parameters
        ----------
        min_items : int
            minimum number of items in functional visual field.
            Default is 1.
        max_items_by_search_type : NamedTuple
            Maps max_items to search type, e.g. 'easy' = 30.
            Must be an instance of fvf.model.MaxItemsBySearchType.
            Default is MaxItemsBySearchType(30, 7, 1).
        prev_patch_memory : int
            Number of previous patches kept in memory. If a patch is
            in memory, it will not be revisited when selecting a new
            patch. Default is 4.
        fixation_duration : int
            duration of a fixation in milliseconds. This amount is
            added to the total trial duration for each fixation.
        quit_threshold : float
            between 0 and 1. Percent of search array that has to be
            seen before quitting search.
            Default is 0.85, i.e. 85%.

        Notes
        -----
        Defaults are as in Hulleman Olivers 2017.
        """
        if min_items < 0 or type(min_items) != int:
            raise ValueError('min_items must be a non-negative integer')

        if type(max_items_by_search_type) != MaxItemsBySearchType:
            raise TypeError('max_items_by_search_type must be an instance of '
                            'aver.fixsim.MaxItemsBySearchType')

        if prev_patch_memory < 0 or type(prev_patch_memory) != int:
            raise ValueError('prev_patch_memory must be a non-negative integer')

        if fixation_duration < 0 or type(fixation_duration) != int:
            raise ValueError('fixation_duration must be a non-negative integer')

        if quit_threshold < 0 or quit_threshold > 1:
            raise ValueError('quit_threshold must be between 0 and 1')

        self.min_items = min_items
        self.max_items_by_search_type = max_items_by_search_type
        self.prev_patch_memory = prev_patch_memory
        self.fixation_duration = fixation_duration
        self.quit_threshold = quit_threshold
        self.fvf_vals = None  # set by self.run_trials function

    def _select_new_patch(self, search_arr, fix_locs):
        """helper function to select a new patch to fixate,
        given a search array and a list of previous fixation locations

        Parameters
        ----------
        search_arr : np.ndarray
        fix_locs : list
            of previous fixation locations, i.e., patches. used when
            selecting new patches. If newly drawn locations are in the
            set of previous locations kept in memory, i.e. in
            `fix_locs[-self.prev_patch_memory:]`, then that patch is
            discarded and another patch is drawn randomly.

        Returns
        -------
        fix_loc : int
            index of new fixation location, i.e. "patch"
        """
        fix_loc = -1
        while fix_loc == -1:
            fix_loc_tmp = np.random.choice(np.arange(len(search_arr)))
            if fix_loc_tmp not in fix_locs[-self.prev_patch_memory:]:
                fix_loc = fix_loc_tmp
        return fix_loc

    def _get_fvf_size(self):
        """helper function to get size of functional visual field for
        a fixation.

        Returns
        -------
        fvf_size : int
            uniform draw from self.min_items to self.max_items inclusive,
            using numpy.random.choice on self.fvf_vals
        """
        return np.random.choice(self.fvf_vals)  # uses uniform probability

    def _fixate(self, search_arr, target, fix_loc, fvf_size, seen_arr):
        """helper function that simulates fixation

        Parameters
        ----------
        search_arr : numpy.ndarray
        target : int
        fix_loc : int
            index of fixation location, i.e. where it starts
        fvf_size : int
            size of functional field of view.
        seen_arr : numpy.ndarray
            Elements in search_arr that are part of a fixation are
            converted to True in this array. Used to determine whether
            quit_threshold has been passed

        Returns
        -------
        fvf: numpy.ndarray
            actual field seen, i.e., search_arr[fix_loc:fix_loc + fvf_size]
        response : bool
            True if target in functional visual field that is found by
            "fixating" (indexing into search_arr). False otherwise.
        seen_arr : numpy.ndarray
            As in Parameters section above, but updated to reflect this
            fixation.

        Notes
        -----
        As currently implemented, the functional visual field starts at
        fix_loc and ends at fix_loc + fvf_size. If this index goes beyond
        the end of the array, the fvf is simply truncated.
        """
        fvf = search_arr[fix_loc:fix_loc + fvf_size]
        seen_arr[fix_loc:fix_loc + fvf_size] = True
        if target in fvf:
            return fvf, True, seen_arr
        else:
            return fvf, False, seen_arr

    def run_trial(self, search_type, search_arr, target=1):
        """run a single trial of visual search task

        Parameters
        ----------
        search_type : str
            One of {'easy', 'medium', 'hard'}. Used to determine
            maximum number of items in functional visual field.
        search_arr : numpy.ndarray
            Array that represent visual search stimulus with a set of
            items.
        target : int
            target that subject searches for in visual search task.
            Default is 1.
        """
        if search_type not in {'easy', 'medium', 'hard'}:
            raise ValueError('search_type must be one of: {\'easy\', \'medium\', \'hard\'}')
        max_items = getattr(self.max_items_by_search_type, search_type)
        # set possible fvf_vals used when drawing fvf_size for each fixation
        self.fvf_vals = np.arange(self.min_items, max_items + 1)

        responded = False
        reaction_time = 0
        fix_locs = []  # locations of fixations
        fvf_sizes = []
        fvf_per_fix = []
        seen_arr = np.zeros(search_arr.shape)

        while responded is False:
            fix_loc = self._select_new_patch(search_arr, fix_locs)
            fix_locs.append(fix_loc)
            fvf_size = self._get_fvf_size()
            fvf_sizes.append(fvf_size)
            fvf, response, seen_arr = self._fixate(search_arr,
                                                   target,
                                                   fix_loc,
                                                   fvf_size,
                                                   seen_arr)
            fvf_per_fix.append(fvf)
            reaction_time += self.fixation_duration
            if response or (np.sum(seen_arr) / seen_arr.shape > self.quit_threshold):
                responded = True
            num_fixations = len(fix_locs)
        return Trial(response,
                     reaction_time,
                     seen_arr,
                     num_fixations,
                     fix_locs,
                     fvf_sizes,
                     fvf_per_fix)
