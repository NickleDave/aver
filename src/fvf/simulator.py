"""Simple fixation-based conceptual framework to explain
reaction times on item-based visual search tasks

Implementation of model proposed in:
Hulleman, J., & Olivers, C. (2017).
The impending demise of the item in visual search.
Behavioral and Brain Sciences, 40, E132.
doi:10.1017/S0140525X15002794

Display size defaults from Young and Hulleman 2013.
"""
import numpy as np

from .model import FVFModel


class Simulator:
    def __init__(self,
                 trials_per_condition=10000,
                 display_sizes=(6, 12, 18),
                 task_difficulties=('easy', 'medium', 'hard'),
                 target_presence=(True, False),
                 target=1):
        """__init__ method

        Parameters
        ----------
        trials_per_condition
        display_sizes
        task_difficulties
        target_presence
        target
        """
        self.trials_per_condition = trials_per_condition
        self.display_sizes = display_sizes
        self.task_difficulties = task_difficulties
        self.target_presence = target_presence
        self.target = target

    def run(self):
        """

        Returns
        -------

    def runall(self, fvf_params=None):
        """run trials for all possible permutations of
        conditions

        Parameters
        ----------
        fvf_params : dict
            of parameters for FVFModel, where key is parameter name, and
            value is the value to pass to FVFModel when creating an instance
            of the model. Default is None, in which case defaults for model are used.

        Returns
        -------
        results : dict
            where each key is tuple representing conditions, and the
            value for each key is a list of all trials
        """
        results = {}
        for search_type in self.task_difficulties:
            print(f'Running trials for task_difficulty {search_type}')
            for display_size in self.display_sizes:
                print(f'Running trials for display size {display_size}')
                for target_present in self.target_presence:
                    if fvf_params:
                        fvf = FVFModel(**fvf_params)
                    else:
                        fvf = FVFModel()

                    trials = self._run_one_condition(fvf, search_type, display_size, target_present,
                                                     self.target, self.trials_per_condition)
                    results[(search_type, display_size, target_present)] = trials

        return results
