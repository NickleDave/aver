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

        """
        results = {}
        for search_type in self.task_difficulties:
            print(f'Running trials for task_difficulty {search_type}')
            for display_size in self.display_sizes:
                for target_present in self.target_presence:
                    fvf = FVFModel()
                    trials = []

                    for trial_num in range(num_trials):
                        search_arr = np.zeros((display_size,))
                        if target_present:
                            target_ind = np.random.choice(np.arange(display_size))
                            search_arr[target_ind] = self.target
                        trials.append(fvf.run_trial())

                    results[(search_type, display_size, target_present)] = trials

        return results
