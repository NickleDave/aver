import numpy as np
import matplotlib.pyplot as plt

SEARCH_TYPE_MARKERS = {
    'easy': 's',
    'medium': "^",
    'hard': "o",
}

SEARCH_TYPE_YLIMS = {
    'easy': (0, 8000),
    'medium': (0, 8000),
    'hard': (0, 8000),
}

TARGET_FILL = {
    True: 'full',
    False: 'none',
}


def mean_reaction_times(mean_RTs_all_display_sizes,
                        mean_RTs_regress_results,
                        search_types=('easy', 'medium', 'hard'),
                        display_sizes=(6, 12, 18),
                        target_present=(True, False),
                        figsize_inches=(6, 10)):
    """plots mean reaction times

    Parameters
    ----------
    mean_RTs_all_display_sizes : dict
    mean_RTs_regress_results : tuple
    search_types : tuple
    display_sizes : tuple
    target_present : tuple
    figsize_inches : tuple
    """
    rows = len(search_types)
    fig, ax = plt.subplots(rows, 1)
    for row_ind, search_type in enumerate(search_types):
        for is_target_present in target_present:
            key = (search_type, is_target_present)
            mean_RT_arr = mean_RTs_all_display_sizes[key]
            regress_results = mean_RTs_regress_results[key]
            if is_target_present:
                label = f'{search_type}, target present'
            else:
                label = f'{search_type}, target absent'
            ax[row_ind].plot(display_sizes, mean_RT_arr,
                             marker=SEARCH_TYPE_MARKERS[search_type],
                             fillstyle=TARGET_FILL[is_target_present],
                             label=label)
            slope = f'{regress_results.slope:.2f} ms/item'
            ax[row_ind].text(display_sizes[-1] + 0.2, mean_RT_arr[-1], slope)
            ax[row_ind].spines["top"].set_visible(False)
            ax[row_ind].spines["right"].set_visible(False)
            ax[row_ind].set_ylim(SEARCH_TYPE_YLIMS[search_type])
            ax[row_ind].set_xticks(display_sizes)
            ax[row_ind].legend(loc='upper left')
    fig.set_size_inches(figsize_inches)
    fig.tight_layout()


def standard_devs(std_RTs_all_display_sizes,
                  search_types=('easy', 'medium', 'hard'),
                  display_sizes=(6, 12, 18),
                  target_present=(True, False),
                  figsize_inches=(6, 10)):
    """plots standard deviations of reaction times

    Parameters
    ----------
    std_RTs_all_display_sizes : dict
    search_types : tuple
    display_sizes : tuple
    target_present : tuple
    figsize_inches : tuple
    """
    rows = len(search_types)
    fig, ax = plt.subplots(rows, 1)
    for row_ind, search_type in enumerate(search_types):
        for is_target_present in target_present:
            key = (search_type, is_target_present)
            std_RT_arr = std_RTs_all_display_sizes[key]
            if is_target_present:
                label = f'{search_type}, target present'
            else:
                label = f'{search_type}, target absent'
            ax[row_ind].plot(display_sizes, std_RT_arr,
                             linestyle='None',
                             marker=SEARCH_TYPE_MARKERS[search_type],
                             fillstyle=TARGET_FILL[is_target_present],
                             label=label)
            ax[row_ind].spines["top"].set_visible(False)
            ax[row_ind].spines["right"].set_visible(False)
            ax[row_ind].set_ylim(SEARCH_TYPE_YLIMS[search_type])
            ax[row_ind].set_xticks(display_sizes)
            ax[row_ind].legend(loc='upper left')
    fig.set_size_inches(figsize_inches)
    fig.tight_layout()
