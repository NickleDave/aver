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


def mean_reaction_times(mean_rts_to_plot,
                        search_types=('easy', 'medium', 'hard'),
                        display_sizes=(6, 12, 18),
                        target_present=(True, False),
                        figsize_inches=(10, 5)):
    """plots mean reaction times

    Parameters
    ----------
    mean_rts_to_plot : dict
    search_types : tuple
    display_sizes : tuple
    target_present : tuple
    figsize_inches : tuple
    """
    cols = len(search_types)
    fig, ax = plt.subplots(1, cols)
    for col_ind, search_type in enumerate(search_types):
        for is_target_present in target_present:
            x, y = mean_rts_to_plot[(search_type, is_target_present)]
            if is_target_present:
                label = f'{search_type}, target present'
            else:
                label = f'{search_type}, target absent'
            ax[col_ind].plot(x, y, marker=SEARCH_TYPE_MARKERS[search_type],
                             fillstyle=TARGET_FILL[is_target_present],
                             label=label)
            ax[col_ind].set_ylim(SEARCH_TYPE_YLIMS[search_type])
            ax[col_ind].set_xticks(display_sizes)
            ax[col_ind].legend(loc='upper left')
    fig.set_size_inches(figsize_inches)
    fig.tight_layout()

