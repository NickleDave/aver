import matplotlib.pyplot as plt
import numpy as np

SEARCH_TYPE_MARKERS = {
    'easy': 's',
    'medium': "^",
    'hard': "o",
}

MN_RT_SEARCH_TYPE_YLIMS = {
    'easy': (0, 8000),
    'medium': (0, 8000),
    'hard': (0, 8000),
}

STD_RT_SEARCH_TYPE_YLIMS = {
    'easy': (0, 2500),
    'medium': (0, 2500),
    'hard': (0, 2500),
}

TARGET_FILL = {
    True: 'full',
    False: 'none',
}

NUM_FIX_SEARCH_TYPE_YLIMS = {
    'easy': (0, 35),
    'medium': (0, 35),
    'hard': (0, 35),
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
            ax[row_ind].set_ylim(MN_RT_SEARCH_TYPE_YLIMS[search_type])
            ax[row_ind].set_xticks(display_sizes)
            ax[row_ind].legend(loc='upper left')
            ax[row_ind].set_ylabel('reaction time (ms)')
    ax[-1].set_xlabel('display size (number of items)')
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
            ax[row_ind].set_ylim(STD_RT_SEARCH_TYPE_YLIMS[search_type])
            ax[row_ind].set_xticks(display_sizes)
            ax[row_ind].legend(loc='upper left')
            ax[row_ind].set_ylabel('reaction time (ms)')
    ax[-1].set_xlabel('display size (number of items)')
    fig.set_size_inches(figsize_inches)
    fig.tight_layout()


def mean_num_fixations(mean_num_fixations_all_display_sizes,
                       search_types=('easy', 'medium', 'hard'),
                       display_sizes=(6, 12, 18),
                       target_present=(True, False),
                       figsize_inches=(6, 10)):
    """plots mean reaction times

    Parameters
    ----------
    mean_num_fixations_all_display_sizes : dict
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
            mean_nf_arr = mean_num_fixations_all_display_sizes[key]
            if is_target_present:
                label = f'{search_type}, target present'
            else:
                label = f'{search_type}, target absent'
            ax[row_ind].plot(display_sizes, mean_nf_arr,
                             marker=SEARCH_TYPE_MARKERS[search_type],
                             fillstyle=TARGET_FILL[is_target_present],
                             label=label)
            ax[row_ind].spines["top"].set_visible(False)
            ax[row_ind].spines["right"].set_visible(False)
            ax[row_ind].set_ylim(NUM_FIX_SEARCH_TYPE_YLIMS[search_type])
            ax[row_ind].set_xticks(display_sizes)
            ax[row_ind].legend(loc='upper left')
            ax[row_ind].set_ylabel('number of fixations')
    ax[-1].set_xlabel('display size (number of items)')
    fig.set_size_inches(figsize_inches)
    fig.tight_layout()

RT_DISTRIB_XLIMS = {
    'easy': (0, 2000),
    'medium': (0, 6000),
    'hard': (0, 12000),
}


def reaction_times_distrib(RTs_by_condition,
                           search_types=('easy', 'medium', 'hard'),
                           display_sizes=(6, 12, 18),
                           target_present=(True, False),
                           figsize_inches=(6, 10)):
    """plot distribution of reaction times

    Parameters
    ----------
    RTs_by_condition : dict
    search_types : tuple
    display_sizes : tuple
    target_present : tuple
    figsize_inches : tuple
    """
    RTs_all_display_sizes = {}
    # do yet more munging before plot
    for search_type in search_types:
        for is_target_present in target_present:
            RT_arrs = []
            for display_size in display_sizes:
                condition_tup = tuple([search_type, display_size, is_target_present])
                RT_arr = RTs_by_condition[condition_tup]
                RT_arrs.append(RT_arr)
            key = tuple([search_type, is_target_present])
            RTs_all_display_sizes[key] = RT_arrs

    rows = len(search_types)
    fig, ax = plt.subplots(rows, 1)
    for row_ind, search_type in enumerate(search_types):
        for is_target_present in target_present:
            key = (search_type, is_target_present)
            RT_arrs = RTs_all_display_sizes[key]
            for display_size, RT_arr in zip(display_sizes, RT_arrs):
                if is_target_present:
                    label = f'{search_type}, {display_size} items, target present'
                    linestyle = '-'
                else:
                    label = f'{search_type}, {display_size} items, target absent'
                    linestyle = '--'
                binedges = np.arange(0, 12001, 250)
                counts = np.histogram(RT_arr, bins=binedges)[0]
                # Normalize using total number of observations, which is what I
                # think both Hulleman Olivers + Wolfe do.
                # Note that we would need to use density=True if we wanted to
                # normalize such that we model a probability density function.
                counts = counts / RT_arr.shape[0]
                ax[row_ind].plot(binedges[:-1], counts,
                                 linestyle=linestyle,
                                 label=label)
            ax[row_ind].spines["top"].set_visible(False)
            ax[row_ind].spines["right"].set_visible(False)
            ax[row_ind].set_ylim([0, 1.1])
            #ax[row_ind].set_xlim(RT_DISTRIB_XLIMS[search_type])
            ax[row_ind].legend(loc='upper right')
            ax[row_ind].set_ylabel('Frequency')
    ax[-1].set_xlabel('reaction times (250 ms bins)')
    fig.set_size_inches(figsize_inches)
    fig.tight_layout()
