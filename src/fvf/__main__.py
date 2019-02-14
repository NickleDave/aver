import argparse
import pickle
import json

import numpy as np

from .simulator import Simulator


def get_parser():
    parser = argparse.ArgumentParser(description='Run simulations of fixation-based framework.')
    return parser


def main():
    parser = get_parser()
    parser.parse_args()
    sim = Simulator()
    results = sim.runall()

    with open('./results.pickle', 'wb') as fp:
        pickle.dump(results, fp)

    reaction_times = {}
    for (search_type, display_size, target_present), trials in results.items():
        RTs = [trial.reaction_time for trial in trials]
        key = ', '.join([search_type, str(display_size), str(target_present)])
        reaction_times[key] = RTs
    with open('./reaction_times.json', 'w') as fp:
        json.dump(reaction_times, fp)


if __name__ == '__main__':
    main()
