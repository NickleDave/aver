import argparse
import pickle
import json
import os
import logging

import numpy as np

from .simulator import Simulator


def get_parser():
    """returns instance of ArgumentParser, used for command-line interface by main function below"""
    parser = argparse.ArgumentParser(description='Run simulations of fixation-based framework.')
    parser.add_argument('results_dir', type=str, help='name of directory where results should be saved')
    parser.add_argument('loglevel', nargs='?', default='INFO', choices=('INFO', 'DEBUG', 'WARNING'),
                        help="logging level (defaults to logging.INFO)")
    parser.add_argument('seed', nargs='?', default=42, type=int,
                        help="seed for numpy random generator, default is 42")
    return parser


def main():
    """main function run from command line"""
    parser = get_parser()
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.loglevel))
    logger = logging.getLogger(__name__)

    if not os.path.isdir(args.results_dir):
        logger.info(f'making directory {args.results_dir}')
        os.makedirs(args.results_dir)

    logger.info('starting simulation')
    logger.info(f'will set numpy RNG seed to {args.seed}')
    sim = Simulator(seed=args.seed)
    results = sim.runall()

    results_pkl = os.path.join(args.results_dir, 'results.pickle')
    logger.info(f'saving results in {results_pkl}')
    with open(results_pkl, 'wb') as fp:
        pickle.dump(results, fp)

    # munge reaction times
    logger.info('getting just reaction times out of {results_pkl}')
    reaction_times = {}
    for (search_type, display_size, target_present), trials in results.items():
        RTs = [trial.reaction_time for trial in trials]
        key = ', '.join([search_type, str(display_size), str(target_present)])
        reaction_times[key] = RTs
    rt_json = os.path.join(args.results_dir, 'reaction_times.json')
    logger.info(f'saving reaction times in {rt_json}')
    with open(rt_json, 'w') as fp:
        json.dump(reaction_times, fp)


if __name__ == '__main__':
    main()
