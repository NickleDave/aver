import argparse

from .simulator import Simulator


def get_parser():
    parser = argparse.ArgumentParser(description='Run simulations of fixation-based framework.')
    return parser


def main():
    parser = get_parser()
    parser.parse_args()
    sim = Simulator()
    sim.run()


if __name__ == '__main__':
    main()
