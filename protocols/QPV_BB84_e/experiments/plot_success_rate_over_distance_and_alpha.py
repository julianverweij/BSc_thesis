from collections import Counter, defaultdict
from QPV_BB84_e.experiments.plot import Plot

import numpy as np
import argparse
import os
import math


HONEST_RATE_RESULT_TEMPLATE = './results/honest_rates_over_distance/result'
ADV_RATE_RESULT_TEMPLATE = './results/adv_rates_over_distance/result'


def calc_R_c(counter):
    if counter[True] + counter[False] > 0:
        return counter[True] / (counter[True] + counter[False])
    else:
        return 0


def calc_R_r(counter):
    if sum(counter.values()) > 0:
        return (counter[True] + counter[False]) / sum(counter.values())
    else:
        return 0


def percentile(values, alpha):
    values = sorted(values)
    index = (len(values) - 1) * alpha

    lower = int(math.floor(index))
    upper = int(math.ceil(index))

    return (1 - (index % 1)) * values[lower] + (index % 1) * values[upper]


def ci_bounds(data, alpha, mode='left-reject'):
    if mode == 'left-reject':
        return percentile(data, alpha), None
    elif mode == 'right-reject':
        return None, percentile(data, 1 - alpha)

    return percentile(data, alpha * 0.5), percentile(data, 1 - alpha * 0.5)


def within_bounds(x, bounds):
    return x > bounds[0]


def get_results(min_dist, max_dist, interval, runs, n, min_alpha, max_alpha, alpha_interval):
    distances = np.arange(min_dist, max_dist, interval)
    alphas = np.arange(min_alpha, max_alpha, alpha_interval)

    result = defaultdict(list)

    for alpha in alphas:
        for d in distances:

            HONEST_RESULTS_FILE_TEMPLATE = f'./results/honest_results_over_distance/result_{d:.1f}'
            ADV_RESULTS_FILE_TEMPLATE = f'./results/adversaries_results_over_distance/result_{d:.1f}'

            R_c_honest = []
            R_r_honest = []

            R_c_adv = []
            R_r_adv = []

            honest_rate_filename = f'{HONEST_RATE_RESULT_TEMPLATE}_{d:.1f}.npz'
            adv_rate_filename = f'{ADV_RATE_RESULT_TEMPLATE}_{d:.1f}.npz'

            if not (os.path.exists(honest_rate_filename) or os.path.exists(adv_rate_filename)):
                for run in range(runs):
                    honest_results = np.load(f'{HONEST_RESULTS_FILE_TEMPLATE}_{run}.npz', allow_pickle=True)
                    counter = Counter(honest_results['alice_data'].item()['r_i'])

                    R_c_honest.append(calc_R_c(counter))
                    R_r_honest.append(calc_R_r(Counter(honest_results['alice_data'].item()['r_i'][:n])))

                    adv_results = np.load(f'{ADV_RESULTS_FILE_TEMPLATE}_{run}.npz', allow_pickle=True)
                    counter = Counter(adv_results['alice_data'].item()['r_i'])

                    R_c_adv.append(calc_R_c(counter))
                    R_r_adv.append(calc_R_r(Counter(adv_results['alice_data'].item()['r_i'][:n])))

                np.savez(honest_rate_filename, R_c_honest=R_c_honest, R_r_honest=R_r_honest)
                np.savez(adv_rate_filename, R_c_adv=R_c_adv, R_r_adv=R_r_adv)
            else:
                honest_results = np.load(honest_rate_filename)
                R_c_honest = honest_results['R_c_honest']
                R_r_honest = honest_results['R_r_honest']

                adv_results = np.load(adv_rate_filename)
                R_c_adv = adv_results['R_c_adv']
                R_r_adv = adv_results['R_r_adv']

            ci_bounds_R_c = ci_bounds(R_c_honest, alpha)
            ci_bounds_R_r = ci_bounds(R_r_honest, alpha)

            R_c_within_bounds = [within_bounds(x, ci_bounds_R_c) for x in R_c_adv]
            R_r_within_bounds = [within_bounds(x, ci_bounds_R_r) for x in R_r_adv]

            print(Counter(R_r_within_bounds))

            round_within_bounds = [all(x) for x in zip(R_c_within_bounds, R_r_within_bounds)]
            result[alpha].append(Counter(round_within_bounds)[True] / len(round_within_bounds))

    return result, distances


def plot_results(result, distances):
    print(result.values())


def main():
    parser = argparse.ArgumentParser(description='Plot the success rate of the fidelity attack over m.')
    parser.add_argument('min_dist', type=float)
    parser.add_argument('max_dist', type=float)
    parser.add_argument('interval', type=float)
    parser.add_argument('runs', type=int)
    parser.add_argument('n', type=int)
    parser.add_argument('min_alpha', type=float)
    parser.add_argument('max_alpha', type=float)
    parser.add_argument('alpha_interval', type=float)

    args = parser.parse_args()

    result, distances = get_results(args.min_dist, args.max_dist, args.interval, args.runs, args.n, args.min_alpha,
                                    args.max_alpha, args.alpha_interval)

    plot_results(result, distances)


if __name__ == '__main__':
    main()
