from collections import Counter
from QPV_BB84_e.experiments.plot import Plot

import numpy as np
import math
import argparse
import os


HONEST_RATE_RESULT_TEMPLATE = './results/honest_rates_over_distance/result'
ADV_RATE_RESULT_TEMPLATE = './results/adv_rates_over_distance/result'


def calc_R_c(counter):
    if counter[True] + counter[False] > 0:
        return counter[True] / (counter[True] + counter[False])
    else:
        return 0


def calc_R_r(counter):
    if sum(counter.values()) > 0:
        return (counter[True] + counter[False]) / (sum(counter.values()) - counter['NOT_SENT'])
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


def get_results(min_dist, max_dist, interval, runs, n, alpha):
    distances = np.arange(min_dist, max_dist, interval)

    result = []

    for d in distances:
        print(d)

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

        round_within_bounds = [all(x) for x in zip(R_c_within_bounds, R_r_within_bounds)]
        result.append(Counter(round_within_bounds)[True] / len(round_within_bounds))

    return result, distances


def plot_results(result, distances, alpha, n):
    title = ''
    xlabel = 'Distance $d$ in km'
    ylabel = 'Success rate'

    stddevs = np.array([np.sqrt((p * (1 - p)) / n) for p in result])

    latex_plot = Plot(title, xlabel, ylabel)
    latex_plot.add_plot(distances, [(1 - alpha)**2] * len(distances),
                        'Honest player\'s success rate', 'green', None, True)
    latex_plot.add_plot(distances, result, 'Adversaries\' success rate', 'red', stddevs, True)
    print(latex_plot.generate_latex_code())

    latex_plot.plot_matplotlib()


def main():
    parser = argparse.ArgumentParser(description='Plot the success rate of the fidelity attack over m.')
    parser.add_argument('min_dist', type=float)
    parser.add_argument('max_dist', type=float)
    parser.add_argument('interval', type=float)
    parser.add_argument('runs', type=int)
    parser.add_argument('n', type=int)
    parser.add_argument('alpha', type=float)

    args = parser.parse_args()

    result, distances = get_results(args.min_dist, args.max_dist, args.interval, args.runs, args.n, args.alpha)

    plot_results(result, distances, args.alpha, args.n)


if __name__ == '__main__':
    main()
