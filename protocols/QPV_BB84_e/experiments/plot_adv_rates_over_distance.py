from collections import defaultdict, Counter
from QPV_BB84_e.experiments.plot import Plot

import numpy as np
import argparse


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


def within_bounds(x, bounds):
    return x > bounds[0]


def get_results(min_dist, max_dist, interval, n, runs):
    distances = np.arange(min_dist, max_dist, interval)

    result = defaultdict(list)

    for d in distances:
        print(f'{d:.1f}')
        RESULTS_FILE_TEMPLATE = f'./results/adversaries_results_over_distance/result_{d:.1f}'

        R_c_honest = []
        R_r_honest = []

        for run in range(runs):
            honest_results = np.load(f'{RESULTS_FILE_TEMPLATE}_{run}.npz', allow_pickle=True)
            counter = Counter(honest_results['alice_data'].item()['r_i'])

            R_c_honest.append(calc_R_c(counter))
            R_r_honest.append(calc_R_r(Counter(honest_results['alice_data'].item()['r_i'][:n])))

        result['R_c'].append(R_c_honest)
        result['R_r'].append(R_r_honest)

    return result, distances


def plot_results(ratios, distances):
    R_c_means = np.mean(ratios['R_c'], axis=1)
    R_c_stddevs = np.std(ratios['R_c'], axis=1)

    title = ''
    xlabel = 'Distance $d$ in km'
    ylabel = 'Correctness rate $R_c$'

    latex_plot = Plot(title, xlabel, ylabel)
    latex_plot.add_plot(distances, R_c_means, 'The adversaries\' correctness rate', 'red', R_c_stddevs, True)

    print(latex_plot.generate_latex_code())

    latex_plot.plot_matplotlib()

    R_r_means = np.mean(ratios['R_r'], axis=1)
    R_r_stddevs = np.std(ratios['R_r'], axis=1)

    title = ''
    xlabel = 'Distance $d$ in km'
    ylabel = 'Reporting rate $R_r$'

    latex_plot = Plot(title, xlabel, ylabel)
    latex_plot.add_plot(distances, R_r_means, 'The adversaries\' reporting rate', 'red', R_r_stddevs, True)

    print(latex_plot.generate_latex_code())

    latex_plot.plot_matplotlib()


def main():
    parser = argparse.ArgumentParser(description='Plot the adversaries\' rates over distance.')
    parser.add_argument('min_dist', type=float)
    parser.add_argument('max_dist', type=float)
    parser.add_argument('interval', type=float)
    parser.add_argument('n', type=int)
    parser.add_argument('runs', type=int)

    args = parser.parse_args()

    result, distances = get_results(args.min_dist, args.max_dist, args.interval, args.n, args.runs)

    plot_results(result, distances)


if __name__ == '__main__':
    main()
