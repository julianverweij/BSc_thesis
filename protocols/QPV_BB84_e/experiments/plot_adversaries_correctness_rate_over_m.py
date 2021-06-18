from collections import Counter
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
        return (counter[True] + counter[False]) / sum(counter.values())
    else:
        return 0


def get_results(min_m, max_m, runs):
    ms = range(min_m, max_m)

    result = []

    for m in ms:
        ADV_RESULTS_FILE_TEMPLATE = f'./results/adversaries_results_over_m/result_{m:.1f}'

        R_c_adv = []
        R_r_adv = []

        for run in range(runs):
            print(run)
            adv_results = np.load(f'{ADV_RESULTS_FILE_TEMPLATE}_{run}.npz', allow_pickle=True)
            counter = Counter(adv_results['alice_data'].item()['r_i'])

            R_c_adv.append(calc_R_c(counter))
            R_r_adv.append(calc_R_r(counter))

        result.append(R_c_adv)

    return result, ms


def plot_results(result, ms):
    means = np.mean(result, axis=1)
    stddevs = np.std(result, axis=1)

    title = """The correctness rate $R_c$ of the adversaries in the fidelity attack over m. $n = 10^3$."""
    xlabel = '$m$'
    ylabel = 'Correctness rate $R_c$'

    latex_plot = Plot(title, xlabel, ylabel)
    latex_plot.add_plot(ms, means, 'Adversaries\' correctness rate', 'red', stddevs, True)
    print(latex_plot.generate_latex_code())

    latex_plot.plot_matplotlib()


def main():
    parser = argparse.ArgumentParser(description='Plot the success rate of the fidelity attack over m.')
    parser.add_argument('min_m', type=int)
    parser.add_argument('max_m', type=int)
    parser.add_argument('runs', type=int)

    args = parser.parse_args()

    result, ms = get_results(args.min_m, args.max_m, args.runs)

    plot_results(result, ms)


if __name__ == '__main__':
    main()
