from mpi4py import MPI
from QPV_BB84_e.attacks.fidelity_attack.attack import Attack

import os
import numpy as np
import argparse

RESULTS_FILE_TEMPLATE = './results/adversaries_results_over_m/result'


def get_results(min_m, max_m, size, rank):
    m_per_inst = (max_m - min_m) / size
    my_min_m = round(rank * m_per_inst + min_m)
    my_max_m = round(my_min_m + m_per_inst)

    n = 1000
    d = 0.1
    v_pos = 0
    delta_p = .00001

    attack_runs = 1000

    ms = np.arange(my_min_m, my_max_m)

    exclfile = open('excl_adversaries_results_over_m.txt', 'r')
    excllist = ['./results/adversaries_results_over_m/' + x.rstrip() for x in exclfile.readlines()]
    exclfile.close()

    for m in ms:
        print('m:', m)

        for i in range(attack_runs):
            filename = f'{RESULTS_FILE_TEMPLATE}_{m:.1f}_{i}.npz'

            if filename in excllist or os.path.exists(filename):
                continue

            stats, alice_data, bob_data = Attack(n, m, -d, d, -d + delta_p, d - delta_p, v_pos).run()

            np.savez(filename, params=[d, n, m, v_pos, delta_p, attack_runs],
                     alice_data=alice_data, bob_data=bob_data, stats=stats)


def main():
    parser = argparse.ArgumentParser(description='Get the adversaries\' results over m.')
    parser.add_argument('min_m', type=float)
    parser.add_argument('max_m', type=float)

    args = parser.parse_args()

    comm = MPI.COMM_WORLD

    get_results(args.min_m, args.max_m, comm.Get_size(), comm.Get_rank())

    if comm.Get_rank() == 0:
        print('done')

    comm.Barrier()


if __name__ == '__main__':
    main()
