from mpi4py import MPI
from QPV_BB84_e.attacks.fidelity_attack.attack import Attack

import os
import numpy as np
import argparse

RESULTS_FILE_TEMPLATE = './results/adversaries_results_over_distance/result'


def get_results(min_dist, max_dist, interval, size, rank):
    dist_per_inst = (max_dist - min_dist) / size
    my_min_dist = round(rank * dist_per_inst + min_dist, 1)
    my_max_dist = round(my_min_dist + dist_per_inst, 1)

    n = 1000
    m = 50
    v_pos = 0
    delta_p = .0001

    attack_runs = 1000

    distances = np.arange(my_min_dist, my_max_dist, interval)

    exclfile = open('excl_adversaries_results_over_distance.txt', 'r')
    excllist = ['./results/adversaries_results_over_distance/' + x.rstrip() for x in exclfile.readlines()]
    exclfile.close()

    for d in distances:
        print(f'Distance: {d:.1f}')

        for i in range(attack_runs):
            filename = f'{RESULTS_FILE_TEMPLATE}_{d:.1f}_{i}.npz'

            if filename in excllist or os.path.exists(filename):
                continue

            attack = Attack(n, m, -d, d, -d + delta_p, d - delta_p, v_pos)

            stats, alice_data, bob_data = attack.run()

            np.savez(filename, params=[d, n, m, v_pos, delta_p, attack_runs],
                     alice_data=alice_data, bob_data=bob_data, stats=stats)


def main():
    parser = argparse.ArgumentParser(description='Get the adversaries\' results over distance.')
    parser.add_argument('min_dist', type=float)
    parser.add_argument('max_dist', type=float)
    parser.add_argument('interval', type=float)

    args = parser.parse_args()

    comm = MPI.COMM_WORLD

    get_results(args.min_dist, args.max_dist, args.interval, comm.Get_size(), comm.Get_rank())

    comm.Barrier()

    if comm.Get_rank() == 0:
        print('done')


if __name__ == '__main__':
    main()
