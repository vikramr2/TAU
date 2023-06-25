# modification of syt3's run_leiden.py that
# sets n_iterations to 5 and seed to 1234
# 2/19/2023

import tau
import igraph
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Script for running TAU.')

    parser.add_argument(
        '-i', metavar='ip_net', type=str, required=True,
        help='input network edge-list path'
        )
    parser.add_argument(
        '--size', type=int, default=60, 
        help='size of population; default is 60'
        )
    parser.add_argument(
        '--workers', type=int, default=-1,
        help='number of workers; default is number of available CPUs'
        )
    parser.add_argument(
        '--max_generations', type=int, default=500, 
        help='maximum number of generations to run; default is 500'
        )
    parser.add_argument(
        '--quiet', type=bool, default=False,
        help='silence output to stdout'
        )
    parser.add_argument(
        '-o', metavar='output', type=str, required=True,
        help='output membership path'
        )
    args = parser.parse_args()

    net = igraph.Graph.Load(args.i, format='edgelist', directed=False)
    partition = tau.tau(net, args.size, args.workers, args.max_generations, args.quiet)
    
    with open(args.o, "w") as f:
        for n, m in enumerate(partition):
            f.write(f"{n}\t{m}\n")