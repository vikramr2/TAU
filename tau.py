import numpy as np
import networkx as nx
import igraph as ig
from sklearn.metrics.cluster import pair_confusion_matrix
import time
from multiprocessing import Pool
import itertools
import os
import random
from partition import Partition

class TAU:
    def __init__(self):
        # globals and hyper-parameters
        self.G_ig = None
        self.POPULATION_SIZE = 60
        self.N_WORKERS = 60
        self.MAX_GENERATIONS = 500
        self.N_IMMIGRANTS = -1
        self.N_ELITE = -1
        self.SELECTION_POWER = 5
        self.PROBS = []
        self.pop = []
        self.p_elite = .1
        self.p_immigrants = .15
        self.stopping_criterion_generations = 10
        self.stopping_criterion_jaccard = .98
        self.elite_similarity_threshold = .9
        
    def flip_coin(self):
        return random.uniform(0, 1) > .5


    def load_graph(self, path):
        nx_graph = nx.read_adjlist(path)
        mapping = dict(zip(nx_graph.nodes(), range(nx_graph.number_of_nodes())))
        nx_graph = nx.relabel_nodes(nx_graph, mapping)
        ig_graph = ig.Graph(len(nx_graph), list(zip(*list(zip(*nx.to_edgelist(nx_graph)))[:2])))
        return ig_graph


    def compute_partition_similarity(self, partition_a, partition_b):
        conf = pair_confusion_matrix(partition_a, partition_b)
        b, d, c, a = conf.flatten()
        jac = a/(a+c+d)
        return jac


    def overlap(self, partition_memberships):
        consensus_partition = partition_memberships[0]
        n_nodes = len(consensus_partition)
        for i, partition in enumerate(partition_memberships[1:]):
            partition = partition
            cluster_id_mapping = {}
            c = 0
            for node_id in range(n_nodes):
                cur_pair = (consensus_partition[node_id], partition[node_id])
                if cur_pair not in cluster_id_mapping:
                    cluster_id_mapping[cur_pair] = c
                    c += 1
                consensus_partition[node_id] = cluster_id_mapping[cur_pair]
        return consensus_partition


    def create_population(self, size_of_population):
        sample_fraction_per_indiv = np.random.uniform(.2, .9, size=size_of_population)
        params = [sample_fraction for sample_fraction in sample_fraction_per_indiv]
        pool = Pool(min(size_of_population, self.N_WORKERS))
        results = [pool.apply_async(Partition, (self, sample_fraction,)) for sample_fraction in params]
        pool.close()
        pool.join()
        population = [x.get() for x in results]
        return population


    def get_probabilities(self, values):
        p = []
        values = np.max(values) + 1 - values
        denom = np.sum(values ** self.SELECTION_POWER)
        for value in values:
            p.append(value ** self.SELECTION_POWER / denom)
        return p


    def single_crossover(self, idx1, idx2):
        partitions_overlap = self.overlap([self.pop[idx1].membership, self.pop[idx2].membership])
        single_offspring = Partition(tau_instance=self, init_partition=partitions_overlap)
        return single_offspring


    def pop_crossover_and_immigration(self, n_offspring):
        idx_to_cross = []
        as_is_offspring = []
        for i in range(n_offspring):
            idx1, idx2 = np.random.choice(len(self.pop), size=2, replace=False, p=self.PROBS)
            if self.flip_coin():
                idx_to_cross.append([idx1, idx2])
            else:
                as_is_offspring.append(self.pop[idx1])
        pool = Pool(self.N_WORKERS)
        results = [pool.apply_async(self.single_crossover, (idx1, idx2)) for idx1, idx2 in idx_to_cross]
        pool.close()
        pool.join()
        crossed_offspring = [x.get() for x in results]
        offspring = crossed_offspring + as_is_offspring

        immigrants = self.create_population(size_of_population=self.N_IMMIGRANTS)

        return offspring, immigrants


    def compute_partition_similarity_by_pop_idx(self, idx1, idx2):
        conf = pair_confusion_matrix(self.pop[idx1].membership, self.pop[idx2].membership)
        b, d, c, a = conf.flatten()
        jac = a/(a+c+d)
        return jac


    def elitist_selection_helper(self, combinations):
        assert 0 < len(combinations) <= self.N_WORKERS
        pool = Pool(len(combinations))
        results = [pool.apply_async(self.compute_partition_similarity_by_pop_idx, (idx1, idx2))
                for idx1, idx2 in combinations]
        pool.close()
        pool.join()
        similarities = [x.get() for x in results]
        similarities_dict = {tuple(sorted((idx1, idx2))): similarities[i] for i, (idx1, idx2) in enumerate(combinations)}
        return similarities_dict


    def get_batch_size_for_n(self, k=0):
        index_batch_size = 0
        used_workers = 0
        while used_workers < self.N_WORKERS:
            index_batch_size += 1
            used_workers = (index_batch_size ** 2 - index_batch_size) / 2 + index_batch_size * k
        return index_batch_size-1


    def elitist_selection(self, similarity_threshold):
        index_batch_size = self.get_batch_size_for_n()
        combinations = list(itertools.combinations(range(index_batch_size), 2))
        similarities_between_solutions = self.elitist_selection_helper(combinations)
        highest_idx_considered = index_batch_size-1
        elite_idx = [0]
        idx = 1
        cnt_cycles = 1
        while len(elite_idx) < self.N_ELITE and idx < len(self.pop):
            if cnt_cycles == 2:
                n_remaining = self.N_ELITE - len(elite_idx)
                elite_idx += list(np.random.choice(np.arange(idx, len(self.pop)), size=n_remaining, replace=False))
                break
            elite_flag = True
            for elite_indiv_idx in elite_idx:
                if (elite_indiv_idx, idx) not in similarities_between_solutions:
                    cnt_cycles += 1
                    index_batch_size = self.get_batch_size_for_n(len(elite_idx))
                    size_of_examined_range = index_batch_size
                    new_idx_range = range(highest_idx_considered, highest_idx_considered+size_of_examined_range)
                    new_combinations = list(itertools.product(new_idx_range, elite_idx))
                    new_combinations += list(itertools.combinations(new_idx_range, 2))
                    highest_idx_considered = new_idx_range[-1]
                    similarities_between_solutions.update(self.elitist_selection_helper(new_combinations))
                jac = similarities_between_solutions[elite_indiv_idx, idx]
                if jac > similarity_threshold:
                    elite_flag = False
                    break
            if elite_flag:
                elite_idx.append(idx)
            idx += 1

        return elite_idx

    def find_partition(self):
        last_best_memb = []
        best_modularity_per_generation = []
        cnt_convergence = 0

        # Population initialization
        self.pop = self.create_population(size_of_population=self.POPULATION_SIZE)

        for generation_i in range(1, self.MAX_GENERATIONS+1):
            start_time = time.time()

            # Population optimization
            pool = Pool(self.N_WORKERS)
            results = [pool.apply_async(indiv.optimize, ()) for indiv in self.pop]
            pool.close()
            pool.join()
            self.pop = [x.get() for x in results]

            pop_fit = [indiv.fitness for indiv in self.pop]
            best_score = np.max(pop_fit)
            best_modularity_per_generation.append(best_score)
            best_indiv = self.pop[np.argmax(pop_fit)]

            # stopping criteria related
            if last_best_memb:
                sim_to_last_best = self.compute_partition_similarity(best_indiv.membership, last_best_memb)
                if sim_to_last_best > self.stopping_criterion_jaccard:
                    cnt_convergence += 1
                else:
                    cnt_convergence = 0
            last_best_memb = best_indiv.membership
            if cnt_convergence == self.stopping_criterion_generations:
                break
            pop_rank_by_fitness = np.argsort(pop_fit)[::-1]
            self.pop = [self.pop[i] for i in pop_rank_by_fitness]
            if generation_i == self.MAX_GENERATIONS:
                break

            # elitist selection
            elite_idx = self.elitist_selection(self.elite_similarity_threshold)
            elite = [self.pop[i] for i in elite_idx]

            # crossover, immigration
            offspring, immigrants = self.pop_crossover_and_immigration(n_offspring=self.POPULATION_SIZE-self.N_ELITE-self.N_IMMIGRANTS)

            # mutation
            pool = Pool(min(len(offspring), self.N_WORKERS))
            results = [pool.apply_async(indiv.mutate, ()) for indiv in offspring]
            pool.close()
            pool.join()
            offspring = [x.get() for x in results]

            print(f'Generation {generation_i} Top fitness: {np.round(best_score, 6)}; Average fitness: '
                f'{np.round(np.mean(pop_fit), 6)}; Time per generation: {np.round(time.time() - start_time, 2)}; '
                f'convergence: {cnt_convergence}')
            self.pop = elite + offspring + immigrants

        # return best and modularity history
        return self.pop[0], best_modularity_per_generation

    '''
    if __name__ == "__main__":
        # parse script parameters
        parser = argparse.ArgumentParser(description='TAU')
        # general parameters
        parser.add_argument('--graph', type=str, help='path to graph file; supports adjacency list format')
        parser.add_argument('--size', type=int, default=60, help='size of population; default is 60')
        parser.add_argument('--workers', type=int, default=-1, help='number of workers; '
                                                                    'default is number of available CPUs')
        parser.add_argument('--max_generations', type=int, default=500, help='maximum number of generations to run;'
                                                                            ' default is 500')
        args = parser.parse_args()

        # set global variable values
        g = load_graph(args.graph)
    '''

    def tau(self, g, size=60, workers=-1, max_generations=500):
        population_size = max(10, size)
        cpus = os.cpu_count()
        self.N_WORKERS = min(cpus, population_size) if workers == -1 else np.min([cpus, population_size, workers])
        self.PROBS = self.get_probabilities(np.arange(population_size))
        self.N_ELITE, self.N_IMMIGRANTS = int(self.p_elite * population_size), int(self.p_immigrants * population_size)
        self.G_ig = g
        self.POPULATION_SIZE = population_size
        self.MAX_GENERATIONS = max_generations

        print(f'Main parameter values: pop_size={self.POPULATION_SIZE}, workers={self.N_WORKERS}, max_generations={self.MAX_GENERATIONS}')

        best_partition, mod_history = self.find_partition()
        
        #np.save(f'TAU_partition_{graph}.npy', best_partition.membership)

        return best_partition.membership
