from __future__ import print_function

from collections import Counter
from itertools import izip

import numpy as np
from numpy import random
import networkx as nx

class ErgodicDiffusion():

    def __init__(self,G,max_ratio_visited=None,max_diff_time=None,N_walker=10000,initial_seeds=None,record_jump_distance=False):
        
        self.N_walker = N_walker

        self.G = nx.convert_node_labels_to_integers(G)
        self.N_nodes = self.G.number_of_nodes()
        self.m = self.G.number_of_edges()

        self.k = np.array( [ self.G.degree(node) for node in xrange(self.N_nodes) ],dtype=np.float64 )
        self.p_expected = 0.5 * self.k / self.m


        if initial_seeds is None:
            initial_node = random.randint(self.N_nodes)
            self.initial_nodes = [ initial_node for walker in xrange(self.N_walkter) ]
        elif initial_seeds == "random":
            self.initial_nodes = random.randint(self.N_nodes,size=self.N_walker)
        elif hasattr(initial_seeds,"__len__"):
            self.initial_nodes = initial_seeds

        self.record_jump_distance = record_jump_distance

        self.rewind()


    def rewind(self):

        self.visited_nodes = [ set() for walker in xrange(self.N_walkter) ]
        for walker in xrange(self.N_walker):
            self.visited_nodes[walker].insert(self.initial_nodes[walker])

        if self.record_jump_distance:
            self.jump_distances = []

        self.remaining_walkers = set(range(self.xrange()))

        self.current_nodes = initial_nodes


    def timestep(self):

        # iterate over all walkers
        for walker in self.remaining_walkers:
            node = self.current_nodes[walker]

            neighbors = self.G.neighbors(node)
            k = len(neighbors)
            neigh = neighbors[random.randint(k)]

            if self.record_jump_distance:
                distance = abs(neighbors[neigh]-node)
                if distance > self.N_nodes/2.:
                    distance = abs(distance-self.N_nodes)
                self.jump_distances.append( distance )

            self.current_nodes[walker] = neigh
            self.visited_nodes[walker].insert(neigh)

    #def simulation(self,tmax):

    #    for t in range(tmax):
    #        self.timestep()

    #    return np.array(self.dist_time), np.array(self.std_time), np.array(self.localization_time)


    def simulate_till_max_time(self,tmax,get_trajectories=False):

        trajectories = np.zeros((self.N_walker,tmax),dtype=int)

        nmax = self.N_nodes

        for t in xrange(tmax):
            self.timestep()

            finished_walkers = []
            for walker in xrange(self.N_walker):
                trajectories[walker,t] = len(self.visited_nodes[walker])
                if trajectories[walker,t] == nmax:
                    finished_walkers.append(walker)

            self.remaining_walkers -= finished_walkers

        if get_trajectories:
            return trajectories
        else:
            return trajectories[:,-1]



if __name__ == "__main__":

    import pylab as pl
    import seaborn as sns
    from networkprops import networkprops as nprops
    import mhrn

    from scipy.optimize import curve_fit

    sns.set_style("white")

    N = 10
    B = 8
    L = 3
    N = B**L
    k = 10
    xi = 0.25
    p_ER = float(k)/(N-1)
    #G = nx.fast_gnp_random_graph(N,p_ER)

    #G = mhrn.fast_mhr_graph(B,L,k,xi)
    G = mhrn.continuous_hierarchical_graph(B**L,k,np.log(xi)/np.log(B),redistribute_probability=False)

