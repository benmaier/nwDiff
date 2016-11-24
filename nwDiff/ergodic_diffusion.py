from __future__ import print_function

from collections import Counter
from itertools import izip

import numpy as np
from numpy import random
import networkx as nx

class ErgodicDiffusion():

    def __init__(self,G,N_walker=10000,initial_seeds=None,record_jump_distance=False):
        
        self.N_walker = N_walker

        self.G = nx.convert_node_labels_to_integers(G)
        self.N_nodes = self.G.number_of_nodes()
        self.m = self.G.number_of_edges()

        self.k = np.array( [ self.G.degree(node) for node in xrange(self.N_nodes) ],dtype=np.float64 )
        self.p_expected = 0.5 * self.k / self.m


        if initial_seeds is None:
            initial_node = random.randint(self.N_walker)
            self.initial_nodes = [ initial_node for walker in xrange(self.N_walker) ]
        elif initial_seeds == "random":
            self.initial_nodes = random.randint(self.N_nodes,size=(self.N_walker,))
        elif hasattr(initial_seeds,"__len__"):
            self.initial_nodes = initial_seeds

        self.record_jump_distance = record_jump_distance

        self.rewind()


    def rewind(self):

        self.visited_nodes = [ set() for walker in xrange(self.N_walker) ]
        for walker in xrange(self.N_walker):
            self.visited_nodes[walker].add(self.initial_nodes[walker])

        if self.record_jump_distance:
            self.jump_distances = []

        self.remaining_walkers = set(range(self.N_walker))

        self.current_nodes = self.initial_nodes




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
            self.visited_nodes[walker].add(neigh)

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

            self.remaining_walkers -= set(finished_walkers)

        if get_trajectories:
            return trajectories
        else:
            return trajectories[:,-1]

    def simulate_till_max_visited_nodes(self,nmax,snapshots_at=[]):

        snapshots = []

        t = 0

        self.arrival_times = np.zeros((self.N_walker,),dtype=int)

        while len(self.remaining_walkers)>0:

            finished_walkers = []
            for walker in self.remaining_walkers:
                visited = len(self.visited_nodes[walker])
                if visited == nmax:
                    finished_walkers.append(walker)
                    self.arrival_times[walker] = t

            self.remaining_walkers -= set(finished_walkers)

            if t in snapshots_at:
                snapshots.append( np.array([len(self.visited_nodes[walker]) for walker in xrange(self.N_walker) ],dtype=int) )

            t += 1

            self.timestep()

        if len(snapshots_at)>0:
            return self.arrival_times, snapshots
        else:
            return self.arrival_times


def step_histogram(data,**kwargs):
    steps, bins = np.histogram(data,**kwargs)

    y = np.concatenate((
                            np.array([0.]),
                            steps,
                            np.array([steps[-1],0]),
                        ))
    x = np.concatenate((
                            np.array([bins[0]]),
                            bins,
                            np.array([bins[-1]]),
                      ))

    return y,x

def cut_histogram(data,bins,**kwargs):

    assert np.std(bins[1:]-bins[:-1]) == 0.
    
    db = bins[1]-bins[0]
    steps, bins = np.histogram(data,bins=bins,**kwargs)

    indices = np.nonzero(steps)[0]
    min_i = indices[0]
    max_i = indices[-1]

    return steps[min_i:max_i+1], min_i, max_i
    



if __name__ == "__main__":

    import pylab as pl
    import seaborn as sns
    from networkprops import networkprops as nprops
    import mhrn

    from scipy.optimize import curve_fit

    sns.set_style("white")

    N_walker = 50
    tmax = 2000

    N = 10
    B = 8
    L = 3
    N = B**L
    k = 10
    xi = 8
    p_ER = float(k)/(N-1)
    #G = nx.fast_gnp_random_graph(N,p_ER)

    #G = mhrn.fast_mhr_graph(B,L,k,xi)
    G = mhrn.continuous_hierarchical_graph(B**L,k,np.log(xi)/np.log(B),redistribute_probability=False)

    props = nprops(G,use_giant_component=True)

    diff = ErgodicDiffusion(props.G,N_walker=N_walker,initial_seeds="random")

    traj = diff.simulate_till_max_time(tmax,get_trajectories=True)

    fig,ax = pl.subplots(1,3,figsize=(12,5))

    t_ = np.arange(tmax)
    
    for walker in xrange(N_walker):
        ax[0].plot(t_,traj[walker,:],'b',alpha=0.1)

    diff.rewind()

    snapshot_times = [ 500, 1000, 2000, 4000 ]

    arrival_times,snapshots = diff.simulate_till_max_visited_nodes(diff.N_nodes,snapshot_times)

    data, bins = step_histogram(arrival_times,bins=np.arange(42)*500)

    ax[1].step(bins,data,where='post')

    for it,t in enumerate(snapshot_times):
        data, bins = step_histogram(snapshots[it],bins=np.arange(56)*10)
        #data, bins = np.histogram(snapshots[it])
        #data = np.concatenate((np.data,np.array([data[-1]])))

        ax[2].step(bins,data,where='post')

    pl.show()


