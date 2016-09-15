from __future__ import print_function

from collections import Counter
from itertools import izip

import numpy as np
from numpy import random
import networkx as nx

class SimpleDiffusion():

    def __init__(self,G,N_walker=10000,initial_distribution=None,normed=True,record_jump_distance=False):
        
        self.N_walker = N_walker

        self.G = nx.convert_node_labels_to_integers(G)
        self.N_nodes = self.G.number_of_nodes()
        self.m = self.G.number_of_edges()

        self.k = np.array( [ self.G.degree(node) for node in xrange(self.N_nodes) ],dtype=np.float64 )
        self.p_expected = 0.5 * self.k / self.m

        if initial_distribution is None:

            initial_node = random.randint(self.N_nodes)
            #put all walkers on initial_node
            self.initial_dist = np.zeros((self.N_nodes,),dtype=np.int64)
            self.initial_dist[initial_node] = self.N_walker
        elif initial_distribution == "random":
            initial_distribution = random.rand(self.N_nodes)
            initial_distribution /= np.sum(initial_distribution)
            self.initial_dist = np.array(initial_distribution*self.N_walker,dtype=np.int64)
            self.N_walker = int(self.initial_dist.sum())
        else:
            self.initial_dist = np.array(initial_distribution*self.N_walker,dtype=np.int64)
            self.N_walker = int(self.initial_dist.sum())

        if normed:
            self.norm = self.N_walker
        else:
            self.norm = 1.

        self.dist_time = [ self.initial_dist / float(self.norm) ]
        self.std_time = [ np.std(self.dist_time[-1]) ]

        self.current_dist = np.array( self.initial_dist )
        self.cos_corr = [ np.dot(self.p_expected,self.current_dist)/np.linalg.norm(self.p_expected)/np.linalg.norm(self.current_dist) ]

        self.record_jump_distance = record_jump_distance
        if record_jump_distance:            
            self.jump_distances = []

    def timestep(self):

        # initiate new walker counter per node
        new_dist = np.zeros((self.N_nodes,))

        # iterate over all nodes carrying walkers
        for node in self.current_dist.nonzero()[0]:

            neighbors = self.G.neighbors(node)
            k = len(neighbors)
            current_walkers = int(self.current_dist[node])
            diffusion_to_neighbor = random.randint(k,size=current_walkers)

            for neigh in diffusion_to_neighbor:

                if self.record_jump_distance:
                    distance = abs(neighbors[neigh]-node)
                    if distance > self.N_nodes/2.:
                        distance = abs(distance-self.N_nodes)
                    self.jump_distances.append( distance )

                new_dist[neighbors[neigh]] += 1

        self.current_dist = new_dist

        new_dist_f = np.array(new_dist,dtype=np.float64) / self.norm
        self.dist_time.append( new_dist_f )
        self.std_time.append( np.std(new_dist_f) )
        self.cos_corr.append( np.dot(self.p_expected,self.current_dist)/np.linalg.norm(self.p_expected)/np.linalg.norm(self.current_dist) )

    def simulation(self,tmax):

        for t in range(tmax):
            self.timestep()

        return np.array(self.dist_time), np.array(self.std_time)


    def simulate_till_equilibration(self,eps=0.01):
        std_expected = np.std(0.5 * self.k / self.m)
        t = 0

        while 1-self.cos_corr[-1] > eps:
            #print(abs(self.std_time[-1] - std_expected) / std_expected,std_expected)
            #print(np.dot(p_expected,self.current_dist)/np.linalg.norm(p_expected)/np.linalg.norm(self.current_dist))
            self.timestep()
            t += 1

        return np.array(self.dist_time), np.array(self.std_time), t, np.array(self.cos_corr)



if __name__ == "__main__":

    import pylab as pl
    import seaborn as sns
    from networkprops import networkprops as nprops
    import mhrn

    from scipy.optimize import curve_fit

    def func(x, a, b, c):
        return np.log(a * np.exp(-b * x) + c)
    sns.set_style("white")

    N = 10
    B = 8
    L = 3
    N = B**L
    k = 10
    xi = B
    p_ER = float(k)/(N-1)
    #G = nx.fast_gnp_random_graph(N,p_ER)

    #G = mhrn.fast_mhr_graph(B,L,k,xi)
    G = mhrn.continuous_hierarchical_graph(B**L,k,np.log(xi)/np.log(B),redistribute_probability=False)

    #G = nx.cycle_graph(N)

    props = nprops(G,use_giant_component=True)


    #diff = SimpleDiffusion(props.G,initial_distribution="random")
    diff = SimpleDiffusion(props.G)

    #dist,std,t,cos_corr = diff.simulate_till_equilibration(2e-2)
    dist,std = diff.simulation(100)
    cos_corr = np.array(diff.cos_corr)

    t = np.arange(len(std))


    lambda_2 = props.get_smallest_laplacian_eigenvalue()
    print(lambda_2)
    std0 = std[0]
    std_eq = std[-1]

    print(std0,std_eq)

    fig,ax = pl.subplots(3,2,figsize=(13,13))
    ax = ax.flatten()
    ax[0].plot(t,std)
    ax[0].set_yscale("log")
    ax[1].imshow(np.log(dist.T),interpolation='nearest',extent=(t.min(), t.max(), 0, diff.N_nodes-1))
    ax[2].plot(np.arange(diff.N_nodes),dist[-1,:])
    ax[2].plot(np.arange(diff.N_nodes),diff.k/diff.m*0.5)
    ax[0].plot(t,np.ones_like(t)*np.std(diff.k/diff.m*0.5))
    ax[3].plot(t,1-cos_corr)
    ax[3].set_yscale("log")

    N_jumps = float(len(diff.jump_distances))
    dist_h = Counter(diff.jump_distances)
    dists = np.arange(1,diff.N_nodes/2)
    vals = np.array([ dist_h[d]/N_jumps for d in dists ])
    ax[4].plot(dists,vals,'.')
    ax[4].set_xscale("log")
    ax[4].set_yscale("log")


    inds = np.nonzero(vals)[0]
    p = np.polyfit(np.log(dists[inds]),np.log(vals[inds]),1)
    mu = p[0]
    a = np.exp(p[1])
    print("mu =",mu)
    print("a =",a)

    ax[4].plot(dists,a * dists**mu,'--')
    ax[4].plot(dists,a * dists**(-1+np.log(xi)/np.log(B)),'--')



    #fit ..
    start_fit = 3
    popt, pcov = curve_fit(func, t[start_fit:], np.log(std[start_fit:]),p0=[std0-std_eq,lambda_2,std_eq])
    tau = 1./popt[1]
    print(tau,1./lambda_2)
            
    print(popt)
    fit_res = np.exp(func(t,*popt))
    ax[0].plot(t,fit_res,'r')
    ax[0].plot(t,fit_res[-1] + (fit_res[0]-fit_res[-1]) * np.exp(-t*lambda_2))

    pl.show()
