from __future__ import print_function

from collections import Counter
izip = zip

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
        self.localization_time = [ np.dot(self.dist_time[-1],self.dist_time[-1]) ]

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
        self.localization_time.append(np.dot(new_dist_f, new_dist_f))
        self.cos_corr.append( np.dot(self.p_expected,self.current_dist)/np.linalg.norm(self.p_expected)/np.linalg.norm(self.current_dist) )

    def simulation(self,tmax):

        for t in range(tmax):
            self.timestep()

        return np.array(self.dist_time), np.array(self.std_time), np.array(self.localization_time)


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
    xi = 0.25
    p_ER = float(k)/(N-1)
    #G = nx.fast_gnp_random_graph(N,p_ER)

    #G = mhrn.fast_mhr_graph(B,L,k,xi)
    G = mhrn.continuous_hierarchical_graph(B**L,k,np.log(xi)/np.log(B),redistribute_probability=False)

    #G = nx.cycle_graph(N)

    props = nprops(G,use_giant_component=True)


    #diff = SimpleDiffusion(props.G,initial_distribution="random")
    diff = SimpleDiffusion(props.G,record_jump_distance=True)

    #dist,std,t,cos_corr = diff.simulate_till_equilibration(2e-2)
    dist,std,loc = diff.simulation(100)
    cos_corr = np.array(diff.cos_corr)

    #loc = std
    std = loc

    t = np.arange(len(std))


    lambda_2 = props.get_smallest_laplacian_eigenvalue()
    print(lambda_2)
    std0 = std[0]
    std_eq = std[-1]

    print(std0,std_eq)

    #fig,ax = pl.subplots(2,2,figsize=(13,9))
    fig = pl.figure(figsize=(13,9))
    ax0 = pl.subplot(221)
    ax1 = pl.subplot(122)
    ax2 = pl.subplot(223)
    ax = np.array([ax0,ax1,ax2])
    ax = ax.flatten()
    ax[0].plot(t,std,label='simulation')
    ax[0].set_yscale("log")
    ax[0].set_title(r"$B=%d,L=%d,\langle k \rangle=%d,\xi=%4.2f$" % (B,L,k,xi))
    ax[0].set_xlabel(r"time $t\times r$")
    ax[0].set_ylabel(r"concentration fluctuation $\sqrt{\mathrm{Var}(c_i(t))}$")

    ax[1].imshow(np.log(dist.T),interpolation='nearest',extent=(t.min(), t.max(), 0, diff.N_nodes-1),aspect='auto')
    ax[1].set_title(r"Walker concentration")
    ax[1].set_xlabel(r"time $t\times r$")
    ax[1].set_ylabel(r"node index")

    # plot distribution of walkers
    #ax[2].plot(np.arange(diff.N_nodes),dist[-1,:])
    #ax[2].plot(np.arange(diff.N_nodes),diff.k/diff.m*0.5)
    #ax[0].plot(t,np.ones_like(t)*np.std(diff.k/diff.m*0.5))

    # plot cosine correlation 
    #ax[3].plot(t,1-cos_corr)
    #ax[3].set_yscale("log")

    # plot jump distribution
    N_jumps = float(len(diff.jump_distances))
    dist_h = Counter(diff.jump_distances)
    dists = np.arange(1,diff.N_nodes/2)
    vals = np.array([ dist_h[d]/N_jumps for d in dists ])
    ax[2].plot(dists,vals,'.',label='simulation')
    ax[2].set_xscale("log")
    ax[2].set_yscale("log")


    inds = np.nonzero(vals)[0]
    p = np.polyfit(np.log(dists[inds]),np.log(vals[inds]),1)
    mu = p[0]
    a = np.exp(p[1])
    print("mu =",mu)
    print("a =",a)

    #ax[2].plot(dists,a * dists**mu,'--',label='$%4.2f|\Delta i|^{%4.2f}$' % (a,mu))
    #ax[2].plot(dists,a * dists**-1,'--',label='$%4.2f|\Delta i|^{%4.2f}$' % (a,mu))
    #ax[2].plot(dists,a * dists**(np.log(xi)/np.log(B)),'--')
    popt, pcov = curve_fit(lambda x, A: np.log( A*x**(-1+np.log(xi)/np.log(B)) ), dists[inds],np.log(vals[inds]),p0=[2*a])
    ax[2].plot(dists,popt[0] * dists**(-1+np.log(xi)/np.log(B)),'-',lw=3,alpha=.5,label=r'$%4.2f|\Delta i|^{\mathrm{log}\xi/\mathrm{log}B-1}$' % (popt[0]))
    ax[2].legend()

    ax[2].set_title(r"jump distribution (pmf)")
    ax[2].set_xlabel(r"jump distance $|\Delta i|$")
    ax[2].set_ylabel(r"probability")




    #fit ..
    start_fit = 3
    popt, pcov = curve_fit(func, t[start_fit:], np.log(std[start_fit:]),p0=[std0-std_eq,lambda_2,std_eq])
    tau = 1./popt[1]
    print(tau,1./lambda_2)
            
    print(popt)
    fit_res = np.exp(func(t,*popt))
    print(fit_res)
    ax[0].plot(t,fit_res,'r',label=r'exp. fit for $t\times r>3$')
    ax[0].plot(t,fit_res[-1] + (fit_res[0]-fit_res[-1]) * np.exp(-t*lambda_2),'--',label=r'$\mathrm{exp}(-\lambda_2rt)$')

    ax[0].legend()

    pl.show()
