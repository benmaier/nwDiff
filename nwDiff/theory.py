
import numpy as np
izip = zip
from scipy.integrate import quad
from scipy.integrate import romberg
from networkprops import networkprops as nprops
from scipy.special import polygamma as psi
import scipy.sparse as sprs
from scipy.sparse.linalg import inv as sprs_inv

def P(t,rates,starting_node):
    #results_per_node = (1.0-np.exp(-rates*t))
    #ndcs = np.array([i for i in xrange(len(rates)) if not (i == starting_node)],dtype=int)
    #product = results_per_node[ndcs].prod()

    product = (1.0-np.exp(-rates*t)).prod()/(1.0-np.exp(-rates[starting_node]*t))

    return product

def P_all_nodes(t,rates):
    results_per_node = (1.0-np.exp(-rates*t))
    return results_per_node.prod()

def get_gmfpt_per_target(degrees,structural_degree_exponent=1.):
    k = degrees.mean()
    N = len(degrees)
    return N*k/degrees**structural_degree_exponent * 1./(1.-1./k)

def get_gmfpt_per_target_estimated_by_neighborhood_second_neighbors(G,target):

    k_mean = np.mean([ float(d[1]) for d in G.degree() ])
    N = float(G.number_of_nodes())

    # find first and unique second neighbors
    n = target
    first_neighs = set(G.neighbors(n))
    first_neighs.add(n)
    second_neighs = set()
    for neigh in first_neighs:
        second_neighs.update(set(G.neighbors(neigh)))
    second_neighs = list(second_neighs - first_neighs)
    first_neighs.remove(n)
    first_neighs = list(first_neighs)

    k = np.array( [ G.degree(u) for u in first_neighs ] +\
                  [ G.degree(v) for v in second_neighs ],
                dtype=float)

    remap = { node: i for i,node in enumerate(first_neighs+second_neighs) }
    a = np.concatenate(( 
                        np.ones_like(first_neighs),
                        np.zeros_like(second_neighs),
                       )
                      )

    A_2 = sprs.lil_matrix((len(k),len(k)))
    for u in first_neighs:
        for v in G.neighbors(u):
            if v != target:
                A_2[ remap[u], remap[v] ] = 1.
                A_2[ remap[v], remap[u] ] = 1.
    A_2 = A_2.tocsc()
    D_2 = sprs.diags(k)
    L_2_inv = sprs_inv((D_2 - A_2).tocsc())
        
    ones = np.ones_like(k)
    b = np.zeros_like(k)
    k_cluster = A_2.dot(ones).flatten()

    nf = len(first_neighs)
    b[nf:] = (k[nf:] - k_cluster[nf:]) / N / k_mean

    beta_target = a.dot(L_2_inv.dot(b))

    return 1./beta_target


def get_gmfpt_per_target_estimated_by_neighborhood_first_neighbors(G,target):

    k_mean = np.mean([ float(d[1]) for d in G.degree() ])
    N = float(G.number_of_nodes())

    # find first and unique second neighbors
    n = target
    first_neighs = list(G.neighbors(n))

    k = np.array( [ G.degree(u) for u in first_neighs ], dtype=float)

    remap = { node: i for i,node in enumerate(first_neighs) }
    a = np.ones_like(k)

    A_2 = sprs.lil_matrix((len(k),len(k)))
    for u in first_neighs:
        for v in G.neighbors(u):
            if v != target and v in first_neighs:
                A_2[ remap[u], remap[v] ] = 1.
                A_2[ remap[v], remap[u] ] = 1.
    A_2 = A_2.tocsc()
    D_2 = sprs.diags(k)
    L_2_inv = sprs_inv((D_2 - A_2).tocsc())
        
    ones = np.ones_like(k)
    k_cluster = A_2.dot(ones).flatten()

    b = (k - k_cluster - 1.) / N / k_mean

    beta_target = a.dot(L_2_inv.dot(b))

    return 1./beta_target

def mean_cover_time_starting_at(rates,starting_node,upper_bound=np.inf):
    result = quad(lambda t,r,s: 1.0-P(t,r,s), 0, upper_bound, args=(rates,starting_node),limit=10000)[0]
    return result

def get_mean_cover_time(degrees=None,rates=None,G=None,structural_degree_exponent=1.,upper_bound=np.inf):

    if rates is None and degrees is not None and G is None:
        rates = 1.0/get_gmfpt_per_target(degrees,structural_degree_exponent=1.)
    if G is not None and degrees is None and rates is None:
        rates = estimate_rates_from_structure(G)
    elif rates is None and degrees is None:
        raise ValueError('Have to get either degrees, rates or networkx graph-object')

    T = np.zeros_like(rates)
    for starting_node in range(len(rates)):
        upper_bound = estimate_upper_bound_P(rates,starting_node)
        T[starting_node] = mean_cover_time_starting_at(rates,starting_node,upper_bound)

    return T.mean()

def estimate_upper_bound_P(rates,starting_node,eps=1e-10):
    t_exponent = 0

    while True:
        this_P = P(10**t_exponent,rates,starting_node)
        if 1.0-this_P<eps:
            break
        else:
            t_exponent += 1
    return 10.0**t_exponent

def estimate_upper_bound_P_all_nodes(rates,eps=1e-10):
    t_exponent = 0

    while True:
        this_P = P_all_nodes(10**t_exponent,rates)
        if 1.0-this_P<eps:
            break
        else:
            t_exponent += 1
    return 10.0**t_exponent

def estimate_lower_bound_P_all_nodes(rates,eps=1e-10):
    t_exponent = 0

    while True:
        this_P = P_all_nodes(10**t_exponent,rates)
        if this_P > eps:
            break
        else:
            t_exponent += 1
    return 10.0**(t_exponent - 1)

def get_mean_cover_time_from_one_integral(degrees=None,rates=None,G=None,structural_degree_exponent=1.,lower_bound=None,upper_bound=np.inf):
    if rates is None and degrees is not None and G is None:
        rates = 1.0/get_gmfpt_per_target(degrees,structural_degree_exponent=1.)
    if G is not None and degrees is None and rates is None:
        rates = estimate_rates_from_structure(G)
    elif rates is None and degrees is None:
        raise ValueError('Have to get either degrees, rates or networkx graph-object')
        

    N = len(rates)

    #result = quad(lambda t,r: 1.0-P_all_nodes(t,r), 0.0, np.inf, args=(rates,),limit=10000)[0]
    if upper_bound == np.inf:
        upper_bound = estimate_upper_bound_P_all_nodes(rates)
    if lower_bound is None:
        lower_bound = estimate_lower_bound_P_all_nodes(rates)
    result = lower_bound + quad(lambda t,r: 1.0-P_all_nodes(t,r), lower_bound, upper_bound, args=(rates,))[0]
    return result

def get_mean_cover_time_for_single_GMFPT(degrees=None,rates=None,G=None,structural_degree_exponent=1.,upper_bound=np.inf):
    if rates is None and degrees is not None and G is None:
        rates = 1.0/get_gmfpt_per_target(degrees,structural_degree_exponent=1.)
    if G is not None and degrees is None and rates is None:
        rates = estimate_rates_from_structure(G)
    elif rates is None and degrees is None:
        raise ValueError('Have to get either degrees, rates or networkx graph-object')

    taus = 1./rates
    mean_tau = np.mean(taus)

    N = len(rates)

    
    return mean_tau * (np.euler_gamma + psi(0, N))

        

def estimate_rates_from_structure(G):
    props = nprops(G)
    mfpts = props.get_mean_first_passage_times_for_all_targets_eigenvalue_method()
    rates = 1.0/mfpts
    return rates

def get_mean_cover_time_analytical(degrees=None,rates=None,structural_degree_exponent=1.):
    """ DON'T USE! NOT FEASIBLE """

    if rates is None and degrees is not None:
        rates = 1.0/get_gmfpt_per_target(degrees,structural_degree_exponent=1.)
    elif rates is None and degrees is None:
        raise ValueError('Have to get either degrees or rates.')

    N = len(rates)
        
    from itertools import chain, combinations

    def powerset(iterable):
        "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
        s = list(iterable)
        return chain.from_iterable(combinations(s, r) for r in range(1,len(s)+1))

    cover_time = 0.
    for subset in powerset(list(range(N))):
        cover_time += (-1.0)**(len(subset)+1) * rates[np.array(subset)].sum()**(-1.)

    return cover_time

if __name__=="__main__":
    import cNetworkDiffusion as diff
    import networkx as nx
    from networkprops import networkprops as nprops
    import pylab as pl

    mode = 'ER'
    #mode = 'BA'

    seed = 140576

    N_meas = 10
    N_nodes = 20

    ks = np.arange(1,21,dtype=float)
    p = ks/(N_nodes-1.0)

    T_sim = np.zeros((len(ks),N_meas))
    T_theory = np.zeros((len(ks),N_meas))
    T_theory_2 = np.zeros((len(ks),N_meas))
    ks_meas = np.zeros((len(ks),N_meas))
    for ik,k in enumerate(ks):

        for meas in range(N_meas):
            if mode == 'ER':
                G = nx.fast_gnp_random_graph(N_nodes,p[ik])
            elif mode == 'BA':
                G = nx.barabasi_albert_graph(N_nodes,int(k))

            # get giant component
            G = nprops(G,use_giant_component=True).G
            N = G.number_of_nodes()
            edges = G.edges()
            degrees = np.array(list(G.degree().values()))

            mmfpt, cover_time = diff.mmfpt_and_mean_cover_time(N,edges,seed=seed+ik*N_meas+meas)
            T_sim[ik,meas] = cover_time

            T_theory[ik,meas] = get_mean_cover_time(degrees)
            T_theory_2[ik,meas] = get_mean_cover_time_from_one_integral(degrees)
            ks_meas[ik,meas] = degrees.mean()

        print("k =", k)

    km = ks_meas.mean(axis=1)
    km_err = ks_meas.std(axis=1)/np.sqrt(N_meas-1)
    Ts = T_sim.mean(axis=1)
    Ts_err = T_sim.std(axis=1)/np.sqrt(N_meas-1)
    Tt = T_theory.mean(axis=1)
    Tt_err = T_theory.std(axis=1)/np.sqrt(N_meas-1)
    Tt2 = T_theory_2.mean(axis=1)
    Tt2_err = T_theory_2.std(axis=1)/np.sqrt(N_meas-1)

    pl.errorbar(
                km,
                Ts,
                xerr = km_err,
                yerr = Tt_err,
                fmt='s',
                mfc='None')
    pl.errorbar(
                km,
                Tt,
                xerr = km_err,
                yerr = Tt_err,
                fmt='d',
                mfc='None')
    pl.errorbar(
                km,
                Tt2,
                xerr = km_err,
                yerr = Tt2_err,
                fmt='d',
                mfc='None')

    pl.xlabel(r'mean degree $\left\langle k\right\rangle$ of giant component')
    pl.ylabel(r'mean coverage time $\left\langle T\right\rangle$ of giant component')
    pl.title('%s networks, $N=%d$'%(mode,N_nodes))
    pl.legend(['simulation','slow heuristics', 'fast heuristics'])

    fig, ax = pl.subplots(1,1)
    ax.plot(km,abs(1-Tt/Tt2))
    pl.show()



