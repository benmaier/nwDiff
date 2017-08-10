import numpy as np
from scipy.integrate import quad
from scipy.integrate import romberg
from networkprops import networkprops as nprops
from scipy.special import polygamma as psi

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
    

def mean_cover_time_starting_at(rates,starting_node,upper_bound=np.inf):
    result = quad(lambda t,r,s: 1.0-P(t,r,s), 0, upper_bound, args=(rates,starting_node),limit=10000)[0]
    return result

def get_mean_cover_time(degrees,structural_degree_exponent=1.,upper_bound=np.inf):

    rates = 1.0/get_gmfpt_per_target(degrees,structural_degree_exponent=1.)

    T = np.zeros_like(rates)
    for starting_node in xrange(len(degrees)):
        T[starting_node] = mean_cover_time_starting_at(rates,starting_node,upper_bound)

    return T.mean()

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

def get_mean_cover_time_from_one_integral(degrees=None,rates=None,G=None,structural_degree_exponent=1.,upper_bound=np.inf):
    if rates is None and degrees is not None and G is None:
        rates = 1.0/get_gmfpt_per_target(degrees,structural_degree_exponent=1.)
    if G is not None and degrees is None and rates is None:
        rates = estimate_rates_from_structure(G)
    elif rates is None and degrees is None:
        raise ValueError('Have to get either degrees, rates or networkx graph-object')
        

    N = len(rates)

    #result = quad(lambda t,r: 1.0-P_all_nodes(t,r), 0.0, np.inf, args=(rates,),limit=10000)[0]
    upper_bound = estimate_upper_bound_P_all_nodes(rates)
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
    for subset in powerset(range(N)):
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
            degrees = np.array(G.degree().values())

            mmfpt, cover_time = diff.mmfpt_and_mean_cover_time(N,edges,seed=seed+ik*N_meas+meas)
            T_sim[ik,meas] = cover_time

            T_theory[ik,meas] = get_mean_cover_time(degrees)
            T_theory_2[ik,meas] = get_mean_cover_time_from_one_integral(degrees)
            ks_meas[ik,meas] = degrees.mean()

        print "k =", k

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



