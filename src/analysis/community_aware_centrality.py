import numpy as np
import src.analysis.modularity_vitality as mv


def getInternalExternalDegrees(g, part, loops=True):
    rows = list(range(g.vcount()))
    cols = part.membership
    A = mv.getSparseA(g)
    group_ind = mv.getGroupIndicator(g, cols, rows)
    node_deg_by_group = A * group_ind
    internal_degrees = node_deg_by_group[rows, cols]
    if loops:
        internal_degrees += A.diagonal()
    external_degrees = node_deg_by_group.sum(1).transpose() - node_deg_by_group[rows, cols]
    internal_degrees = np.array(internal_degrees).flatten()
    external_degrees = np.array(external_degrees).flatten()
    return internal_degrees, external_degrees


def getGroupFraction(g, part):
    internal_degrees, external_degrees = getInternalExternalDegrees(g, part)
    group_ind = mv.getGroupIndicator(g, part.membership)
    # internal_fraction = internal_degrees / (internal_degrees + external_degrees)
    # group_fraction = internal_fraction * group_ind
    group_fraction = (internal_degrees * group_ind) / ((internal_degrees + external_degrees) * group_ind)
    return group_fraction


def modularity_vitality(g, part):
    return mv.modularity_vitality(g, part)


def absolute_modularity_vitality(g, part):
    mv = modularity_vitality(g, part)
    amv = np.abs(mv).tolist()
    return amv


def masuda(g, part):
    if not g.is_weighted():
        g.es['weight'] = np.ones(g.ecount())
    group_G = part.cluster_graph(combine_edges='sum')
    # not really needed but good check:
    for edge in group_G.es:
        if edge.is_loop():
            edge['weight'] = 0
    group_eigs, group_lambda = group_G.eigenvector_centrality(return_eigenvalue=True, weights='weight')

    A = mv.getSparseA(g)
    rows = list(range(g.vcount()))
    cols = part.membership
    group_ind = mv.getGroupIndicator(g, cols, rows)
    node_deg_by_group = A * group_ind
    # node_deg_by_group[rows, cols] = 0
    node_deg_by_group = node_deg_by_group - node_deg_by_group.multiply(group_ind)
    x = (node_deg_by_group * group_eigs) / group_lambda
    mu_c = group_ind * group_eigs
    mas = (2 * mu_c - x) * (node_deg_by_group * group_eigs)
    mas = mas.tolist()
    return mas


def community_hub_bridge(g, part):
    internal_degrees, external_degrees = getInternalExternalDegrees(g, part, False)
    A = mv.getSparseA(g)
    group_ind = mv.getGroupIndicator(g, part.membership)
    node_deg_by_group = A * group_ind
    b_c = np.array((node_deg_by_group != 0).sum(1) - 1).flatten()
    c_k = [len(comm) for comm in part]
    c_k_vec = np.array([c_k[i] for i in part.membership])
    chb = c_k_vec * internal_degrees + b_c * external_degrees
    chb = chb.tolist()
    return chb


def weighted_modular_centrality_degree(g, part):
    internal_degrees, external_degrees = getInternalExternalDegrees(g, part)
    mu_c = getGroupFraction(g, part)
    group_ind = mv.getGroupIndicator(g, part.membership)
    node_mu = group_ind * mu_c.transpose()
    wmc_d = node_mu * internal_degrees + (1 - node_mu) * external_degrees
    wmc_d = wmc_d.tolist()
    return wmc_d


def adjusted_modular_centrality_degree(g, part):
    internal_degrees, external_degrees = getInternalExternalDegrees(g, part)
    mu_c = getGroupFraction(g, part)
    group_ind = mv.getGroupIndicator(g, part.membership)
    node_mu = group_ind * mu_c.transpose()
    amc_d = (1 - node_mu) * internal_degrees + node_mu * external_degrees
    amc_d = amc_d.tolist()
    return amc_d


def degree(g, part):
    if g.is_weighted():
        return g.strength(weights='weight')
    else:
        return g.degree()
