import src.analysis.mod_deltas as md
import numpy as np


def comm_hub_bridge(g, part):
    N = g.vcount()
    A = md.get_sparse_A(g)
    rows = list(range(N))
    cols = part.membership
    vals = np.ones(N)
    group_indicator = md.csr_matrix((vals, (rows, cols)), shape=(N, len(part)))
    n_groups = A * group_indicator
    internal_deg = n_groups[rows, cols]
    total_deg = A.sum(0)
    c_k = [len(comm) for comm in part]
    b_c = (n_groups != 0).sum(1) - 1
    intra = (group_indicator * np.array(c_k)) * np.array(internal_deg)
    inter = np.array(b_c) * np.array(total_deg - internal_deg).transpose()
    centralities = intra + inter.transpose()
    return centralities.flatten()


def modular_centrality(g, part):
    # cherifi
    N = g.vcount()
    A = md.get_sparse_A(g)
    rows = list(range(N))
    cols = part.membership
    vals = np.ones(N)
    group_indicator = md.csr_matrix((vals, (rows, cols)), shape=(N, len(part)))
    n_groups = A * group_indicator
    internal_deg = n_groups[rows, cols]
    total_deg = A.sum(0)

    group_internal = internal_deg * group_indicator
    group_total = total_deg * group_indicator
    mu_c = (np.array(group_total) - np.array(group_internal)) / np.array(group_total)
    node_mu = (group_indicator * mu_c.transpose())
    weighted_local = (1 - node_mu) * np.array(internal_deg).transpose()
    weighted_global = node_mu * np.array(total_deg - internal_deg).transpose()
    alpha = weighted_local + weighted_global
    return alpha.flatten()


def adjusted_modular_centrality(g, part):
    N = g.vcount()
    A = md.get_sparse_A(g)
    rows = list(range(N))
    cols = part.membership
    vals = np.ones(N)
    group_indicator = md.csr_matrix((vals, (rows, cols)), shape=(N, len(part)))
    n_groups = A * group_indicator
    internal_deg = n_groups[rows, cols]
    total_deg = A.sum(0)

    group_internal = internal_deg * group_indicator
    group_total = total_deg * group_indicator
    mu_c = np.array(group_internal) / np.array(group_total)
    node_mu = (group_indicator * mu_c.transpose())
    weighted_local = (1 - node_mu) * np.array(internal_deg).transpose()
    weighted_global = node_mu * np.array(total_deg - internal_deg).transpose()
    alpha = weighted_local + weighted_global
    return total_deg, alpha.flatten()


def masuda(g, part):
    N = g.vcount()
    A = md.get_sparse_A(g)
    rows = list(range(N))
    cols = part.membership
    vals = np.ones(N)
    group_indicator = md.csr_matrix((vals, (rows, cols)), shape=(N, len(part)))

#     group_A = (group_indicator.transpose() * A * group_indicator).todense()
#     np.fill_diagonal(group_A, 0)
#     rows, cols = group_A.nonzero()
#     weights = group_A[rows,cols].tolist()[0]
#     group_G = ig.Graph()
#     group_G.add_vertices(len(part))
#     group_G.add_edges(zip(rows, cols))
#     group_G.es['weight'] = weights
    if not g.is_weighted():
        g.es['weight'] = np.ones(g.ecount())
    group_G = part.cluster_graph(combine_edges='sum')
    # not really needed but good check:
    for edge in group_G.es:
        if edge.is_loop():
            edge['weight'] = 0
    group_eigs, group_lambda = group_G.eigenvector_centrality(return_eigenvalue=True, weights='weight')

    n_groups = A * group_indicator
    n_groups[rows, cols] = 0
    x = (n_groups * group_eigs) / group_lambda
    mu_k = group_indicator * group_eigs
    delta_lambda = (2 * mu_k - x) * (n_groups * group_eigs)
    return delta_lambda


def masuda_attack(g, part, n_immune):
    N = g.vcount()
    A = md.get_sparse_A(g)
    rows = list(range(N))
    cols = part.membership
    vals = np.ones(N)
    group_indicator = md.csr_matrix((vals, (rows, cols)), shape=(N, len(part)))

#     group_A = (group_indicator.transpose() * A * group_indicator).todense()
#     np.fill_diagonal(group_A, 0)
#     rows, cols = group_A.nonzero()
#     weights = group_A[rows,cols].tolist()[0]
#     group_G = ig.Graph()
#     group_G.add_vertices(len(part))
#     group_G.add_edges(zip(rows, cols))
#     group_G.es['weight'] = weights
    if not g.is_weighted():
        g.es['weight'] = np.ones(g.ecount())
    group_G = part.cluster_graph(combine_edges='sum')
    # not really needed but good check:
    for edge in group_G.es:
        if edge.is_loop():
            edge['weight'] = 0
    group_eigs, group_lambda = group_G.eigenvector_centrality(return_eigenvalue=True, weights='weight')

    n_groups = A * group_indicator
    n_groups[rows, cols] = 0
    x = (n_groups * group_eigs) / group_lambda
    mu_k = group_indicator * group_eigs
    delta_lambda = (2 * mu_k - x) * (n_groups * group_eigs)
    return np.argsort(-1 * delta_lambda)[:n_immune]
