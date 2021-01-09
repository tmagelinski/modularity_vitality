import numpy as np
# import pandas as pd
# import networkx as nx
# import community
import scipy
from scipy.sparse import csr_matrix
from scipy.sparse import diags

import igraph as ig


def get_sparse_A(g):
    '''returns sparse adjacency matrix for igraph graph g'''
    if g.is_weighted():
        edge_data = [(e.source, e.target, e['weight']) for e in g.es]
    else:
        edge_data = [(e.source, e.target, 1) for e in g.es]
    sources, targets, weights = list(map(np.array, (zip(*edge_data))))

    non_self_loops = (sources != targets)

    non_loop_sources = list(sources[non_self_loops])
    non_loop_targets = list(targets[non_self_loops])
    non_loop_weights = list(weights[non_self_loops])

    if not g.is_directed():
        # symetrize
        all_sources = non_loop_sources + non_loop_targets
        all_targets = non_loop_targets + non_loop_sources
        all_weights = non_loop_weights + non_loop_weights
    else:
        all_sources = non_loop_sources
        all_targets = non_loop_targets
        all_weights = non_loop_weights

    loop_sources = list(sources[~non_self_loops])
    loop_targets = list(targets[~non_self_loops])
    loop_weights = list(weights[~non_self_loops])

    all_sources += loop_sources
    all_targets += loop_targets
    all_weights += loop_weights

    A = csr_matrix((all_weights,
                    (all_sources, all_targets)),
                   shape=(g.vcount(), g.vcount()))
    return A


def get_group_indicator(g, part):
    ''' creates sparse matrix indicated group of each node '''
    rows = list(range(g.vcount()))
    cols = part.membership
    vals = np.ones(len(cols))
    group_indicator_mat = csr_matrix((vals, (rows, cols)),
                                     shape=(g.vcount(), len(part)))
    return rows, group_indicator_mat


def get_deg_mat(n_groups, rows, cols):
    '''
    returns matrix with degrees in nodes group index
    n_groups indicates links from nodes to each group
    rows is index of each node
    cols is indiex of each node's group assignment
    '''
    degrees = n_groups.sum(1)
    degrees = np.array(degrees)
    deg_mat = csr_matrix((degrees.flatten(), (rows, cols)),
                         shape=n_groups.shape)
    return degrees, deg_mat


def true_deltas(g, part):
    if g.is_weighted():
        weight_key = 'weight'
    else:
        weight_key = None

    q0 = part.modularity
    m = sum(g.strength(weights=weight_key)) / 2

    A = get_sparse_A(g)
    self_loops = A.diagonal().sum()
    rows, group_indicator_mat = get_group_indicator(g, part)
    n_groups = A * group_indicator_mat

    internal_edges = (n_groups[rows, part.membership].sum() + self_loops) / 2

    degrees, deg_mat = get_deg_mat(n_groups, rows, part.membership)
    n_groups += deg_mat

    group_degs = (deg_mat + diags(A.diagonal()) * group_indicator_mat).sum(0)
    group_degs = np.array(group_degs).flatten()
#     print(group_degs - np.array(deg_mat.sum(0)).flatten())

    internal_deg = n_groups[rows, part.membership].transpose() - degrees

    p0 = 4 * (m ** 2) * (q0 - internal_edges / m)
    delta_p = n_groups.power(2).sum(1) - 2 * n_groups * group_degs[:, np.newaxis]

    q1_links = (internal_edges - internal_deg) / (m - degrees)
    q1_degrees = (p0 - delta_p) / (4 * (m - degrees)**2)
    q1 = q1_links + q1_degrees
    deltas = q1 - q0
    return np.array(deltas).flatten().tolist()


def mod_deltas(g, part):
    if g.is_weighted():
        weight_key = 'weight'
    else:
        weight_key = None

    q0 = part.modularity
    m = sum(g.strength(weights=weight_key)) / 2

    A = get_sparse_A(g)
    self_loops = A.diagonal().sum()
    rows, group_indicator_mat = get_group_indicator(g, part)
    n_groups = A * group_indicator_mat

    internal_edges = (n_groups[rows, part.membership].sum() + self_loops) / 2

    degrees, deg_mat = get_deg_mat(n_groups, rows, part.membership)
    n_groups += deg_mat

    group_degs = (deg_mat + diags(A.diagonal()) * group_indicator_mat).sum(0)

    internal_deg = n_groups[rows, part.membership].transpose() - degrees

    q1_links = (internal_edges - internal_deg) / (m - degrees)
    # expanding out (group_degs - n_groups)^2 is faster:
    expected_impact = np.power(group_degs, 2).sum() - 2 * (n_groups * group_degs.transpose()) + n_groups.multiply(n_groups).sum(1)
    q1_degrees = expected_impact / (4 * (m - degrees)**2)
    q1 = q1_links - q1_degrees
    deltas = q1 - q0
    return np.array(deltas).flatten().tolist()


def split_deltas(g, part):
    if g.is_weighted():
        weight_key = 'weight'
    else:
        weight_key = None

    m = sum(g.strength(weights=weight_key)) / 2

    A = get_sparse_A(g)
    self_loops = A.diagonal().sum()
    rows, group_indicator_mat = get_group_indicator(g, part)
    n_groups = A * group_indicator_mat

    internal_edges = (n_groups[rows, part.membership].sum() + self_loops) / 2

    degrees, deg_mat = get_deg_mat(n_groups, rows, part.membership)
    n_groups += deg_mat

    group_degs = (deg_mat + diags(A.diagonal()) * group_indicator_mat).sum(0)

    internal_deg = n_groups[rows, part.membership].transpose() - degrees

    expected_impact = np.power(group_degs, 2).sum() - 2 * (n_groups * group_degs.transpose()) + n_groups.multiply(n_groups).sum(1)

    q1_links = (internal_edges - internal_deg) / (m - degrees)
    q1_degrees = expected_impact / (4 * (m - degrees)**2)

    neg = q1_degrees - np.power(group_degs, 2).sum() / (4 * (m ** 2))
    pos = q1_links - (internal_edges / m)

    return pos, neg


def delete_rows_csr(mat, indices):
    """
    Remove the rows denoted by ``indices`` form the CSR sparse matrix ``mat``.
    """
    if not isinstance(mat, scipy.sparse.csr_matrix):
        raise ValueError("works only for CSR format -- use .tocsr() first")
    indices = list(indices)
    mask = np.ones(mat.shape[0], dtype=bool)
    mask[indices] = False
    return mat[mask]


def delete_both_csr(mat, indices):
    """
    Remove the rows denoted by ``indices`` form the CSR sparse matrix ``mat``.
    """
    if not isinstance(mat, scipy.sparse.csr_matrix):
        raise ValueError("works only for CSR format -- use .tocsr() first")
    assert(mat.shape[0] == mat.shape[1])
    indices = list(indices)
    mask = np.ones(mat.shape[0], dtype=bool)
    mask[indices] = False
    return mat[mask][:, mask]


def RM(g, part, cutoff):
    if g.is_weighted():
        weight_key = 'weight'
    else:
        weight_key = None

    node_names = [n['name'] for n in g.vs()]
    q0 = part.modularity
    m = sum(g.strength(weights=weight_key)) / 2

    A = get_sparse_A(g)
    rows, group_indicator_mat = get_group_indicator(g, part)

    self_loops = A.diagonal().sum()
    n_groups = A * group_indicator_mat
    internal_edges = (n_groups[rows, part.membership].sum() + self_loops) / 2

    degrees, deg_mat = get_deg_mat(n_groups, rows, part.membership)
    n_groups += deg_mat

    group_degs = (deg_mat + diags(A.diagonal()) * group_indicator_mat).sum(0)
    membership = part.membership
    internal_deg = n_groups[rows, membership].transpose() - degrees

    deleted_nodes = []
    mods = [q0]
    for i in range(cutoff):
        q1_links = (internal_edges - internal_deg) / (m - degrees)
        # expanding out (group_degs - n_groups)^2 is faster:
        expected_impact = np.power(group_degs, 2).sum() - 2 * (n_groups * group_degs.transpose()) + n_groups.multiply(n_groups).sum(1)
        q1_degrees = expected_impact / (4 * (m - degrees)**2)
        q1 = q1_links - q1_degrees

        n2del = np.argmax(q1)

        mods.append(q1[n2del].item())
        n_remove = node_names.pop(n2del)
        deleted_nodes.append(n_remove)

        m -= degrees[n2del].item()

        group_indicator_mat = delete_rows_csr(group_indicator_mat, [n2del])
        A = delete_both_csr(A, [n2del])
        membership.pop(n2del)
        rows = rows[:-1]

        self_loops = A.diagonal().sum()
        n_groups = A * group_indicator_mat
        degrees, deg_mat = get_deg_mat(n_groups, rows, membership)
        internal_edges = (n_groups[rows, membership].sum() + self_loops) / 2
        n_groups += deg_mat

        group_degs = (deg_mat + diags(A.diagonal()) * group_indicator_mat).sum(0)
        internal_deg = n_groups[rows, membership].transpose() - degrees
    return mods, deleted_nodes


def abs_mods(g, part):
    if g.is_weighted():
        weight_key = 'weight'
    else:
        weight_key = None

    q0 = part.modularity
    m = sum(g.strength(weights=weight_key)) / 2

    A = get_sparse_A(g)
    self_loops = A.diagonal().sum()
    rows, group_indicator_mat = get_group_indicator(g, part)
    n_groups = A * group_indicator_mat

    internal_edges = (n_groups[rows, part.membership].sum() + self_loops) / 2

    degrees, deg_mat = get_deg_mat(n_groups, rows, part.membership)
    n_groups += deg_mat

    group_degs = (deg_mat + diags(A.diagonal()) * group_indicator_mat).sum(0)

    internal_deg = n_groups[rows, part.membership].transpose() - degrees

    # q1_links = (internal_edges - internal_deg) / (m - degrees)
    # expanding out (group_degs - n_groups)^2 is faster:
    expected_impact = np.power(group_degs, 2).sum() + 2 * (n_groups * group_degs.transpose()) + n_groups.multiply(n_groups).sum(1)
    q1_degrees = expected_impact / (4 * (m - degrees)**2)
    # q1 = q1_links + q1_degrees
    # deltas = q1 - q0
    return np.array(q1_degrees).flatten().tolist()


def RM_deception(g, part, cutoff):
    if g.is_weighted():
        weight_key = 'weight'
    else:
        weight_key = None

    node_names = [n['name'] for n in g.vs()]
    q0 = part.modularity
    m = sum(g.strength(weights=weight_key)) / 2

    A = get_sparse_A(g)
    rows, group_indicator_mat = get_group_indicator(g, part)

    self_loops = A.diagonal().sum()
    n_groups = A * group_indicator_mat
    internal_edges = (n_groups[rows, part.membership].sum() + self_loops) / 2

    degrees, deg_mat = get_deg_mat(n_groups, rows, part.membership)
    n_groups += deg_mat

    group_degs = (deg_mat + diags(A.diagonal()) * group_indicator_mat).sum(0)
    membership = part.membership
    internal_deg = n_groups[rows, membership].transpose() - degrees

    deleted_nodes = []
    mods = [q0]
    for i in range(cutoff):
        q1_links = (internal_edges - internal_deg) / (m - degrees)
        # expanding out (group_degs - n_groups)^2 is faster:
        expected_impact = np.power(group_degs, 2).sum() - 2 * (n_groups * group_degs.transpose()) + n_groups.multiply(n_groups).sum(1)
        q1_degrees = expected_impact / (4 * (m - degrees)**2)
        q1 = q1_links - q1_degrees

        n2del = np.argmin(q1)

        mods.append(q1[n2del].item())
        n_remove = node_names.pop(n2del)
        deleted_nodes.append(n_remove)

        m -= degrees[n2del].item()

        group_indicator_mat = delete_rows_csr(group_indicator_mat, [n2del])
        A = delete_both_csr(A, [n2del])
        membership.pop(n2del)
        rows = rows[:-1]

        self_loops = A.diagonal().sum()
        n_groups = A * group_indicator_mat
        degrees, deg_mat = get_deg_mat(n_groups, rows, membership)
        internal_edges = (n_groups[rows, membership].sum() + self_loops) / 2
        n_groups += deg_mat

        group_degs = (deg_mat + diags(A.diagonal()) * group_indicator_mat).sum(0)
        internal_deg = n_groups[rows, membership].transpose() - degrees
    return mods, deleted_nodes


# def preselected_mod_deltas(g, part, nodes):
#     is_weighted = g.is_weighted()
#     if is_weighted:
#         weight_key = 'weight'
#     else:
#         weight_key = None

#     m = sum(g.strength(weights=weight_key)) / 2

#     num_groups = len(part)
#     mods = [part.modularity]
#     for node in nodes:
#         n_groups = np.zeros(num_groups)
#         for neighbor in g.neighborood(node, mindist=1):
#             if is_weighted:
#                 n_groups[membership[neighbor]] +=


#     A = get_sparse_A(g)
#     self_loops = A.diagonal().sum()
#     rows, group_indicator_mat = get_group_indicator(g, part)
#     n_groups = A * group_indicator_mat

#     internal_edges = (n_groups[rows, part.membership].sum() + self_loops) / 2

#     degrees, deg_mat = get_deg_mat(n_groups, rows, part.membership)
#     n_groups += deg_mat

#     group_degs = (deg_mat + diags(A.diagonal()) * group_indicator_mat).sum(0)
