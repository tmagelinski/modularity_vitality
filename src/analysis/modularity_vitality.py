import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse import diags


def getSparseA(g):
    edges = [(e.source, e.target) for e in g.es()]
    sources, targets = list(zip(*edges))
    if g.is_weighted():
        weights = np.array(g.es['weight'], dtype=float)
    else:
        weights = np.ones(len(sources))
    self_loop_inds = (np.array(sources) == np.array(targets))
    weights[self_loop_inds] = weights[self_loop_inds] / 2
    weights = list(weights)
    A = csr_matrix((weights + weights, (sources + targets, targets + sources)),
                   shape=(g.vcount(), g.vcount()))
    return A


def getGroupIndicator(g, membership, rows=None):
    if not rows:
        rows = list(range(g.vcount()))
    cols = membership
    vals = np.ones(len(cols))
    group_indicator_mat = csr_matrix((vals, (rows, cols)),
                                     shape=(g.vcount(), max(membership) + 1))
    return group_indicator_mat


def getDegMat(node_deg_by_group, rows, cols):
    degrees = node_deg_by_group.sum(1)
    degrees = np.array(degrees).flatten()
    deg_mat = csr_matrix((degrees, (rows, cols)),
                         shape=node_deg_by_group.shape)
    degrees = degrees[:, np.newaxis]
    return degrees, deg_mat


def newMods(g, part):
    if g.is_weighted():
        weight_key = 'weight'
    else:
        weight_key = None
    index = list(range(g.vcount()))
    membership = part.membership

    m = sum(g.strength(weights=weight_key)) / 2

    A = getSparseA(g)
    self_loops = A.diagonal().sum()
    group_indicator_mat = getGroupIndicator(g, membership, rows=index)
    node_deg_by_group = A * group_indicator_mat

    internal_edges = (node_deg_by_group[index, membership].sum() + self_loops) / 2

    degrees, deg_mat = getDegMat(node_deg_by_group, index, membership)
    node_deg_by_group += deg_mat

    group_degs = (deg_mat + diags(A.diagonal()) * group_indicator_mat).sum(0)

    internal_deg = node_deg_by_group[index, membership].transpose() - degrees

    starCenter = (degrees == m)
    degrees[starCenter] = 0  # temp replacement avoid division by 0

    q1_links = (internal_edges - internal_deg) / (m - degrees)
    # expanding out (group_degs - node_deg_by_group)^2 is slightly faster:
    expected_impact = np.power(group_degs, 2).sum() - 2 * (node_deg_by_group * group_degs.transpose()) +\
        node_deg_by_group.multiply(node_deg_by_group).sum(1)
    q1_degrees = expected_impact / (4 * (m - degrees)**2)
    q1s = q1_links - q1_degrees
    q1s[starCenter] = 0
    q1s = np.array(q1s).flatten()
    return q1s


def modularity_vitality(g, part):
    q0 = part.modularity
    q1s = newMods(g, part)
    vitalities = (q0 - q1s).tolist()
    return vitalities
