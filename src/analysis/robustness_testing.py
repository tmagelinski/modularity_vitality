import numpy as np
import igraph as ig
import src.analysis.community_aware_centrality as cac


def sigma_calc(g, nodes, calcs):
    stepsize = int(np.floor(len(nodes) / calcs))
    stepsize = np.maximum(stepsize, 1)
    h = g.copy()
    lc = len(max(h.components(), key=len))
    N = g.vcount()
    M = g.ecount()
    sigmas = [float(lc) / N]
    rho = [0]
    rho_e = [0]
    steps = int(np.ceil(len(nodes) / stepsize))
    for i in range(steps):
        nodes2del = nodes[i * stepsize:(i + 1) * stepsize]
        h.delete_vertices(nodes2del)
        lc = len(max(h.components(), key=len))
        sigmas.append(float(lc) / N)
        rho.append((i + 1) * stepsize / N)
        rho_e.append(1 - h.ecount() / M)
    return rho, sigmas, rho_e


def initial_attack(g, part, centrality_function, nodes_attacked=None, calculations=None):
    if (nodes_attacked is not None) and (calculations is not None):
        if nodes_attacked <= calculations:
            calculations = nodes_attacked
    elif calculations is not None:
        if g.vcount() <= calculations:
            calculations = g.vcount() - 1
    else:
        calculations = g.vcount() - 1

    centrality_values = centrality_function(g, part)
    attack_order = np.argsort(centrality_values)
    if centrality_function.__name__ == 'modularity_vitality':
        attack = list(attack_order)  # attack negative-first only for MV
    else:
        attack = list(reversed(attack_order))
    if nodes_attacked:
        attack = attack[:nodes_attacked]
    else:
        attack = attack[:-1]

    if 'name' not in g.vertex_attributes():
        g.vs['name'] = list(map(str, range(g.vcount())))
    attack = [g.vs[i]['name'] for i in attack]

    rho, sigmas, rho_e = sigma_calc(g, attack, calculations)
    return centrality_values, rho, sigmas, rho_e
