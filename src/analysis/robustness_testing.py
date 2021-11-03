import numpy as np
import igraph as ig


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


def initial_attack(g, part, centrality_function, nodes_attacked=None, calculations=None, return_values=False):
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
    elif type(g.vs['name'][0] == int):
        g.vs['name'] = list(map(str, g.vs['name']))
    attack = [g.vs[i]['name'] for i in attack]

    rho, sigmas, rho_e = sigma_calc(g, attack, calculations)
    if return_values:
        return centrality_values, rho, sigmas, rho_e
    else:
        return rho, sigmas, rho_e


def get_lc(g):
    lc = max(g.components(), key=len)
    lc_names = set(g.vs[lc]['name'])
    return lc_names


def get_bridge_nodes(g, part):
    bridge_nodes = {}
    membership = part.membership
    for edge in g.es:
        if membership[edge.source] != membership[edge.target]:
            s_name = g.vs[edge.source]['name']
            t_name = g.vs[edge.target]['name']
            if s_name not in bridge_nodes:
                bridge_nodes[s_name] = [t_name]
            else:
                bridge_nodes[s_name].append(t_name)
            if t_name not in bridge_nodes:
                bridge_nodes[t_name] = [s_name]
            else:
                bridge_nodes[t_name].append(s_name)
    return bridge_nodes


def update_bridges(bridges, node2delete):
    nodes2delete = [node2delete]
    for bridge_neighbor in bridges[node2delete]:
        bridges[bridge_neighbor].remove(node2delete)
        if len(bridges[bridge_neighbor]) == 0:
            nodes2delete.append(bridge_neighbor)
    for n2d in nodes2delete:
        del bridges[n2d]
    return bridges


def module_based_attack(g, part, centrality_function, nodes_attacked=None, calculations=None):
    if not nodes_attacked:
        nodes_attacked = g.vcount() - 1
    if 'name' not in g.vertex_attributes():
        g.vs['name'] = list(map(str, range(g.vcount())))
    elif type(g.vs['name'][0] == int):
        g.vs['name'] = list(map(str, g.vs['name']))

    bridges = get_bridge_nodes(g, part)

    lc_names = get_lc(g)

    centrality_values = np.array(centrality_function(g, part))
    centrality_order = np.argsort(centrality_values)
    if centrality_function.__name__ == 'modularity_vitality':
        centrality_order = list(centrality_order)  # attack negative-first only for MV
    else:
        centrality_order = list(reversed(centrality_order))
    sorted_names = [g.vs[n]['name'] for n in centrality_order]
    potential_nodes = [n for n in sorted_names if n in bridges]

    attack = []
    unordered_attack = set()
    h = g.copy()
    while len(potential_nodes) != 0:
        for node in potential_nodes:
            if (node in lc_names) and (node in bridges):
                attack.append(node)
                unordered_attack.add(node)
                h.delete_vertices(node)
                lc_names = get_lc(h)
                bridges = update_bridges(bridges, node)
        lc_bridges = [n for n in lc_names if n in bridges]
        if (len(lc_bridges) == 0) or (len(attack) == nodes_attacked):
            break
        else:
            potential_nodes = [n for n in potential_nodes if (n in bridges) and (n not in unordered_attack)]

    rho, sigmas, rho_e = sigma_calc(g, attack, calculations)
    return rho, sigmas, rho_e


def repeated_attack(g, part, centrality_function, nodes_attacked=None, calculations=None):
    if not nodes_attacked:
        nodes_attacked = g.vcount() - 1
    if 'name' not in g.vertex_attributes():
        g.vs['name'] = list(map(str, range(g.vcount())))
    elif type(g.vs['name'][0] == int):
        g.vs['name'] = list(map(str, g.vs['name']))

    h = g.copy()
    h_part = part
    attack = []
    while (len(attack) != nodes_attacked) and (h.ecount() != 0):
        values = centrality_function(h, h_part)
        if centrality_function.__name__ == 'modularity_vitality':
            node = np.argmin(values)
        else:
            node = np.argmax(values)
        attack.append(h.vs[node]['name'])
        h.delete_vertices(node)
        new_mem = h_part.membership
        new_mem.pop(node)
        h_part = ig.VertexClustering(h, membership=new_mem)
    rho, sigmas, rho_e = sigma_calc(g, attack, calculations)
    return rho, sigmas, rho_e
