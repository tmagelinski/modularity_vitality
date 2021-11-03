import igraph as ig
import numpy as np
import src.analysis.robustness_testing as rt
import src.analysis.community_aware_centrality as cac


def random_bounded_numbers_binomial(Nc, Na, lower, upper):
    done = False
    while not done:
        total_sum, cnt = 0, 0
        cell_sizes = []
        while total_sum < Na and cnt < Nc:
            cell_size = np.random.randint(lower, upper)
            total_sum += cell_size
            cnt += 1
            cell_sizes.append(cell_size)
        if total_sum == Na and cnt == Nc:
            done = True
    return cell_sizes


def random_bounded_numbers_normal(Nc, Na):

    # std_dev is just a value I selected that makes the most sense

    avg_cell_size = Na / Nc
    st_dev = Nc / 5

    done = False
    while not done:
        total_sum, cnt = 0, 0
        cell_sizes = []
        while total_sum < Na and cnt < Nc:
            cell_size = int(np.random.normal(avg_cell_size, st_dev))
            total_sum += cell_size
            cnt += 1
            cell_sizes.append(cell_size)
        if total_sum == Na and cnt == Nc:
            done = True
    return cell_sizes


def ER_graph(N, p, name_start=False, selfloops=True, between_communities=False):

    G = ig.Graph()
    if name_start or name_start == 0:
        G.add_vertices([str(i) for i in range(name_start, name_start + N)])
    else:
        G.add_vertices(N)

    for v_s in G.vs():
        for v_t in G.vs():
            if between_communities:
                if np.random.random() <= p:
                    G.add_edge(v_s, v_t)
            else:
                if np.random.random() <= p and v_t not in v_s.neighbors():
                    G.add_edge(v_s, v_t)
    if not selfloops:
        G = G.simplify()

    if between_communities:
        G.es['weight'] = [2 if np.random.random() < p else 1 for i in range(len(G.es))]
#        print(G.es['weight'])
    return G


def cellular(Na, Nc, d_i, d_o):

    # currently using normal distribution but you can switch to binomial

    # Na - number of nodes
    # Nc - number of cells
    # d_i - density of links within a cell
    # d_o - density of links between cells (if a link exists between cells a node from a cell was selected at random to be connected to another
    # random node from another cell)

    G_main = ig.Graph()

    cell_sizes = random_bounded_numbers_normal(Nc, Na)
    name_start = [sum(cell_sizes[:i]) for i in range(len(cell_sizes))]

    cells = [ER_graph(cell_sizes[i], d_i, name_start[i]) for i in range(len(cell_sizes))]

    name_start.append(Na - 1)
    between_cells = ER_graph(Nc, d_o, selfloops=False, between_communities=True)
    G_main = cells[0].disjoint_union(cells[1:])

    for edge in between_cells.es():

        c_1 = edge.tuple[0]
        c_2 = edge.tuple[1]

#        print(edge['weight'])

        id1_1 = np.random.randint(name_start[c_1], name_start[c_1 + 1])
        id2_1 = np.random.randint(name_start[c_2], name_start[c_2 + 1])

        G_main.add_edge(id1_1, id2_1)

        if edge['weight'] == 2:

            id1_2, id2_2 = id1_1, id2_1

            while (id1_1, id2_1) == (id1_2, id2_2):

                id1_2 = np.random.randint(name_start[c_1], name_start[c_1 + 1])
                id2_2 = np.random.randint(name_start[c_2], name_start[c_2 + 1])

            G_main.add_edge(id1_2, id2_2)

    return G_main


def gen_cellular(number_of_nodes):
    cell_count = np.random.randint(np.floor(number_of_nodes * 0.01), np.floor(number_of_nodes * 0.05))
    internal_density_bounds = [0.1, 0.25]
    external_density_bounds = [0, 0.5]

    while True:
        d_i = np.random.rand() * (internal_density_bounds[1] - internal_density_bounds[0]) + internal_density_bounds[0]
        d_o = np.random.rand() * (external_density_bounds[1] - external_density_bounds[0]) + external_density_bounds[0]
        d_i = round(d_i, 2)
        d_o = round(d_o, 2)
        try:
            g = cellular(number_of_nodes, cell_count, d_i, d_o)
            g.vs['name'] = list(map(str, range(g.vcount())))
            break
        except:
            continue
    return g


def gen_er(number_of_nodes, p=0.015):
    while True:
        g = ig.Graph.Erdos_Renyi(n=number_of_nodes, p=p)
        if g.is_connected():
            break
    return g


def gen_barabasi(number_of_nodes, m=8, power=1.5):
    while True:
        g = ig.Graph.Barabasi(number_of_nodes, m=m, power=power)
        if g.is_connected():
            break
    return g


def simulate_and_test(number_of_nodes=1000, reps=100, graph_type='cellular', centralities=None):
    if not centralities:
        centralities = [cac.modularity_vitality,
                        cac.absolute_modularity_vitality,
                        cac.adjusted_modular_centrality_degree,
                        cac.masuda,
                        cac.community_hub_bridge,
                        cac.weighted_modular_centrality_degree,
                        cac.degree]

    centrality_names = [f.__name__ for f in centralities]
    attacks = ['initial', 'MBA', 'recomputed']
    results = {attack: {f: [[], []] for f in centrality_names} for attack in attacks}

    mods = []

    for rep in range(reps):
        print('running rep', rep)

        if graph_type == 'cellular':
            g = gen_cellular(number_of_nodes)
        elif graph_type == 'er':
            g = gen_er(number_of_nodes)
        elif graph_type == 'barabasi':
            g = gen_barabasi(number_of_nodes)
        else:
            print('wrong graph type')

        if g.is_weighted():
            weight_key = 'weight'
        else:
            weight_key = None

        part = g.community_leiden(objective_function='modularity', weights=weight_key)
        mods.append(part.modularity)

        for centrality in centralities:
            rho, sigma, rho_e = rt.initial_attack(g, part, centrality, calculations=50)
            node_cost = np.trapz(y=sigma, x=rho)
            edge_cost = np.trapz(y=sigma, x=rho_e)

            name_str = centrality.__name__
            results['initial'][name_str][0].append(node_cost)
            results['initial'][name_str][1].append(edge_cost)

        for centrality in centralities:
            rho, sigma, rho_e = rt.module_based_attack(g, part, centrality, calculations=50)
            node_cost = np.trapz(y=sigma, x=rho)
            edge_cost = np.trapz(y=sigma, x=rho_e)
            name_str = centrality.__name__
            results['MBA'][name_str][0].append(node_cost)
            results['MBA'][name_str][1].append(edge_cost)

        for centrality in centralities:
            rho, sigma, rho_e = rt.repeated_attack(g, part, centrality, calculations=50)
            node_cost = np.trapz(y=sigma, x=rho)
            edge_cost = np.trapz(y=sigma, x=rho_e)
            name_str = centrality.__name__
            results['recomputed'][name_str][0].append(node_cost)
            results['recomputed'][name_str][1].append(edge_cost)
    return results


def summarize_results(results):
    for attack in results:
        print(attack, 'results:')
        print()
        print('node costs:')
        for strategy in results[attack]:
            print(strategy)
            print(np.mean(results[attack][strategy][0]))
        print()
        print('edge costs:')
        for strategy in results[attack]:
            print(strategy)
            print(np.mean(results[attack][strategy][1]))
        print()
    return


def main():
    graph_types = ['cellular', 'er', 'barabasi']
    all_results = {}
    for gt in graph_types:
        print(gt, ":")
        results = simulate_and_test(graph_type=gt, centralities=[cac.masuda])
        all_results[gt] = results
    for gt, results in all_results.items():
        summarize_results(results)
