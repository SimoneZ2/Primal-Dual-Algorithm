import numpy as np

def grafo_source_sink(start_nodes, end_nodes, capacities, unit_costs, supplies, flows):
    modified_graph = {
        'nodes': set(),
        'edges': [],
        'supply_s': 0
    }
    supply_s = sum(max(0, supply) for supply in supplies)
    modified_graph['supply_s'] = supply_s
    modified_graph['supply_t'] = -supply_s
    modified_graph['nodes'].add('s')
    modified_graph['nodes'].add('t')
    for node, supply in enumerate(supplies, start=1):
        if supply > 0:
            modified_graph['edges'].append(('s', node, supply, 0,0))
    for node, supply in enumerate(supplies, start=1):
        if supply < 0:
            modified_graph['edges'].append((node, 't', -supply, 0,0))
    for i in range(len(start_nodes)):
        start_node, end_node, capacity, cost, flow = start_nodes[i], end_nodes[i], capacities[i], unit_costs[i], flows[i]
        modified_graph['nodes'].add(start_node)
        modified_graph['nodes'].add(end_node)
        modified_graph['edges'].append((start_node, end_node, capacity, cost, flow))
    return modified_graph

def convert_to_graph_format(modified_graph):
    graph = {'s': {}, 't': {}}

    for edge in modified_graph['edges']:
        start_node, end_node, capacity, cost, flow = edge
        if start_node not in graph:
            graph[start_node] = {}
        graph[start_node][end_node] = [capacity, cost, flow]
    return graph

def dijkstra(graph, start):
    # Inizializzazione delle distanze con infinito per tutti i nodi tranne il nodo di partenza
    distances = {node: float('inf') for node in graph}
    distances[start] = 0

    # Inizializzazione dei nodi visitati
    visited = set()

    while len(visited) < len(graph):
        # Trova il nodo con la distanza minima tra quelli non visitati
        current_node = min((node for node in graph if node not in visited), key=lambda x: distances[x])

        # Aggiorna i nodi vicini
        for neighbor, weight in graph[current_node].items():
            distance = distances[current_node] + weight[1]

            # Aggiorna la distanza se trovato un percorso più breve
            if distance < distances[neighbor]:
                distances[neighbor] = distance

        # Aggiungi il nodo corrente ai visitati
        visited.add(current_node)

    return distances

def calculate_node_potentials(shortest_distances):
    # Inverti i segni delle distanze per ottenere i node potentials
    node_potentials = {node: -distance for node, distance in shortest_distances.items()}
    return node_potentials

def calculate_reduced_costs(node_potentials, graph):
    reduced_costs = {}

    for node_from, neighbors in graph.items():
        for node_to, cost in neighbors.items():
            reduced_cost = cost[1] - node_potentials[node_from] + node_potentials[node_to]
            reduced_costs[(node_from, node_to)] = reduced_cost

    return reduced_costs

def remove_non_zero_reduced_cost_edges(graph, reduced_costs):
    # Lista degli archi da rimuovere
    edges_to_remove = [edge for edge, reduced_cost in reduced_costs.items() if reduced_cost != 0]

    # Rimuovi gli archi con costo ridotto diverso da 0
    for edge in edges_to_remove:
        node_from, node_to = edge
        del graph[node_from][node_to]

    return graph


def create_zero_matrix(graph):
    return [[0] * len(graph) for _ in range(len(graph))]

def from_graph_to_capacitiesMatrix(graph, matrice):
    for arcs in graph['s'].items():
        matrice[0][arcs[0]] = arcs[1][0]

    for arcs in graph['t'].items():
        matrice[0][arcs[0]] = arcs[1][0]

    for arcs in graph.items():
        for arc in arcs[1]:
            # print(arc)
            # print(arcs[1].items())
            # sos = list(arcs[1].items())[0]
            # print(sos[1][0])
            arcs[1].items()
            if ((arcs[0] != 's') & (arcs[0] != 't') & (arc != 's') & (arc != 't')):
                capacities = list(arcs[1].items())[0]
                matrice[arcs[0]][arc] = capacities[1][0]
            elif (arc == 't'):
                capacities = list(arcs[1].items())[0]
                matrice[arcs[0]][len(graph) - 1] = capacities[1][0]
            elif(arc == 's'):
                capacities = list(arcs[1].items())[0]
                matrice[arcs[0]][0] = capacities[1][0]
    return matrice


def BFS(graph, s, t, parent):
    # Return True if there is node that has not iterated.
    visited = [False] * len(graph)
    queue = []
    queue.append(s)
    visited[s] = True

    while queue:
        u = queue.pop(0)
        for ind in range(len(graph[u])):
            if visited[ind] is False and graph[u][ind] > 0:
                queue.append(ind)
                visited[ind] = True
                parent[ind] = u

    return True if visited[t] else False


def FordFulkerson(graph, source, sink):
    # This array is filled by BFS and to store path
    parent = [-1] * (len(graph))
    max_flow = 0
    while BFS(graph, source, sink, parent):
        path_flow = float("Inf")
        s = sink

        while s != source:
            # Find the minimum value in select path
            path_flow = min(path_flow, graph[parent[s]][s])
            s = parent[s]

        max_flow += path_flow
        v = sink

        while v != source:
            u = parent[v]
            graph[u][v] -= path_flow
            graph[v][u] += path_flow
            v = parent[v]
    return max_flow

def update_graph(mfp_result, grafo):
    grafo['supply_s'] = grafo['supply_s'] - mfp_result
    grafo['supply_t'] = grafo['supply_t'] + mfp_result









start_nodes = np.array([1, 1, 2, 2])
end_nodes = np.array([3, 4, 3, 4])
capacities = np.array([1, 1, 1, 1])
unit_costs = np.array([1, 2, 1, 2])
flow = np.array([0, 0, 0, 0])
supplies = [2, 2, -2, -2]

grafo_conST = grafo_source_sink(start_nodes, end_nodes, capacities, unit_costs, supplies, flow) #grafo con source s e sink t


print("Grafo Modificato:")
print("Nodi:", grafo_conST['nodes'])
print("Archi:", grafo_conST['edges'])
print("Supply di s:", grafo_conST['supply_s'])
print("Supply di t:", grafo_conST['supply_t'])



grafo_conST_formato = convert_to_graph_format(grafo_conST)
print("Grafo in diverso formato:")
print(grafo_conST_formato) #Grafo su cui useremo l'algortimo di Djkstra

start_node = 's'
shortest_distances = dijkstra(grafo_conST_formato, start_node)

node_potentials = calculate_node_potentials(shortest_distances)
for node, potential in node_potentials.items():
    print(f'Node Potential di {node}: {potential}')


reduced_costs = calculate_reduced_costs(node_potentials, grafo_conST_formato)
print("\nCosti Ridotti degli Archi:")
for edge, reduced_cost in reduced_costs.items():
    print(f'Costo ridotto dell\'arco {edge}: {reduced_cost}')



admissible_network = remove_non_zero_reduced_cost_edges(grafo_conST_formato, reduced_costs)
print("Rete Ammissibile:")
for node, neighbors in admissible_network.items():
    print(f'{node}: {neighbors}')

matrice_capacita = from_graph_to_capacitiesMatrix(admissible_network, create_zero_matrix(admissible_network))
print(matrice_capacita)

graph = matrice_capacita
source, sink = 0, 5
mfp_result = FordFulkerson(graph, source, sink)
print("Valore MFP:", mfp_result)
print("Grafo residui attuale:", graph)


update_graph(mfp_result, grafo_conST)
print("Nuovo e(s):",grafo_conST['supply_s'])
print("Nuovo e(t):",grafo_conST['supply_t'])



def update_graph_with_capacities(original_graph, flow_matrix):
    updated_graph = {node: {} for node in original_graph.keys()}

    for i, row in enumerate(flow_matrix):
        for j, capacity in enumerate(row):
            #if capacity > 0:
                # Trova l'arco corrispondente nel grafo originale
                for node, neighbors in original_graph.items():
                    for neighbor, (original_capacity, cost) in neighbors.items():
                        if node=='s':
                            node = 0
                        if node=='t':
                            node = len(graph)-1
                        if i == node and j == neighbor:
                            # Aggiorna la capacità nel grafo originale
                            if node==0:
                                node='s'
                            if node==len(graph)-1:
                                node='t'
                            updated_graph[node][neighbor] = [capacity, cost]

    return updated_graph
# Esempio di utilizzo
original_graph = {'s': {1: [2, 0], 2: [2, 0]}, 't': {}, 3: {'t': [2, 0]}, 4: {'t': [2, 0]}, 1: {3: [1, 1], 4: [1, 2]}, 2: {3: [1, 1], 4: [1, 2]}}
flow_matrix = [[0, 1, 1, 0, 0, 0], [1, 0, 0, 0, 1, 0], [1, 0, 0, 0, 1, 0], [0, 1, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 2, 0, 0]]
updated_graph = update_graph_with_capacities(original_graph, flow_matrix)
# Stampa il grafo aggiornato
print(updated_graph)

def update_flow():
    modified_graph = {
        'nodes': set(),
        'edges': [],
        'supply_s': 2,
        'supply_t': -2
    }
    start_nodes = ['s','s',1,1,2,2,3,3,4,'t']
    end_nodes=    [1,2,'s',4,'s',4,1,2,'t',3]
    capacities=   [1,1,1,1,1,1,1,1,2,2]
    unit_costs=   [0,0,0,2,0,2,1,1,0,0]
    flows=        [1,1,1,1,1,1,1,1,0,2]
    supplies=     [2,0,0,0,0,-2]

    for i in range(len(start_nodes)):
        start_node, end_node, capacity, cost, flow = start_nodes[i], end_nodes[i], capacities[i], unit_costs[i], flows[i]
        modified_graph['nodes'].add(start_node)
        modified_graph['nodes'].add(end_node)
        modified_graph['edges'].append((start_node, end_node, capacity, cost, flow))


    #nuovo_grafo = modify_graph(start_nodes, end_nodes, capacities, unit_costs, supplies, flow)
    return modified_graph

nuovo_grafo = update_flow()
print("NUOVO Grafo :")
print("Nodi:", nuovo_grafo['nodes'])
print("Archi:", nuovo_grafo['edges'])
print("Supply di s:", nuovo_grafo['supply_s'])
print("Supply di t:", nuovo_grafo['supply_t'])

grafo_conST_formato_nuovo = convert_to_graph_format(nuovo_grafo)
start_node = 's'
shortest_distances = dijkstra(grafo_conST_formato_nuovo, start_node)

node_potentials = calculate_node_potentials(shortest_distances)
for node, potential in node_potentials.items():
    print(f'Node Potential di {node}: {potential}')


reduced_costs = calculate_reduced_costs(node_potentials, grafo_conST_formato_nuovo)
print("\nCosti Ridotti degli Archi:")
for edge, reduced_cost in reduced_costs.items():
    print(f'Costo ridotto dell\'arco {edge}: {reduced_cost}')



admissible_network = remove_non_zero_reduced_cost_edges(grafo_conST_formato_nuovo, reduced_costs)
print("Rete Ammissibile:")
for node, neighbors in admissible_network.items():
    print(f'{node}: {neighbors}')

matrice_capacita = from_graph_to_capacitiesMatrix(admissible_network, create_zero_matrix(admissible_network))
print(matrice_capacita)

graph_nuovo = matrice_capacita
source, sink = 0, 5
mfp_result = FordFulkerson(graph_nuovo, source, sink)
print("Valore MFP:", mfp_result)
print("Grafo residui attuale:", graph_nuovo)













