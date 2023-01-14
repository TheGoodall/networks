import numpy as np
import random

def L() -> int:
    while 1:
        l = np.random.poisson(25)
        if l > 0:
            return l
    return 0



def make_vdws_graph(num_nodes, rewiring_prob):
    
    #initialize empty graph
    ws_graph = {}
    for vertex in range(num_nodes): ws_graph[vertex] = set()

    localdegrees = [L() for i in range(num_nodes)]


    #add edges from each vertex to neighbours
    for vertex in range(num_nodes):
        for neighbour in range(vertex - localdegrees[vertex], vertex + localdegrees[vertex] + 1):
            neighbour = neighbour % num_nodes
            if neighbour != vertex:
                ws_graph[vertex].add(neighbour)
                ws_graph[neighbour].add(vertex)

    #rewiring
    # for vertex in range(num_nodes):
        # temp_neighbours = ws_graph[vertex].copy()
        # for neighbour in temp_neighbours:
            # random_number = random.random()
            # if random_number < rewiring_prob:
                # random_node = random.randint(0, num_nodes-1)
                # if random_node != vertex and random_node != neighbour and neighbour in ws_graph[vertex] and vertex in ws_graph[neighbour]:
                    # ws_graph[vertex].remove(neighbour)
                    # ws_graph[neighbour].remove(vertex)
                    # ws_graph[vertex].add(random_node)
                    # ws_graph[random_node].add(vertex)
    for vertex in range(num_nodes):                                             #consider each vertex
        for neighbour in range(vertex + 1, vertex + localdegrees[vertex] + 1):  #consider each clockwise neighbour
            neighbour = neighbour % num_nodes                                   #correct node label if value too high
            random_number = random.random()                                     #generate random number
            if random_number < rewiring_prob:                                   #decide whether to rewire
                random_node = random.randint(0, num_nodes-1)                    #choose random node
                if random_node != vertex and random_node not in ws_graph[vertex]:   #make sure no loops or duplicate edges
                    ws_graph[vertex].remove(neighbour)                          #delete edge from dictionary          
                    ws_graph[neighbour].remove(vertex)                          #in two places
                    ws_graph[vertex].add(random_node)                           #add new edge to dictionary
                    ws_graph[random_node].add(vertex)                           #in two places  
    return ws_graph

def iterate(G, t, vaccination_sample, pis, piv, pvis, pviv):
    # Roll infections
    for node in list(filter(lambda x: G[x]["state"] in ["I", "VI"], G.keys())):
        if G[node]["state"] == "I":
            for neighbour in G[node]["edges"]:
                random_number = random.random()
                if G[neighbour]["state"] == "S":
                    if random_number < pis:
                        G[neighbour]["state"] = "I"
                        G[neighbour]["infected_for"] = 0
                elif G[neighbour]["state"] == "V":
                    if random_number < piv:
                        G[neighbour]["state"] = "VI"
                        G[neighbour]["infected_for"] = 0

        if G[node]["state"] == "VI":
            for neighbour in G[node]["edges"]:
                random_number = random.random()
                if G[neighbour]["state"] == "S":
                    if random_number < pvis:
                        G[neighbour]["state"] = "I"
                        G[neighbour]["infected_for"] = 0
                elif G[neighbour]["state"] == "V":
                    if random_number < pviv:
                        G[neighbour]["state"] = "VI"
                        G[neighbour]["infected_for"] = 0
    # Roll Vaccinations
    if t >= 50:
        for node in vaccination_sample(G):
            if G[node]["state"] == "I":
                G[node]["state"] = "VI"
            elif G[node]["state"] == "S":
                G[node]["state"] = "V"

    # Roll recovery

    for node in G:
        if G[node]["state"] == "I" or G[node]["state"] == "VI":
            if G[node]["infected_for"] >= 3:
                G[node]["state"] = "R"
                G[node]["infected_for"] = None
            else:
                G[node]["infected_for"] += 1

    return t+1

def vaccine_uniform(G):
    elegible = list(filter(lambda x: G[x]["state"] != "V" and G[x]["state"] != "R" and G[x]["state"] != "VI", G.keys()))
    return random.sample(elegible, k=min(len(elegible), 400))

from tqdm import tqdm

def trial(name, vaccine, pis, piv, pvis, pviv):
    graph = make_vdws_graph(200000, 0.01)


    SIR = {}
    for node in graph.items():
        SIR[node[0]] = {"state": "S", "infected_for":None, "edges": node[1]}

    initial_inf = random.sample(list(SIR.keys()), k=5)
    for inf in initial_inf:
        SIR[inf]["state"] = "I"
        SIR[inf]["infected_for"] = 0
    t = 0
    with open(name+".data", "w") as f:
        f.write("Timestep, S, V, S+V, I+VI, R, I, VI\n")
        for i in tqdm(range(400)):
            t = iterate(SIR, t, vaccine, pis, piv, pvis, pviv)
            states = {"S":0, "I": 0, "V":0, "VI":0, "R":0}
            for node in SIR:
                states[SIR[node]["state"]] += 1
            f.write("{}, {}, {}, {}, {}, {}, {}, {}\n".format(t, states['S'], states['V'], states['S'] + states['V'] + states['R'], states['I'] + states['VI'], states['R'], states['I'], states['VI']))
            


# trial("Disease1", vaccine_uniform, 0.01, 0.01, 0.01, 0.01)
# trial("Disease2", vaccine_uniform, 0.01, 0.005, 0.005, 0.0025)
# trial("Disease3", vaccine_uniform, 0.01, 0.001, 0.001, 0.0001)


# Degree centrality
def vaccine_degree_max(G):
    elegible = list(filter(lambda x: G[x]["state"] != "V" and G[x]["state"] != "R" and G[x]["state"] != "VI", G.keys()))
    elegible.sort(key=lambda x: len(G[x]["edges"]), reverse=True)
    return elegible[:min(len(elegible), 400)]
# trial("Degree_max1", vaccine_degree_max, 0.01, 0.01, 0.01, 0.01)
# trial("Degree_max2", vaccine_degree_max, 0.01, 0.005, 0.005, 0.0025)
# trial("Degree_max3", vaccine_degree_max, 0.01, 0.001, 0.001, 0.0001)

# Degree centrality probabalistic
def vaccine_degree_prob(G):
    elegible = list(filter(lambda x: G[x]["state"] != "V" and G[x]["state"] != "R" and G[x]["state"] != "VI", G.keys()))
    total_degree = sum([len(G[node]["edges"]) for node in elegible])
    p=[len(G[node]["edges"])/total_degree for node in elegible]
    if len(p) == 0:
        return []
    return np.random.choice(elegible, size=min(len(elegible), 400),p=p)
trial("Degree_prob1", vaccine_degree_prob, 0.01, 0.01, 0.01, 0.01)
trial("Degree_prob2", vaccine_degree_prob, 0.01, 0.005, 0.005, 0.0025)
trial("Degree_prob3", vaccine_degree_prob, 0.01, 0.001, 0.001, 0.0001)

# Adjacency centrality
def adjacency(node, G):
    dj = len(G[node]["edges"])
    s = sum([(dj-len(G[neighbour]["edges"]))/(dj+(len(G[neighbour]["edges"]))) for neighbour in G[node]["edges"] ])
    return s/dj


def vaccine_adjacency(G):
    elegible = list(filter(lambda x: G[x]["state"] != "V" and G[x]["state"] != "R" and G[x]["state"] != "VI", G.keys()))
    elegible = sorted(elegible, key=lambda x: adjacency(x,G), reverse=True)
    return elegible[:min(len(elegible), 400)]
trial("Adjacency1", vaccine_adjacency, 0.01, 0.01, 0.01, 0.01)
trial("Adjacency2", vaccine_adjacency, 0.01, 0.005, 0.005, 0.0025)
trial("Adjacency3", vaccine_adjacency, 0.01, 0.001, 0.001, 0.0001)
