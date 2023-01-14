def load_graph(graph_txt):
    """
    Loads a graph from a text file.
    Then returns the graph as a dictionary.
    """
    graph = open(graph_txt)
    
    answer_graph = {}
    nodes = 0
    for line in graph:
        neighbors = line.split(' ')
        node = int(neighbors[0])
        answer_graph[node] = set([])
        for neighbor in neighbors[1 : -1]:
            answer_graph[node].add(int(neighbor))
        nodes += 1
    print ("Loaded graph with", nodes, "nodes")
    
    return answer_graph

net = {
            1: [4],
            2: [4],
            3: [4],
            4: [1, 2, 3, 5, 6],
            5: [4],
            6: [4, 7, 8, 9, 10, 11],
            7: [6, 8, 11],
            8: [6, 7, 9, 11],
            9: [6, 8, 10],
            10: [6, 9, 11, 12],
            11: [6, 7, 8, 10],
            12: [10]
        }

def breadth_first(n_from, n_to, net) -> int:
    q: list[tuple[int, int]] = [(n_from, 0)]
    searched = []
    while q:
        c = q.pop(0)
        searched.append(c[0])
        for n in net[c[0]]:
            if n == n_to:
                return c[1]+1
            if n not in searched:
                q.append((n, c[1]+1))
    return 0


import random
def sample(n, net):
    
    return random.sample(list(net.keys()), k=min(len(net), n))

def closeness(net):
    return [1/sum([breadth_first(i, j, net)for i in filter(lambda x: x != j, sample(4, net))]) for j in sample(30, net)]

def nearness(net):
    return [sum([1/breadth_first(i, j, net)for i in filter(lambda x: x != j, sample(4, net))]) for j in sample(30, net)]

def degree(net):
    return [len(net[j]) for j in net]

def adjacency(net):
    d = {}
    for j in net:
        d[j] = len(net[j])

    return [sum([(d[j]-d[n])/(d[j]+d[n]) for n in net[j] ])/d[j] for j in net]


with open("topic_three_one_stats.data", "w") as f:
    f.write("Closeness:\n{}\n\nNearness:\n{}\n\nDegree\n{}\n\nAdjacency\n{}".format(closeness(net), nearness(net), degree(net), adjacency(net)))

def load_london():
    graph = open("topic3networks/london_transport_raw.edges.txt")
    
    answer_graph = {}
    for line in graph:
        stations = line.strip("\n").split(' ')
        station1 = stations[1]
        station2 = stations[2]
        try:
            _ = answer_graph[station1]
        except KeyError:
            answer_graph[station1] = set([])
        try:
            _ = answer_graph[station2]
        except KeyError:
            answer_graph[station2] = set([])
        
        answer_graph[station1].add(station2)
        answer_graph[station2].add(station1)
    
    return answer_graph

def load_roget():
    graph = open("topic3networks/Roget.txt")
    g = graph.readlines()
    nodes = {}

    _ = g.pop(0)
    n = g.pop().strip()
    while n != "*Arcslist":
        sp = n.split(" ")
        nodes[int(sp[0])] = sp[1]
        n = g.pop(0).strip()

    answer_graph = {}

    for line in g:
        nodes = line.strip("\n").split(' ')
        for node in nodes:
            for othernode in nodes:
                if node != othernode:
                    try:
                        _ = answer_graph[node]
                    except KeyError:
                        answer_graph[node] = set([])
                    answer_graph[node].add(othernode)

    return answer_graph

def load_CCSB():
    graph = open("topic3networks/CCSB-Y2H.txt")
    
    answer_graph = {}
    for line in graph:
        if line == "from http://interactome.dfci.harvard.edu/S_cerevisiae/index.php?page=home\n":
            continue
        stations = line.strip("\n").split('\t')
        station1 = stations[0]
        station2 = stations[1]
        try:
            _ = answer_graph[station1]
        except KeyError:
            answer_graph[station1] = set([])
        try:
            _ = answer_graph[station2]
        except KeyError:
            answer_graph[station2] = set([])
        
        answer_graph[station1].add(station2)
        answer_graph[station2].add(station1)
    
    return answer_graph


def find_largest_component(graph):
    components = []
    while graph:
        
        first = graph.popitem()

        component = [first[0]]
        nodequeue = set()
        for node in first[1]:
            if node not in component:
                nodequeue.add(node)
        while nodequeue:
            node = nodequeue.pop()
            component.append(node)
            for newNode in graph.pop(node):
                if newNode not in component:
                    nodequeue.add(newNode)
        components.append(component)

    return max(components, key=lambda component:len(component))

def strip(G, sub):
    
    G2 = {}
    for node in G:
        if node in sub:
            G2[node] = G[node]
    G_out =  {}
    for node in G2:
        n = []
        for neighbour in G2[node]:
            if neighbour in sub:
                n.append(neighbour)
        G_out[node] = n
    return G_out
def a():
    print("loading ccsb")
    CCSB_graph = load_CCSB()
    G_CCSB = strip(CCSB_graph, find_largest_component(load_CCSB()))
    print("working on CCSB")
    with open("ccsb.data", "w") as f:
        f.write("Closeness:\n{}\n\nNearness:\n{}\n\nDegree\n{}\n\nAdjacency\n{}".format(
            sorted(closeness(   G_CCSB), reverse=True)[:20],
            sorted(nearness(    G_CCSB), reverse=True)[:20],
            sorted(degree(      G_CCSB), reverse=True)[:20],
            sorted(adjacency(   G_CCSB), reverse=True)[:20]
            )
        )

def b():
    print("loading london")
    london_graph = load_london()
    G_london = strip(london_graph, find_largest_component(load_london()))
    print("working on London")
    with open("london.data", "w") as f:
        f.write("Closeness:\n{}\n\nNearness:\n{}\n\nDegree\n{}\n\nAdjacency\n{}".format(
            sorted(closeness(G_london), reverse=True)[:20],
            sorted(nearness(G_london), reverse=True)[:20],
            sorted(degree(G_london), reverse=True)[:20],
            sorted(adjacency(G_london), reverse=True)[:20]
            )
        )
def c():
    print("loading roget")
    Roget = load_roget()
    G_roget = strip(Roget, find_largest_component(load_roget()))
    print("working on roget")
    with open("roget.data", "w") as f:
        f.write("Closeness:\n{}\n\nNearness:\n{}\n\nDegree\n{}\n\nAdjacency\n{}".format(
            sorted(closeness(   G_roget), reverse=True)[:20],
            sorted(nearness(    G_roget), reverse=True)[:20],
            sorted(degree(      G_roget), reverse=True)[:20],
            sorted(adjacency(   G_roget), reverse=True)[:20]
            )
        )

from multiprocessing import Process
if __name__ == "__main__":
    pa = Process(target=a)
    pb = Process(target=b)
    pc = Process(target=c)
    pa.start()
    pb.start()
    pc.start()
    pa.join()
    pb.join()
    pc.join()
