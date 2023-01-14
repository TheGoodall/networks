



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

graph = load_graph("alg_phys-cite.txt")

# Make graph connected
for node in graph:
    for otherNode in graph[node]:
        graph[otherNode].add(node)

# Create component list
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

G_nodes = max(components, key=lambda component:len(component))

with open("G_Size.data", "w") as f:
    f.write(str(len(G_nodes)))

def compute_in_degrees(digraph):
	"""Takes a directed graph and computes the in-degrees for the nodes in the
	graph. Returns a dictionary with the same set of keys (nodes) and the
	values are the in-degrees."""
	#initialize in-degrees dictionary with zero values for all vertices
	in_degree = {}
		
	for vertex in digraph:
		in_degree[vertex] = 0
	#consider each vertex
	for vertex in digraph:
		#amend in_degree[w] for each outgoing edge from v to w
		for neighbour in digraph[vertex]:
			in_degree[neighbour] += 1
	return in_degree

def in_degree_distribution(digraph):
	"""Takes a directed graph and computes the unnormalized distribution of the
	in-degrees of the graph.  Returns a dictionary whose keys correspond to
	in-degrees of nodes in the graph and values are the number of nodes with
	that in-degree. In-degrees with no corresponding nodes in the graph are not
	included in the dictionary."""
	#find in_degrees
	in_degree = compute_in_degrees(digraph)
	#initialize dictionary for degree distribution
	degree_distribution = {}
	#consider each vertex
	for vertex in in_degree:
		#update degree_distribution
		if in_degree[vertex] in degree_distribution:
			degree_distribution[in_degree[vertex]] += 1
		else:
			degree_distribution[in_degree[vertex]] = 1
	return degree_distribution

def compute_out_degrees(digraph):
	out_degree = {}
		
	for vertex in digraph:
		out_degree[vertex] = 0
	#consider each vertex
	for vertex in digraph:
		#amend in_degree[w] for each outgoing edge from v to w
		for _ in digraph[vertex]:
			out_degree[vertex] += 1
	return out_degree

def out_degree_distribution(digraph):
	"""Takes a directed graph and computes the unnormalized distribution of the
	out-degrees of the graph.  Returns a dictionary whose keys correspond to
	out-degrees of nodes in the graph and values are the number of nodes with
	that out-degree. Out-degrees with no corresponding nodes in the graph are not
	included in the dictionary."""
	#find out_degrees
	out_degree = compute_out_degrees(digraph)
	#initialize dictionary for degree distribution
	degree_distribution = {}
	#consider each vertex
	for vertex in out_degree:
		#update degree_distribution
		if out_degree[vertex] in degree_distribution:
			degree_distribution[out_degree[vertex]] += 1
		else:
			degree_distribution[out_degree[vertex]] = 1
	return degree_distribution

def normalise(distro):
    distro = sorted(list(distro.items()))
    distro_total = sum([line[1] for line in distro])
    return [(line[0], line[1]/distro_total) for line in distro]

def output_distro(filename, distro):
    distro_output =  [str(line[0]) + ", " + format(line[1], '.22f') for line in distro]
    with open(filename, "w") as f:
        f.write("degree, count\n")
        for line in distro_output:
            f.write(line + "\n")



G = {}
graph = load_graph("alg_phys-cite.txt")

for node in G_nodes:
    G[node] = set(filter(lambda n : n in G_nodes, graph[node]))

in_distro = normalise(in_degree_distribution(G))
out_distro = normalise(out_degree_distribution(G))

output_distro("out_distro.data", out_distro)
output_distro("in_distro.data", in_distro)


import numpy
from tqdm import tqdm

# probability of edge forming with a given node e.g. in distribution
def p(G):
    count_dist = [max(len(list(filter(lambda x: i in x, G.values()))), 1) for i in G]
    total = sum(count_dist)
    return [i/total for i in count_dist]

# probability of a given number of citations, e.g. out distribution
def s(G):
    return min(len(G), int(numpy.random.exponential(50)))

def edge_func(G):
    return set(
            numpy.random.choice(
                list(G.keys()),
                p=p(G),
                size=s(G),
                replace=False
                )
            )

G = {1:set([1])}
for i in tqdm(range(3000)):
    G[i] = edge_func(G)

output_distro("PA_out_distro.data", normalise(out_degree_distribution(G)))
output_distro("PA_in_distro.data", normalise(in_degree_distribution(G)))
