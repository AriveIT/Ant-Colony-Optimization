import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from GraphVisualization import GraphVisualization as gv
from GraphBuilder import GraphBuilder

min_weight = float("inf") # weight for brute force algorithm

def main():
    np.random.seed(seed=1234)
    np.set_printoptions(precision=4, suppress=True, linewidth=200)
    #ex1(5)
    ex2(20)

# nearest_neighbours and brute force
def ex1(n):
    gb = GraphBuilder.complete_graph(n, scale=10)
    g = gb.Graph

    titles = ["default", "nearest_neighbours", "brute_force"]
    path1 = default_tsp(g, n)
    path2 = nearest_neighbour(g, n)
    path3 = brute_force(g, n)

    gv.visualizeMultiplePaths(gb, [path1, path2, path3], titles)

# nearest_neighbours vs ACO 
def ex2(n):
    gb = GraphBuilder.complete_graph(n, scale=10)
    g = gb.Graph

    p0 = nearest_neighbour(g, n)
    p1, lengths, best_lengths, pheromones = ant_colony(g, n)
    gv.visualizeAOC(gb, p0, p1, lengths, best_lengths, pheromones)

# -----------------------------
# Ant Colony Optimization (ACO)
# -----------------------------
def ant_colony(graph, n, max_iterations=100):
    # ant colony optimization parameters
    alpha = 1.0 # pheromone weight
    beta = 2.0 # greedy weight
    e = 0.01 # evaporation rate
    num_ants = 10 # number of ants simulated per batch

    # initializations
    tau0 = get_initial_pheromone_weight(graph, n, num_ants)
    pheromones = np.array([[tau0] * n]  * n, dtype=np.float64)
    distances = nx.adjacency_matrix(graph)
    
    best_length = float("inf")
    best_path = []

    # plotting
    lengths = []
    best_lengths = []

    # run batches
    for _ in range(max_iterations):
        path, path_length = run_aco_batch(graph, n, distances, pheromones, alpha, beta, e, num_ants)
        
        #print(f"length: {round(path_length, 4)}, path: {path}")
        #print(f"{pheromones}\n")

        if path_length < best_length:
            best_length = path_length
            best_path = path

        lengths.append(path_length)
        best_lengths.append(best_length)

    return best_path, lengths, best_lengths, pheromones

def get_initial_pheromone_weight(graph, n, num_ants):
    path = nearest_neighbour(graph, n)

    # convert path from edges to nodes
    path_nodes = [edge[0] for edge in path]
    path_nodes.append(path[0][0])

    total_weight = nx.path_weight(graph, path_nodes, "weight")

    return num_ants / total_weight
    

def run_aco_batch(graph, n, distances, pheromones, alpha, beta, e, num_ants):
    best_length = float("inf")
    best_path = []

    ant_paths = []
    delta_pheromones = []

    # evaporate
    pheromones *= (1 - e)

    # run the batch
    for i in range(num_ants):
        
        path, path_length = get_random_ant_path_from(graph, n, distances, pheromones, alpha, beta, np.random.randint(0, n))

        if path_length < best_length:
            best_length = path_length
            best_path = path
        
        # track path and delta_pheromones
        ant_paths.append(path)
        delta_pheromones.append(1 / path_length) # Q = 1


    # update weights
    for i in range(num_ants):
        for edge in ant_paths[i]:
                n1 = edge[0]
                n2 = edge[1]
                pheromones[n1, n2] += delta_pheromones[i]
                pheromones[n2, n1] += delta_pheromones[i]
    """
    # update weights after normalizing
    sum = np.sum(pheromones)
    for i in range(n):
        for j in range(n):
            if i == j: continue
            pheromones[i, j] = 2 * pheromones[i, j] / sum
    """

    return best_path, best_length


def get_random_ant_path_from(graph, n, distances, pheromones, alpha, beta, root):
    path = []
    path_distance = 0
    cur_node = root
    visited_nodes = [0] * n
    visited_nodes[root] = 1

    while len(path) < n - 1:
        possible_next = []
        probability_sum = 0

        # get probabilities to go to next node
        for node in graph.neighbors(cur_node):
            if visited_nodes[node] == 1:
                continue
            
            possible_next.append(node)
            probability_sum += get_transition_probability(cur_node, node, alpha, beta, pheromones, distances)
        
        # randomly decide which next node to pick
        random_num = np.random.random() * probability_sum
        cur_probability = 0
        for node in possible_next:
            cur_probability += get_transition_probability(cur_node, node, alpha, beta, pheromones, distances)
            if random_num < cur_probability:
                path.append((cur_node, node))
                path_distance += distances[cur_node, node]
                cur_node = node
                visited_nodes[node] = 1
                break

    # return to root
    path.append((cur_node, root))
    path_distance += distances[cur_node, root]
            
    return path, path_distance

def get_transition_probability(i, j, alpha, beta, pheromones, distances):
    pheromone_level = max(pheromones[i, j], 1e-5)
    return (pheromone_level ** alpha) * (distances[i,j] ** -beta)



# ------------
# Default TSP
# ------------
def default_tsp(G, n):
    return [(i, i+1) for i in range(n-1)]

# -----------------
# Nearest Neighbour
# -----------------
def nearest_neighbour(graph, n):
    A = nx.adjacency_matrix(graph)
    
    visited_nodes = [0] * n
    visited_nodes[0] = 1 # 1 = visited, 0 = not visited
    path = []
    cur_node = 0

    for _ in range(n-1):
        min_weight = float("inf")
        next_node = None

        # find closest adjacent node not already visited
        for n in graph.neighbors(cur_node):
            if visited_nodes[n] == 0 and A[cur_node, n] < min_weight:
                min_weight = A[cur_node, n]
                next_node = n

        path.append((cur_node, next_node))
        visited_nodes[next_node] = 1
        cur_node = next_node

    path.append((cur_node, 0))
    return path

# ---------------
# Brute Force
# ---------------
def brute_force(graph, n):
    root = 0
    visited_nodes = [0] * n
    visited_nodes[root] = 1 # we have already visited root
    best_path = brute_force_recursion(
        graph, root, root, visited_nodes,
        path=[], best_path=[], weight=0)

    return best_path

def brute_force_recursion(graph, root, cur_node, visited_nodes, path, best_path, weight):
    global min_weight
    
    # prune if already longer than best path
    if weight > min_weight:
        return best_path
    
    # if traversed whole graph, return to root
    if len(path) == len(graph) - 1:
        weight += nx.adjacency_matrix(graph)[cur_node, root]

        # if this path is best so far, update best_path
        if weight < min_weight:
            min_weight = weight
            best_path = [edge for edge in path]
            best_path.append((cur_node, root))
        
        return best_path


    # continue traversal
    for node in graph.neighbors(cur_node):
        # if already visited this node, continue
        if visited_nodes[node] == 1:
            continue

        # updated visited nodes + path
        visited_nodes[node] = 1
        path.append((cur_node, node))

        # continue traversal
        edge_weight = nx.adjacency_matrix(graph)[cur_node, node]
        best_path = brute_force_recursion(
            graph, root, node, visited_nodes,
            path, best_path, weight + edge_weight
            )

        # undo visited nodes + path so we can visit next node
        visited_nodes[node] = 0
        del path[-1]

    return best_path

if __name__ == "__main__":
    main()
