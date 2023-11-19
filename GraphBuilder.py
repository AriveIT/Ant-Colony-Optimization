import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from math import sqrt

class GraphBuilder:

    def __init__(self, scale=1):
        self.edges = []
        self.nodes = []
        self.positions = []
        self.Graph = nx.Graph()
        self.isAssembled = False
        self.scale = scale
    
    def addWeightedEdge(self, a, b):
        w = self._euclidianDistance(a, b)
        self.edges.append([a, b, {'weight': w}])
    
    def addNode(self, a, pos):
        self.nodes.append(a)
        self.positions.append(pos)
    
    def assembleGraph(self):
        self.Graph.add_edges_from(self.edges)
        self.Graph.add_nodes_from(self.nodes)
        self.isAssembled = True
    
    def getGraphWeight(self):
        return sum(nx.get_edge_attributes(self.Graph, 'weight').values())
    
    def getPathWeight(self, path):
        weights = nx.adjacency_matrix(self.Graph)
        return sum([weights[edge] for edge in path])

    def _euclidianDistance(self, a, b):
        pos1 = self.positions[a]
        pos2 = self.positions[b]
        return sqrt((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2)
    
    # ----------------
    # Assembling graph
    # ----------------
    @staticmethod
    def complete_graph(n, scale=1):
        gb = GraphBuilder(scale)
        GraphBuilder.add_random_vertices(gb, n, 10)
        GraphBuilder.add_all_edges(gb, n)
        gb.assembleGraph()
        return gb

    @staticmethod
    def add_random_vertices(graphbuilder, n, scale):
        pos = np.random.rand(n, 2)
        pos[:,0] *= scale
        pos[:,1] *= scale

        for i in range(len(pos)):
            graphbuilder.addNode(i, pos[i])

    @staticmethod
    def add_all_edges(graphbuilder, n):
        for i in range(n):
            for j in range(i + 1, n):
                graphbuilder.addWeightedEdge(i, j)

