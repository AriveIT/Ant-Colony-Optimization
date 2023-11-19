import networkx as nx
import matplotlib.pyplot as plt
import math

class GraphVisualization:

    @staticmethod
    def visualize(gb):
        nx.draw_networkx(gb.Graph, gb.positions, with_labels=False, node_size=100)
        plt.axis('off')
        plt.show()
    
    @staticmethod
    def visualizePath(gb, path):
        nx.draw_networkx(gb.Graph, gb.positions, edgelist=path)
        plt.axis('off')
        plt.show()
    
    @staticmethod
    def visualizeMultiple(gb):

        for i, gb in enumerate(gb):
            plt.figure(i)
            nx.draw_networkx(gb.Graph, gb.positions, with_labels=False, node_size=100)
        
        plt.axis('off')
        plt.show()

    @staticmethod
    def visualizeMultiplePaths(gb, paths, titles=None):
        if titles == None:
            titles = []

        subplotDim = GraphVisualization._getDimensionsForSubplots(len(paths))
        GraphVisualization._formatTitlesArray(titles, len(paths))


        for i, path in enumerate(paths):
            title = f"{titles[i]}: {round(gb.getPathWeight(path), 4)}"
            plt.subplot(subplotDim + i + 1)
            plt.title(title)
            nx.draw_networkx(gb.Graph, gb.positions, with_labels=True, node_size=25, edgelist=path)

        plt.axis('off')
        plt.show()

    def visualizeAOC(gb, nearest_neighbour, aoc, lengths, best_lengths, pheromones):
        dim = 140

        # nearest neighbours for reference
        plt.figure(figsize=(12,10))
        plt.subplot(dim + 1)
        plt.title(f"nearest: {round(gb.getPathWeight(nearest_neighbour), 4)}")
        nx.draw_networkx(gb.Graph, gb.positions, with_labels=True, node_size=25, edgelist=nearest_neighbour)
        
        # final ACO
        plt.subplot(dim + 2)
        plt.title(f"ACO: {round(gb.getPathWeight(aoc), 4)}")
        nx.draw_networkx(gb.Graph, gb.positions, with_labels=True, node_size=25, edgelist=aoc)

        # lenghts over time
        plt.subplot(dim + 3)
        plt.title("ACO over time")
        plt.plot(lengths)
        plt.plot(best_lengths)

        # build pheromone graph
        pheromone_graph = nx.from_numpy_array(pheromones)
        max_value = pheromones.max()
        n = gb.Graph.number_of_nodes()

        plt.subplot(dim + 4)
        plt.title("Pheromones")

        # set edge opacity based on pheromone value
        for i in range(n):
            for j in range(n):
                edge = (i, j)
                nx.draw_networkx_edges(pheromone_graph, gb.positions, edgelist=[edge], alpha = pheromones[i,j]/max_value)
        plt.show()

    @staticmethod
    def _getDimensionsForSubplots(numPlots, max_columns=4):
        
        if numPlots >= max_columns:
            ncols = max_columns
            nrows = math.ceil(numPlots / max_columns)
            return nrows * 100 + ncols * 10
        else:
            return 100 + numPlots * 10

    def _formatTitlesArray(titles, n):

        for i in range(n - len(titles)):
            titles.append(i+1)


    """
    def _formatEdgeLabels(self, labels):
        for key in labels.keys():
            labels[key] = round(labels[key], 4)
    """