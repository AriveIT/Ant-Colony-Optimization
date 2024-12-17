# Ant  Colony Optimization

Ant Colony Optimization is a randomized algorithm that gives approximate solutions to the travelling salesman problem that simulates ant foraging behaviour in finding the shortest path to food.

Ant's take random paths through the graph, laying pheromones behind them. The shorter their path ends up, the stronger their pheromones are.
Subsequent ants will be more likely to cross edges with stronger pheromones. That being said, some randomness is injected to encourage constant exploration, in order to test new possible routes.

Results.png shows a screenshot of applying this algorithm, along with the nearest neighbours algorithm to an example graph with 20 vertices
