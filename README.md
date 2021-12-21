Solving the Travelling Salesman Problem in Python 
===================
## Implemented techniques
* __Simulated Annealing__
* __Advanced Greedy__

## Definition
The Travelling Salesman Problem is a well known NP-Hard problem. Given a list of cities, find the shortest path that visits all cities once and returns to the start.

Simulated annealing is a well known stochastic method for solving optimisation problems and is a well known non-exact algorithm for solving the TSP. However, it's effectiveness is dependent on initial parameters such as the starting temperature and cooling rate which is often chosen empirically.

## Usage

### Executing
```sh
import classes

# Initializes class solver with given temperature and cooling rate
solver = classes.SimulatedAnnealing(temperature=100, cooling_rate=0.9)

# Imports dataset from tests/ folder
solver.import_data('tsp250.txt')

# Solves the problem
solver.solve()

# Final order and distance
order = solver.get_best_order()
distance = solver.get_total_dist(order)

# Vizualization of the way solution was achieved 
solver.animated(fullscreen=True, repeat=False, save=False)
```

### Animation saving
If `save` parameter is `True` for `solver.animated()` method, instead of showing, animation will be saved to the current folder with the name pattern: `<Solver Name>_<Amount of cities>_<Total distance>` as a .gif and .svg files.
