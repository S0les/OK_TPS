from __future__ import annotations

import math

from abc import ABC, abstractmethod
from math import dist
from random import shuffle, uniform, randrange

from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation


class Solver(ABC):

    def __init__(self, cities: list[tuple[float, float]]) -> "Solver":
        self.cities = cities
        self.n = len(cities)
        self.order = list(range(self.n))
        self.adj = [
            [dist(i, j) if i != j else 0 for i in cities] for j in cities
        ]

    def get_total_dist(self, order: list[int]) -> float:
        """
        Get the total distance between the cities based off the ordering

        @param order: a list containing the order of cities to visit
        @return: the total distance of travelling in the order
        and returning to the start
        """

        if not order:
            return 0

        total = 0.0

        for i in range(len(order) - 1):
            total += self.adj[order[i + 1]][order[i]]
        total += self.adj[order[-1]][order[0]]

        return total

    @abstractmethod
    def solve(self) -> None:
        """
        Loop the get_next_order() method until the solver
        has the optimal (or what determines it) to be
        the optimal solution.
        """

    @abstractmethod
    def get_next_order(self) -> list[int]:
        """
        Returns the list of the next ordering of the path.
        @return an empty list if there's no more orderings.
        """

    @abstractmethod
    def get_best_order(self) -> list[int]:
        """
        @return the list of the current best ordering.
        """

###############################################################################


class SimulatedAnnealing(Solver):
    """
    Solver using the simmulated annealing algorithm
    """

    def __init__(self, cities, temperature: float = 100,
                 cooling_rate: float = 0.999):
        super().__init__(cities)
        shuffle(self.order)
        self.temperature = temperature
        self.cooling_rate = cooling_rate
        self.initial_temperature = self.temperature
        self.curr_dist = self.get_total_dist(self.order)
        self.iterations = 0
        self.max_repeats = int(10 * (1 / (1 - self.cooling_rate)))
        self.history = [self.order]

    def _get_order_from_history(self):
        if len(self.history) > 100:
            self.history = self.history[90:]
        elif len(self.history) > 10:
            self.history = self.history[9:]
        return self.history.pop(0)

    def _animation(self, i) -> None:
        if len(self.history) > 0:
            order = self._get_order_from_history()
            plt.cla()
            plt.title(label=f"Distance is: {self.get_total_dist(order):.2f}",
                      fontsize=18)
            order.append(order[0])
            cities = [self.cities[i] for i in order]
            cities_x, cities_y = zip(*cities)
            plt.plot(cities_x, cities_y)

        return

    def animated(self) -> None:
        if len(self.history) < 2:
            self.solve()
        plt.style.use("fivethirtyeight")
        ani = FuncAnimation(plt.gcf(), self._animation, interval=100)
        plt.tight_layout()
        manager = plt.get_current_fig_manager()
        manager.full_screen_toggle()
        plt.show()
        return

    def solve(self) -> None:
        """
        Continue cooling and finding distance until the optimal distance has
        not changed after self.max_repeats iterations
        """

        repeat = 0
        lowest_dist = float("inf")

        while repeat < self.max_repeats:
            self.get_next_order()
            if self.curr_dist < lowest_dist:
                repeat = 0
                lowest_dist = self.curr_dist
            else:
                repeat += 1

        return

    def get_next_order(self) -> list[int]:
        # Lower the temperature
        self.iterations += 1

        # TODO: tweak the temperature based off number of cities
        self.temperature = self.temperature * self.cooling_rate

        # Find new order
        a, b = self.get_two_cities()
        loss = self.get_swap_cost(a, b)
        if (loss <= 0 or self.temperature <= 0):
            prob = 0
        else:
            prob = math.exp(-loss / self.temperature)

        # If new distance shorter, or within probability then use it
        if loss <= 0 or uniform(0, 1) < prob:
            self.two_opt(a, b)
            self.curr_dist = self.get_total_dist(self.order)
            self.history.append(self.order)
        return self.order

    def get_best_order(self) -> list[int]:
        return self.order

    def get_two_cities(self) -> tuple[int, int]:
        """
        @return: two indexes between 0 and n, where the first is smaller
        """
        a = randrange(self.n)
        b = randrange(self.n)
        return (a, b) if a < b else (b, a)

    def get_swap_cost(self, a: int, b: int) -> float:
        """
        Given two indexes, return the cost if we were to reverse the
        ordering between the two indexes

        @param a: the lower index
        @param b: the higher index
        @return: the change in distance after the swap
        """

        n, order = self.n, self.order

        # Find which cities a and b are, and their next city
        a1, a2 = order[a], order[(a + 1) % n]
        b1, b2 = order[b], order[(b + 1) % n]

        # Find the current and new distance
        curr_dist = self.adj[a1][a2] + self.adj[b1][b2]
        new_dist = self.adj[a1][b1] + self.adj[a2][b2]

        return new_dist - curr_dist

    def two_opt(self, a: int, b: int) -> None:
        """
        Reverse the position between two values in the ordering,
        so that the path "uncrosses" itself

        @param a: the lower index
        @param b: the higher index
        """

        self.order = (self.order[:a + 1] + self.order[b:a:-1]
                      + self.order[b + 1:])

