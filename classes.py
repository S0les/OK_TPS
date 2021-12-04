from __future__ import annotations

import math

from abc import ABC, abstractmethod
from math import dist
from random import shuffle, uniform, randrange


class Solver(ABC):

    def __init__(self) -> "Solver":
        self.cities = list()
        self.n = 0
        self.order = list()
        self.adj = list()

    def import_data(self, filename: str) -> None:
        """
        Import cities from file and initialize them.

        Structure of file is next:
            amount_of_cities
            1 X1 Y1
            2 X2 Y2
            ...
            n Xn Yn

        @param filename: name of the file placed in tests/ folder
        @return None
        """

        with open('tests/' + filename, 'r') as data_file:
            data_file.readline()
            for line in data_file:
                current_city = line.split()
                current_city = (float(current_city[1]), float(current_city[2]))
                if current_city not in self.cities:
                    self.cities.append(current_city)
                    self.n += 1

        self.order = list(range(self.n))
        self.adj = [
            [dist(i, j) if i != j else 0 for i in self.cities]
            for j in self.cities
        ]
        return

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

    def __init__(self, temperature: float = 100,
                 cooling_rate: float = 0.999):
        super().__init__()
        shuffle(self.order)
        self.temperature = temperature
        self.cooling_rate = cooling_rate
        self.initial_temperature = self.temperature
        self.curr_dist = self.get_total_dist(self.order)
        self.iterations = 0
        self.max_repeats = int(10 * (1 / (1 - self.cooling_rate)))

    def solve(self) -> None:
        """
        Continue cooling and finding distance until the optimal distance has
        not changed after self.max_repeats iterations
        """

        if not self.cities:
            return

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


###############################################################################


class AdvancedGreedy(Solver):
    """
    Solver using the greedy algorithm for every possible starting point
    """

    def __init__(self):
        super().__init__()
        self.curr_dist = float('inf')

    def solve(self) -> None:
        """
        For every starting city continue search for the closest city until
        eleminate every.
        """

        for start_index in range(self.n):
            order = [start_index]
            not_visited = [x for x in range(self.n)
                           if x != start_index]
            length = 0
            start = start_index

            for _ in range(self.n - 1):
                start, distance = self.get_closest_city(start, not_visited)
                length += distance
                order.append(start)
                not_visited.remove(start)

            length += self.adj[start_index][order[-1]]
            order.append(start_index)

            if length < self.curr_dist:
                self.curr_dist = length
                self.order = order
        return

    def get_closest_city(self, start_index: int,
                         not_visited: list[int]) -> [int, int]:
        """
        Get the index of the clossest city for the current one.
        """
        distance = float('inf')

        for city in not_visited:
            if self.adj[start_index][city] < distance:
                distance = self.adj[start_index][city]
                closest_city_index = city
        return closest_city_index, distance

    def get_best_order(self):
        return self.order

    def get_next_order(self):
        return list()
