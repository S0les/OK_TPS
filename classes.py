from __future__ import annotations

import copy
import math

from abc import ABC, abstractmethod
from math import dist
from random import shuffle, uniform, randrange

from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation


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
        self.curr_dist = self.get_total_dist(self.order)
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

    @staticmethod
    def init_func() -> None:
        """
        Initial function to be called before matplotlib.pyplot
        was build. Makes initial figure setup.
        """

        plt.style.use('fivethirtyeight')
        plt.gcf().set_size_inches(20.5, 10.5)
        plt.tight_layout()
        return

    @abstractmethod
    def get_frames(self) -> list[list[int]]:
        """
        Returns the frames, needed for the solution animation,
        as the two dimensional list of orders solution was achieved.
        @return two dimensional list of orders, an empty if there's none.
        """

    @abstractmethod
    def solve(self) -> None:
        """
        Loop the get_next_order() method until the solver
        has the optimal (or what determines it) to be
        the optimal solution.
        """

    @abstractmethod
    def animated(self, fullscreen: bool = True, repeat: bool = False,
                 save: bool = False) -> None:
        """
        Builds the matplotlib.pyplot figure with vizualization
        solution was achieved for the given algorithm.

        @param fullscreen: Toggles whether animation will be shown
        in a fullscreen or not
        @param repeat: Repeats animation if True
        @param save: Saves the vizualization as a .gif file instead of showing
        an animation
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
    Solver using the simmulated annealing algorithm with cities swap
    """

    def __init__(self, temperature: float = 100,
                 cooling_rate: float = 0.999):
        super().__init__()
        shuffle(self.order)
        self.temperature = temperature
        self.cooling_rate = cooling_rate
        self.initial_temperature = self.temperature
        self.curr_dist = self.get_total_dist(self.order)
        self.max_repeats = int(10 * (1 / (1 - self.cooling_rate)))
        self.history = [self.order]

    def solve(self) -> None:
        """
        Continue cooling and finding distance until the optimal distance has
        not changed after self.max_repeats iterations
        """

        if not self.cities:
            return

        repeat = 0
        lowest_dist = self.get_total_dist(self.order)
        best_order = self.order

        while repeat < self.max_repeats:
            self.get_next_order()
            if self.curr_dist < lowest_dist:
                repeat = 0
                lowest_dist = self.curr_dist
                best_order = self.order
                self.history.append(best_order)
            else:
                repeat += 1

        self.curr_dist = lowest_dist
        self.order = best_order
        return

    def get_next_order(self) -> list[int]:
        self.temperature = self.temperature * self.cooling_rate

        a, b = self.get_two_cities()
        loss = self.get_swap_cost(a, b)
        if (loss <= 0 or self.temperature <= 0):
            prob = 0
        else:
            prob = math.exp(-loss / self.temperature)

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

        a1, a2 = order[a], order[(a + 1) % n]
        b1, b2 = order[b], order[(b + 1) % n]

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

    def animated(self, fullscreen: bool = True, repeat: bool = False,
                 save: bool = False) -> None:

        frames = self.get_frames()

        if not frames:
            return

        def _animation(frame) -> None:
            plt.cla()
            plt.title(label=f"Distance is: \
                      {self.get_total_dist(frame):.2f}",
                      fontsize=18)
            cities = [self.cities[i] for i in frame]
            cities_x, cities_y = zip(*cities)
            plt.plot(cities_x, cities_y, marker='o',
                     markerfacecolor='indianred')
            return

        animation = FuncAnimation(plt.gcf(), _animation, frames=frames,
                                  interval=100, cache_frame_data=False,
                                  init_func=self.init_func, repeat=repeat)
        if fullscreen:
            manager = plt.get_current_fig_manager()
            manager.full_screen_toggle()

        if save:
            self.init_func()
            animation.save(f"SimulatedAnnealing_{self.n}_{self.curr_dist}.gif")
            print(f"Saved as SimulatedAnnealing_{self.n}"
                  + f"_{self.curr_dist}.gif!")
            plt.savefig(f"graphs/SimulatedAnnealing_{self.n}"
                        + f"{self.curr_dist}.svg")
        else:
            plt.show()
        return

    def get_frames(self) -> list[list[int]]:
        frames = []
        if not self.history:
            return frames

        history = copy.deepcopy(self.history)
        while(history):
            if len(history) > 100:
                history = history[90:]
            elif len(history) > 10:
                history = history[9:]
            frames.append(history.pop(0))
            frames[-1].append(frames[-1][0])
        return frames


###############################################################################


class SimulatedAnnealingV2(SimulatedAnnealing):
    def __init__(self, temperature=0.5, cooling_rate=0.9):
        super().__init__(temperature=temperature, cooling_rate=cooling_rate)
        self.nArray = [None] * 6

    def solve(self):
        if not self.cities:
            return

        # Limit of tries to make a change per temperature step
        self.nover = 100 * self.n

        # Limit of total changes per temperature step
        self.nlimit = 10 * self.n

        shuffle(self.order)
        self.curr_dist = self.get_total_dist(self.order)
        self.history = [self.order]

        for j in range(100):
            nsucc = 0
            for k in range(self.nover):
                not_in = self.get_slice()
                if uniform(0, 1) < 0.5:
                    self.nArray[2] = (self.nArray[1]
                                      + int(abs(not_in - 1)*uniform(0, 1)) + 1)
                    self.nArray[2] %= self.n
                    cost = self.transportcost()
                    answer = self.metrop(cost)
                    if answer:
                        nsucc += 1
                        self.curr_dist += cost
                        self.transport()
                        self.history.append(self.order)

                else:
                    cost = self.reversecost()
                    answer = self.metrop(cost)
                    if answer:
                        nsucc += 1
                        self.curr_dist += cost
                        self.reverse()
                        self.history.append(self.order)

                if nsucc >= self.nlimit:
                    break

            self.temperature *= self.cooling_rate
            if nsucc == 0:
                return

    def get_slice(self) -> int:
        """
        Gets random segment of order with amount of cities not in
        equal or more than two.

        @return amount of cities not in segment.
        """
        not_in = 0
        while (not_in < 2):
            self.nArray[0] = int(self.n * uniform(0, 1))
            self.nArray[1] = int((self.n - 1) * uniform(0, 1))
            if self.nArray[1] >= self.nArray[0]:
                self.nArray[1] += 1
            not_in = ((self.nArray[0] - self.nArray[1] + self.n - 1) % self.n)
        return not_in

    def reversecost(self):
        self.nArray[2] = (self.nArray[0] + self.n - 1) % self.n
        self.nArray[3] = (self.nArray[1] + 1) % self.n
        indexes = [None] * 4
        for i in range(4):
            indexes[i] = self.order[self.nArray[i]]
        cost = -self.adj[indexes[0]][indexes[2]]
        cost -= self.adj[indexes[1]][indexes[3]]
        cost += self.adj[indexes[0]][indexes[3]]
        cost += self.adj[indexes[1]][indexes[2]]
        return cost

    def reverse(self):
        to_swap = (1 + ((self.nArray[1] - self.nArray[0] + self.n)
                        % self.n))//2
        for i in range(to_swap):
            k = (self.nArray[0] + i) % self.n
            j = (self.nArray[1] - i + self.n) % self.n
            self.order[k], self.order[j] = self.order[j], self.order[k]
        return

    def transportcost(self):
        self.nArray[3] = (self.nArray[2] + 1) % self.n
        self.nArray[4] = (self.nArray[0] + self.n - 1) % self.n
        self.nArray[5] = (self.nArray[1] + 1) % self.n
        indexes = [None] * 6
        for i in range(6):
            indexes[i] = self.order[self.nArray[i]]

        cost = -self.adj[indexes[1]][indexes[5]]
        cost -= self.adj[indexes[0]][indexes[4]]
        cost -= self.adj[indexes[2]][indexes[3]]
        cost += self.adj[indexes[0]][indexes[2]]
        cost += self.adj[indexes[1]][indexes[3]]
        cost += self.adj[indexes[4]][indexes[5]]
        return cost

    def transport(self):
        neworder = copy.copy(self.order)
        m1 = (self.nArray[1] - self.nArray[0] + self.n) % self.n
        m2 = (self.nArray[4] - self.nArray[3] + self.n) % self.n
        m3 = (self.nArray[2] - self.nArray[5] + self.n) % self.n
        index = 0
        for i in range(m1 + 1):
            ii = (i + self.nArray[0]) % self.n
            neworder[index] = self.order[ii]
            index += 1
        for i in range(m2 + 1):
            ii = (i + self.nArray[3]) % self.n
            neworder[index] = self.order[ii]
            index += 1
        for i in range(m3 + 1):
            ii = (i + self.nArray[5]) % self.n
            neworder[index] = self.order[ii]
            index += 1
        self.order = copy.copy(neworder)
        return

    def metrop(self, cost):
        return cost < 0 or uniform(0, 1) < math.exp(-cost/self.temperature)


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

        for self.start in range(self.n):
            order = self.get_next_order()
            length = self.get_total_dist(order)
            if length < self.curr_dist:
                self.curr_dist = length
                self.order = order + [order[0]]
        return

    def get_next_order(self) -> list[int]:
        order = [self.start]
        not_visited = [x for x in range(self.n) if x != self.start]
        for _ in range(self.n - 1):
            self.start = self.get_closest_city(self.start, not_visited)
            order.append(self.start)
            not_visited.remove(self.start)
        return order

    def get_frames(self) -> list[list[int]]:
        frames = [[self.order[x] for x in range(i)] for i in range(1, self.n)]
        frames.append(self.order)
        frames.append(self.order + [self.order[0]])
        return frames

    def animated(self, fullscreen: bool = True, repeat: bool = False,
                 save: bool = False) -> None:
        frames = self.get_frames()

        def _animation(frame) -> None:

            plt.cla()
            plt.title(label="Greedy Solution with distance: " +
                      f"{self.get_total_dist(frame):.2f}.", fontsize=18)
            cities = [self.cities[i] for i in frame]
            cities_x, cities_y = zip(*cities)
            plt.plot(cities_x, cities_y, marker='o',
                     markerfacecolor="indianred")
            cities_x, cities_y = zip(*self.cities)
            plt.scatter(cities_x, cities_y, color="indianred")
            return

        animation = FuncAnimation(plt.gcf(), _animation, interval=100,
                                  init_func=self.init_func,
                                  cache_frame_data=False,
                                  frames=frames, repeat=repeat)
        if fullscreen:
            manager = plt.get_current_fig_manager()
            manager.full_screen_toggle()
        if save:
            self.init_func()
            animation.save(f"graphs/AdvancedGreedy_{self.n}"
                           + f"_{self.curr_dist}.gif")
            print(f"Saved as AdvancedGreedy_{self.n}_{self.curr_dist}.gif!")
            plt.savefig(f"graphs/AdvancedGreedy_{self.n}_{self.curr_dist}.svg")
        else:
            plt.show()
        return

    def get_closest_city(self, start_index: int,
                         not_visited: list[int]) -> int:
        """
        Get the index of the clossest city for the current one.
        """
        distance = float('inf')

        for city in not_visited:
            if self.adj[start_index][city] < distance:
                distance = self.adj[start_index][city]
                closest_city_index = city
        return closest_city_index

    def get_best_order(self):
        return self.order
