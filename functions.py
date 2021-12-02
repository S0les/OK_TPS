from math import dist

def greedy_algorithm(cities, amount_of_cities):
    """
    greedy_algorithm(cities: list[tuple[float, float]],
                        amount_of_cities: int) -> list, int

    Implementation of greedy algorithm for TPS problem.

    In order to see every single path, uncomment the #print in the while loop.

    cities: [(float, float), (float, float)]

    Returns the best path and its length
    """

    best_order = list()
    best_length = float('inf')
    adj = [
        [dist(i, j) if i != j else 0 for i in cities] for j in cities
    ]

    for index_start in range(amount_of_cities):

        order = [index_start]
        not_visited = [x for x in range(amount_of_cities)
                       if x != index_start]
        length = 0
        start = index_start

        while len(order) < amount_of_cities:
            start, distance = get_closest(start, adj, not_visited)
            length += distance
            order.append(start)
            not_visited.remove(start)

        length += adj[index_start][order[-1]]
        order.append(index_start)

        if length < best_length:
            best_length = length
            best_order = order

        # Uncomment below
        # print("Try #%d" % index_start)
        # print(order)
        # print(length)
        # print()

    return best_order, best_length


def get_closest(city, adj, not_visited):
    """
    get_closest(city: int, visited: list[int]) -> int, tuple, float


    A function to choose the closest neighbour city.

    city: (float, float)

    cities: ((float, float), (float, float), (float, float))

    visited: [int, int, int]

    city - A city for which we're finding the closest neighbour
    cities - Tuple of all cities we have
    visited - A list of all visited cities

    Returns index of the closest city, coords of the closest city
    and distance to it.
    """

    best_distance = float('inf')

    for i in not_visited:

        distance = adj[city][i]

        if distance < best_distance:
            closest_city = i
            best_distance = distance

    return closest_city, best_distance
