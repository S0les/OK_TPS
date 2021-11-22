from math import sqrt

def import_data():
    """
    import_data() -> list[tuple[float, float]], int
    """

    # <filename> has to be in tests/ folder
    filename = input("Read data from: ")

    cities = list()
    with open('tests/' + filename, 'r') as data_file:
        amount_of_cities = int(data_file.readline())
        for line in data_file:
            current_city = line.split()
            cities.append((float(current_city[1]), float(current_city[2])))
    return cities, amount_of_cities


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

    for index_start, start in enumerate(cities):

        order = [index_start]
        length = 0

        index_next, next_city, distance = get_closest(start, cities, order)
        length += distance
        order.append(index_next)

        while len(order) < amount_of_cities:
            index_next, next_city, distance = get_closest(next_city,
                                                          cities, order)
            length += distance
            order.append(index_next)

        distance = distance_squared(start, cities[order[amount_of_cities-1]])
        length += distance
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


def get_closest(city, cities, visited):
    """
    get_closest(city: tuple, cities: list[tuple[float, float]],
                    visited: list[int]) -> int, tuple, float


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

    for i, c in enumerate(cities):

        if i not in visited:
            distance = distance_squared(city, c)

            if distance < best_distance:
                closest_city = c
                index_closest_city = i
                best_distance = distance

    return index_closest_city, closest_city, best_distance


def distance_squared(city_a, city_b):
    """
    distance_squared(city_a: tuple[float, float],
                        city_b: tuple[float, float]) -> float

    A function to calculate distance between two nodes.

    Uses pythagorean theorem in order to calculate hypotenuse square.

    city_a: (x, y)

    city_b: (x, y)

    Returns square of distance between two cities.
    """

    leg_a = city_a[0] - city_b[0]
    leg_b = city_a[1] - city_b[1]

    square_distance = leg_a ** 2 + leg_b ** 2

    return sqrt(square_distance)
