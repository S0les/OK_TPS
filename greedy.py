from functions import import_data, greedy_algorithm

if __name__ == '__main__':
    cities, amount_of_cities = import_data()
    best_route, best_length = greedy_algorithm(cities, amount_of_cities)

    print("Answer is:")
    print(best_route)
    print("\nWith distance:")
    print(best_length)
